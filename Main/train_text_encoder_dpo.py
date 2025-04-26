import argparse
import contextlib
import io
import logging
import math
import os
import random
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from functools import partial
import peft
import gc # 가비지 컬렉션
import glob # 체크포인트 삭제용

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

try:
	import wandb
	has_wandb = True
except ImportError:
	has_wandb = False

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, IterableDataset # IterableDataset 임포트
from huggingface_hub import create_repo, upload_folder
from packaging import version

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig 

import diffusers
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.loaders import LoraLoaderMixin 


try:
	from dpo_data import preprocess_data, collate_fn, prepare_dataset
	from dpo_helpers import (
		extract_lora_weights,
		log_validation,
		VALIDATION_PROMPTS,
		check_positive
	)
	from model_load import load_models_and_configure
	from tokenizer_load import load_tokenizers
	from training_loop import training_loop, compute_time_ids
	from optimizer_load import create_optimizer_and_scheduler
except ImportError as e:
	print(f"Error importing from helper modules (dpo_data.py, dpo_helpers.py): {e}")
	print("Ensure these files exist in the same directory as the training script.")
	exit(1)


# 최소 버전 체크
check_min_version("0.25.0") # 또는 더 높은 버전 권장

# 메인 로거 설정 (Accelerator 이후 초기화)
logger: Optional[logging.Logger] = None

# --- 인수 파서 정의 ---
def parse_args(input_args=None):
	parser = argparse.ArgumentParser(description="SDXL Text Encoder LoRA DPO training script using Image Preferences.")
	# --- 기본 경로 ---
	parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to the local Stable Diffusion checkpoint (.safetensors, .ckpt) or directory.") # 설명 수정
	parser.add_argument("--base_model_id_for_tokenizer", type=str, required=True, help="Base model ID/path for loading tokenizers (e.g., stabilityai/stable-diffusion-xl-base-1.0).")
	parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None, help="Optional: Path to a separate VAE checkpoint.")
	parser.add_argument("--pretrained_lora_path", type=str, required=True, help="Path to the local Lora")
	# --- 데이터 경로 ---
	parser.add_argument("--dataset_path", type=str, required=True, help="Path to the preference dataset in .jsonl format.")
	parser.add_argument("--image_data_root", type=str, required=True, help="Root directory containing image folders/files.")
	# --- 출력 및 기타 경로 ---
	parser.add_argument("--output_dir", type=str, default="diffusion-dpo-lora-te-img-v2")
	parser.add_argument("--cache_dir", type=str, default=None)
	parser.add_argument("--logging_dir", type=str, default="logs")
	# --- 설정값 ---
	parser.add_argument("--revision", type=str, default=None)
	parser.add_argument("--variant", type=str, default=None)
	parser.add_argument("--run_validation", default=False, action="store_true")
	parser.add_argument("--validation_steps", type=int, default=500)
	parser.add_argument("--max_train_samples", type=int, default=None)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--resolution_width", type=int, default=512, help="Target width for image processing.")
	parser.add_argument("--resolution_height", type=int, default=512, help="Target height for image processing.")
	parser.add_argument("--vae_encode_batch_size", type=int, default=1) # Latent 생성 시 사용 (현재 코드에서는 직접 사용 안 함)
	parser.add_argument("--no_hflip", action="store_true") # dpo_data.py 에서 사용
	parser.add_argument("--random_crop", default=False, action="store_true") # dpo_data.py 에서 사용 (현재 비활성)
	parser.add_argument("--train_batch_size", type=int, default=1)
	parser.add_argument("--num_train_epochs", type=int, default=1)
	parser.add_argument("--max_train_steps", type=int, default=None)
	parser.add_argument("--checkpointing_steps", type=int, default=500)
	parser.add_argument("--checkpoints_total_limit", type=int, default=2)
	parser.add_argument("--resume_from_checkpoint", type=str, default=None)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
	parser.add_argument("--gradient_checkpointing", action="store_true")
	parser.add_argument("--beta_dpo", type=float, default=0.1)
	parser.add_argument("--learning_rate", type=float, default=5e-7)
	parser.add_argument("--lr_scheduler", type=str, default="cosine")
	parser.add_argument("--lr_warmup_steps", type=int, default=100)
	parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1)
	parser.add_argument("--lr_scheduler_power", type=float, default=1.0)
	parser.add_argument("--dataloader_num_workers", type=int, default=0) # 0으로 고정 권장
	parser.add_argument("--use_8bit_adam", action="store_true")
	parser.add_argument("--adam_beta1", type=float, default=0.9)
	parser.add_argument("--adam_beta2", type=float, default=0.999)
	parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
	parser.add_argument("--adam_epsilon", type=float, default=1e-08)
	parser.add_argument("--max_grad_norm", default=1.0, type=float)
	parser.add_argument("--num_vae_workers", type=check_positive, default=1,help="Number of workers for preparing VAE latents (0 for sequential main process).")
	parser.add_argument("--skip_latent_check", action="store_true", help="Skip checking for latent caches if you are sure they exist.") # Latent 체크 건너뛰기 옵션
	# --- Hub 관련 ---
	parser.add_argument("--push_to_hub", action="store_true")
	parser.add_argument("--hub_token", type=str, default=None)
	parser.add_argument("--hub_model_id", type=str, default=None)
	# --- 기타 ---
	parser.add_argument("--allow_tf32", action="store_true")
	parser.add_argument("--report_to", type=str, default="tensorboard")
	parser.add_argument("--mixed_precision", type=str, default="fp16")
	parser.add_argument("--is_turbo", action="store_true") # 사용되지 않음 (참고용)
	parser.add_argument("--tracker_project_name", type=str, default="sdxl_te_dpo_img")

	# --- GUI 호환용 (사용 안 함) ---
	parser.add_argument("--base_model_id", type=str, default=None)
	parser.add_argument("--dpo_loss_type", type=str, default="sigmoid") # 현재 sigmoid만 구현
	parser.add_argument("--max_token_length", type=int, default=77) # Tokenizer 기본값 사용
	parser.add_argument("--noise_offset", type=float, default=0.0) # 사용되지 않음
	parser.add_argument("--dataset_split_name", type=str, default="train", help="Dataset split to use (e.g., 'train', 'validation').")
	parser.add_argument("--use_streaming", action="store_true", help="Load dataset in streaming mode (useful for large datasets).") # 스트리밍 옵션 추가

	if input_args is not None: args = parser.parse_args(input_args)
	else: args = parser.parse_args()

	# --- 인수 유효성 검사 ---
	if not os.path.exists(args.dataset_path):
		raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
	if not os.path.isdir(args.image_data_root):
		raise NotADirectoryError(f"Image data root not found or not a directory: {args.image_data_root}")

	# pretrained_model_name_or_path 와 base_model_id 동기화 (GUI 호환성)
	if args.base_model_id and args.base_model_id != args.pretrained_model_name_or_path:
		 print(f"Warning: Both base_model_id ({args.base_model_id}) and pretrained_model_name_or_path ({args.pretrained_model_name_or_path}) provided. Using pretrained_model_name_or_path.")
	elif args.base_model_id and not args.pretrained_model_name_or_path:
		 args.pretrained_model_name_or_path = args.base_model_id

	if args.is_turbo:
		if "turbo" not in args.pretrained_model_name_or_path.lower():
			 print("Warning: --is_turbo flag is set, but 'turbo' not found in pretrained_model_name_or_path.")

	# 스트리밍 사용 시 max_train_steps 필수 확인
	if args.use_streaming and args.max_train_steps is None:
		raise ValueError("--max_train_steps must be specified when using --use_streaming.")

	return args

# --- 초기화 함수 ---
def initialize_environment(args):
	global logger
	logging_dir = Path(args.output_dir, args.logging_dir)
	accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
	accelerator = Accelerator(
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		mixed_precision=args.mixed_precision,
		log_with=args.report_to,
		project_config=accelerator_project_config,
	)

	logger = get_logger(__name__, log_level="INFO")
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)

	logger.info(accelerator.state, main_process_only=False)
	if accelerator.is_local_main_process:
		transformers.utils.logging.set_verbosity_warning()
		diffusers.utils.logging.set_verbosity_info()
	else:
		transformers.utils.logging.set_verbosity_error()
		diffusers.utils.logging.set_verbosity_error()

	if args.seed is not None:
		set_seed(args.seed)
		accelerator.print(f"Set seed for reproducibility: {args.seed}")

	if args.allow_tf32:
		torch.backends.cuda.matmul.allow_tf32 = True
		logger.info("Enabled TF32 for matmul.")

	# Hub 설정
	repo_id = None
	if accelerator.is_main_process and args.push_to_hub:
		if args.hub_token is None:
			logger.warning("No hub_token provided. Will try to use default login.")
		if not args.hub_model_id:
			args.hub_model_id = Path(args.output_dir).name
		try:
			repo_id = create_repo(repo_id=args.hub_model_id, exist_ok=True, token=args.hub_token).repo_id
			logger.info(f"Push-to-hub enabled for repository: {repo_id}")
		except Exception as e:
			logger.error(f"Failed to create Hub repository: {e}")
			args.push_to_hub = False # Hub 생성 실패 시 비활성화

	# 출력 디렉토리 생성
	if accelerator.is_main_process and args.output_dir:
		os.makedirs(args.output_dir, exist_ok=True)
		accelerator.print(f"Output directory created: {args.output_dir}")

	return accelerator, repo_id

# --- 메인 실행 함수 ---
def main(args):
	# --- 1. 환경 초기화 ---
	accelerator, repo_id = initialize_environment(args)
	if accelerator is None: return

	# --- 2. Latent 캐시 확인/준비 ---
	try:
		with open(args.dataset_path, 'r', encoding='utf-8') as f:
			first_line = f.readline()
			sample = json.loads(first_line)
			for key in ["image_0_relative_path", "image_1_relative_path"]:
				img_rel_path = sample.get(key)
				if img_rel_path:
					cache_path = os.path.splitext(os.path.join(args.image_data_root, img_rel_path))[0] + ".pt"
					if not os.path.exists(cache_path):
						logger.warning(f"Latent cache missing for first sample image: {cache_path}. Ensure latents are prepared before running.")
						break 
	except Exception as e:
		logger.warning(f"Could not perform quick latent cache check: {e}")

	# --- 3. Tokenizer 로드 ---
	tokenizer_one, tokenizer_two = load_tokenizers(args.base_model_id_for_tokenizer, args.revision, args.cache_dir, accelerator)
	if tokenizer_one is None: logger.error("Failed to load tokenizers."); return

	# --- 4. 모델 로드 및 설정 (LoRA 포함) ---
	if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
	elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16
	else: weight_dtype = torch.float32
	unet, text_encoder_one, text_encoder_two, noise_scheduler, params_to_optimize = load_models_and_configure(
		args, weight_dtype, accelerator.device, accelerator
	)
	if unet is None or params_to_optimize is None:
		logger.error("Failed to load models or configure LoRA."); return
	
	# --- 5. 데이터셋 및 DataLoader 준비 ---
	train_dataloader, train_dataset_processed = prepare_dataset(args, accelerator, tokenizer_one, tokenizer_two)
	if train_dataloader is None: logger.error("Failed to prepare dataset or dataloader."); return

	# --- 6. 옵티마이저 및 스케줄러 생성 (및 학습 단계 계산) ---
	overrode_max_train_steps = False
	num_update_steps_per_epoch = float('inf')
	if not args.use_streaming:
		try:
			num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
			if num_update_steps_per_epoch == 0: num_update_steps_per_epoch = 1
		except TypeError: logger.error("Could not determine DataLoader length."); return

		if args.max_train_steps is None:
			args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
			overrode_max_train_steps = True # 자동 계산됨 플래그
			accelerator.print(f"Calculated max_train_steps: {args.max_train_steps} ({args.num_train_epochs} epochs)")
		else:
			args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
			logger.info(f"Using provided max_train_steps: {args.max_train_steps}, calculated epochs: {args.num_train_epochs}")


	optimizer, lr_scheduler = create_optimizer_and_scheduler(
		args, params_to_optimize, accelerator, num_update_steps_per_epoch
	)
	if optimizer is None or lr_scheduler is None: # 생성 실패 시 종료
		logger.error("Failed to create optimizer or scheduler."); return

	# --- 7. Accelerator Prepare ---
	try:
		optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
			optimizer, train_dataloader, lr_scheduler
		)
		accelerator.print("Optimizer, DataLoader, LRScheduler prepared with Accelerator.")
	except Exception as e: logger.error(f"Error during accelerator.prepare: {e}", exc_info=True); return

	# --- 8. 체크포인트 재개 ---
	initial_global_step = 0
	first_epoch = 0
	if args.resume_from_checkpoint:
		# (이전과 동일한 체크포인트 재개 로직)
		if args.resume_from_checkpoint != "latest": path = args.resume_from_checkpoint
		else:
			dirs = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), key=lambda x: int(os.path.basename(x).split('-')[1]))
			path = dirs[-1] if dirs else None
		if path and os.path.isdir(path):
			accelerator.print(f"Resuming from checkpoint {path}")
			try:
				accelerator.load_state(path) # 옵티마이저, 스케줄러 등 상태 복원
				global_step_resume = int(os.path.basename(path).split("-")[1])
				initial_global_step = global_step_resume
				if num_update_steps_per_epoch != float('inf'): first_epoch = global_step_resume // num_update_steps_per_epoch
				accelerator.print(f"Resumed state from step {initial_global_step}, epoch {first_epoch}.")
				# LoRA 재로드는 model_load 에서 이미 처리되었으므로 여기서 다시 할 필요 없음
				# 단, save_state 가 LoRA 파라미터 자체를 저장/로드 하는지 확인 필요
				# 만약 save_state가 LoRA 파라미터를 덮어쓴다면, load_state 후에 다시 LoRA를 로드해야 할 수 있음
				# 일단은 load_state 만으로 충분하다고 가정. Hook 에서 LoRA 저장을 분리했으므로 괜찮을 가능성 높음.
				# 필요시 아래 LoRA 재로드 코드 활성화:
				# lora_resume_path = os.path.join(path, "pytorch_lora_weights.safetensors")
				# if os.path.exists(lora_resume_path): ... (LoRA 재로드 로직) ...
			except Exception as e: logger.error(f"Failed to resume from checkpoint: {e}", exc_info=True); initial_global_step = 0; first_epoch = 0
		else: logger.warning(f"Checkpoint '{args.resume_from_checkpoint}' not found or invalid. Starting fresh.")


	# --- 9. 학습 루프 실행 ---
	# prepare 후 dataloader 길이가 바뀔 수 있으므로 다시 계산
	if not args.use_streaming:
		try:
			num_update_steps_per_epoch_prepared = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
			if num_update_steps_per_epoch_prepared == 0: num_update_steps_per_epoch_prepared = 1
			if overrode_max_train_steps: # max_steps가 자동계산된 경우, prepare 후 길이로 다시 계산
				args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch_prepared
				logger.info(f"Recalculated max_train_steps after prepare: {args.max_train_steps}")
			calculated_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch_prepared)
			if args.num_train_epochs != calculated_epochs:
				logger.warning(f"Epoch mismatch after prepare. Using calculated epochs: {calculated_epochs}")
				args.num_train_epochs = calculated_epochs
			num_update_steps_per_epoch = num_update_steps_per_epoch_prepared # 최종값 업데이트
		except TypeError: logger.warning("Could not determine wrapped DataLoader length after prepare.")

	final_global_step = training_loop(
		args, accelerator,
		unet, text_encoder_one, text_encoder_two, noise_scheduler,
		optimizer, params_to_optimize, lr_scheduler,
		train_dataloader,
		weight_dtype,
		initial_global_step, first_epoch,
		num_update_steps_per_epoch_prepared,
	)


	# --- 10. 최종 처리 (저장, 검증, Hub 업로드) ---
	accelerator.wait_for_everyone()
	if accelerator.is_main_process:
		# 최종 LoRA 저장
		logger.info("Saving final Text Encoder LoRA weights...")
		try:
			# <<< extract_lora_weights 사용 (text_encoder 객체 필요) >>>
			lora_state_dict = extract_lora_weights(text_encoder_one, text_encoder_two, logger)
			if lora_state_dict:
				from safetensors.torch import save_file
				save_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
				save_file(lora_state_dict, save_path, metadata={"format": "pt"})
				logger.info(f"Saved final Text Encoder LoRA weights to: {save_path}")
			else: logger.warning("No final Text Encoder LoRA weights found to save!")
		except Exception as e: logger.error(f"Error saving final TE LoRA weights: {e}", exc_info=True)

		# 최종 검증
		if args.run_validation:
			logger.info("Running final validation...")
			try:
				# <<< log_validation 호출 (unet, te 객체 필요) >>>
				log_validation(args, text_encoder_one, text_encoder_two, unet, accelerator, weight_dtype, final_global_step, True, logger, VALIDATION_PROMPTS)
			except Exception as e: logger.error(f"Final validation failed: {e}", exc_info=True)

		# Hub 업로드
		if args.push_to_hub and repo_id:
			logger.info(f"Pushing final model and logs to Hub repo: {repo_id}")
			try:
				upload_folder(repo_id=repo_id, folder_path=args.output_dir, commit_message="End of DPO TE LoRA training", ignore_patterns=["step_*", "epoch_*", "checkpoint-*/"], token=args.hub_token)
				logger.info(f"Successfully pushed to Hub repo: {repo_id}")
			except Exception as e: logger.error(f"Failed to push to Hub: {e}")

	accelerator.end_training()
	accelerator.print("Script finished successfully.")


if __name__ == "__main__":
	args = parse_args()
	main(args)
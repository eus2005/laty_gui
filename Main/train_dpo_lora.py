# --- START OF FILE train_dpo_lora.py ---

import argparse
import json
import os
import random
import math
from datetime import datetime
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.loaders import LoraLoaderMixin, AttnProcsLayers # LoRA 로딩/저장
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0 # 타입 확인용 (현재 미사용)
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import bitsandbytes as bnb # 8-bit Adam
from peft import PeftModel

# 로거 설정
logger = get_logger(__name__)

# --- 데이터셋 클래스 정의 ---
# !!! 중요: 현재 데이터셋은 (prompt, score) 형식입니다.
# 이상적인 DPO는 (prompt, chosen, rejected) 쌍을 사용합니다.
# 아래 코드는 score를 이용해 DPO Loss를 근사합니다.
class DPOPreferenceDataset(Dataset):
	def __init__(self, jsonl_path):
		self.data = []
		try:
			with open(jsonl_path, 'r', encoding='utf-8') as f:
				for line_num, line in enumerate(f):
					try:
						entry = json.loads(line.strip())
						prompt = entry.get("prompt_text_reference")
						# IMPORTANT: Use 'comparison_score' (+1, -1, 0) saved by the node
						score = entry.get("comparison_score")

						if prompt is not None and score is not None:
							try:
								score_float = float(score)
								# Allow 0 score (equal preference) as well
								if -1.0 <= score_float <= 1.0:
									# We only need prompt and score for the trainer's loss logic
									self.data.append({"prompt": prompt, "score": score_float})
								else:
									logger.warning(f"L{line_num+1}: Skipping entry with score outside [-1, 1]: {score_float} in {jsonl_path}")
							except (ValueError, TypeError):
								logger.warning(f"L{line_num+1}: Skipping entry with non-numeric score: {score} in {jsonl_path}")
						else:
							# Log missing essential fields
							missing = []
							if prompt is None: missing.append("'prompt_text_reference'")
							if score is None: missing.append("'comparison_score'")
							logger.warning(f"L{line_num+1}: Skipping entry with missing fields: {', '.join(missing)} in {jsonl_path}")

					except json.JSONDecodeError:
						logger.warning(f"L{line_num+1}: Skipping invalid JSON line in {jsonl_path}: {line.strip()}")
					except Exception as e: # Catch other potential errors during parsing
						logger.warning(f"L{line_num+1}: Error processing line in {jsonl_path}: {e} - Line: {line.strip()}")

			if not self.data:
				 raise ValueError(f"No valid comparison data loaded from the dataset file: {jsonl_path}")
			logger.info(f"Loaded {len(self.data)} valid preference entries from {jsonl_path}")

		except FileNotFoundError:
			raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
		except Exception as e:
			raise RuntimeError(f"Error loading dataset {jsonl_path}: {e}")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		# Return prompt and the comparison score (+1, -1, or 0)
		return self.data[idx]

# --- Log Probability 계산 함수 (MSE 기반 근사) ---
def calculate_log_probs(predicted_noise: torch.Tensor, actual_noise: torch.Tensor) -> torch.Tensor:
	"""
	Calculates an approximate log probability for each item in the batch based on MSE loss.
	logp ≈ -MSE(predicted_noise, actual_noise)
	Args:
		predicted_noise: Noise predicted by the model (batch_size, channels, height, width).
		actual_noise: The actual noise added to the latents (batch_size, channels, height, width).
	Returns:
		Tensor of shape (batch_size,) containing the approximate log probability for each batch item.
	"""
	mse = F.mse_loss(predicted_noise.float(), actual_noise.float(), reduction="none")
	# 각 배치 아이템별로 채널/높이/너비 차원에 대해 평균 MSE 계산
	mse_per_item = torch.mean(mse, dim=list(range(1, mse.ndim))) # B,C,H,W -> B
	# Negative MSE를 log probability의 근사치로 사용
	log_probs = -mse_per_item
	return log_probs


# --- 메인 학습 함수 ---
def main(args):
	# *** 1. Accelerator 초기화 ***
	logging_dir = os.path.join(args.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
	accelerator = Accelerator(
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		mixed_precision=args.mixed_precision,
		log_with="tensorboard",
		project_dir=logging_dir,
		# --- REMOVED: max_grad_norm=args.max_grad_norm, ---
	)
	logger.info(f"Accelerator state initialized. Device: {accelerator.device}, Distributed: {accelerator.distributed_type}, Mixed Precision: {args.mixed_precision}")
	logger.warning("This script uses scores to approximate DPO loss. For best results, use a dataset with chosen/rejected pairs.")

	# *** 2. 경로 및 기본 검사 ***
	# ... (기존 경로 검사 코드 유지) ...
	if not os.path.isfile(args.pretrained_checkpoint_path):
		logger.error(f"오류: 지정된 체크포인트 파일을 찾을 수 없습니다: {args.pretrained_checkpoint_path}")
		sys.exit(1)
	if not os.path.exists(args.pretrained_lora_path):
		 logger.error(f"오류: 지정된 기존 LoRA 파일을 찾을 수 없습니다: {args.pretrained_lora_path}")
		 sys.exit(1)

	# <<< Added resolution checks
	if args.resolution_width % 8 != 0 or args.resolution_height % 8 != 0:
		logger.error(f"오류: 해상도 너비 ({args.resolution_width})와 높이 ({args.resolution_height})는 8의 배수여야 합니다.")
		sys.exit(1)
	if args.resolution_width <= 0 or args.resolution_height <= 0:
		 logger.error(f"오류: 해상도 너비 ({args.resolution_width})와 높이 ({args.resolution_height})는 0보다 커야 합니다.")
		 sys.exit(1)
	logger.info(f"Using training resolution: {args.resolution_width}x{args.resolution_height}")


	logger.info(f"Using base model architecture/tokenizer from: '{args.base_model_id}'")
	logger.info(f"Loading weights from checkpoint: '{args.pretrained_checkpoint_path}'")
	logger.info(f"Fine-tuning LoRA: '{args.pretrained_lora_path}'")

	# *** 3. 나머지 초기화 ***
	if accelerator.is_main_process:
		os.makedirs(args.output_dir, exist_ok=True)
		logger.info(f"Output directory: {args.output_dir}")
		logger.info(f"Logging directory (Tensorboard): {logging_dir}")
	if args.seed is not None:
		set_seed(args.seed)
		logger.info(f"Set random seed to: {args.seed}")

	# --- 모델 및 토크나이저 로드 ---
	logger.info("Loading pipeline from single file checkpoint...")
	try:
		# pipe: 학습 대상 모델 (Policy)
		pipe = StableDiffusionXLPipeline.from_single_file(
			args.pretrained_checkpoint_path,
			torch_dtype=torch.float16,
			variant="fp16", use_safetensors=True,
		)
		logger.info("Policy pipeline loaded successfully from checkpoint.")
		unet = pipe.unet
		vae = pipe.vae
		text_encoder_one = pipe.text_encoder
		text_encoder_two = pipe.text_encoder_2
		tokenizer_one = pipe.tokenizer
		tokenizer_two = pipe.tokenizer_2
		scheduler = pipe.scheduler
	except Exception as e:
		 logger.error(f"오류: 체크포인트/베이스 모델 로딩 실패: {e}")
		 sys.exit(1)

	# --- 모델 가중치 동결 (LoRA 파라미터 제외) ---
	unet.requires_grad_(False)
	text_encoder_one.requires_grad_(False)
	text_encoder_two.requires_grad_(False)
	if vae: vae.requires_grad_(False)

	# --- 기존 LoRA 가중치 로드 (Policy 모델에) ---
	logger.info(f"Loading existing LoRA weights into Policy UNet: {args.pretrained_lora_path}")
	try:
		pipe.load_lora_weights(
			os.path.dirname(args.pretrained_lora_path),
			weight_name=os.path.basename(args.pretrained_lora_path),
			adapter_name="dpo_lora" # 학습 대상 LoRA
		)
		logger.info(f"Successfully loaded LoRA weights adapter 'dpo_lora' into Policy UNet.")
	except Exception as e:
		 logger.error(f"오류: Policy 모델 LoRA 로딩 실패: {e}")
		 sys.exit(1)

	# --- 학습 대상 파라미터 설정 (Policy UNet의 LoRA) ---
	params_to_optimize = []
	for name, param in unet.named_parameters(): # Policy UNet 사용
		if param.requires_grad:
			params_to_optimize.append(param)
	logger.info(f"Number of trainable LoRA parameters found in Policy UNet: {len(params_to_optimize)}")
	if not params_to_optimize:
		 logger.error("오류: 학습 가능한 LoRA 파라미터를 찾을 수 없습니다.")
		 sys.exit(1)

	# --- 그래디언트 체크포인팅 ---
	if args.gradient_checkpointing:
		unet.enable_gradient_checkpointing() # Policy UNet에 적용
		logger.info("Gradient checkpointing enabled for Policy UNet.")

	# --- 옵티마이저 설정 ---
	optimizer_cls = torch.optim.AdamW
	if args.use_8bit_adam:
		try:
			optimizer_cls = bnb.optim.AdamW8bit
			logger.info("Using 8-bit AdamW optimizer.")
		except ImportError:
			 logger.warning("bitsandbytes not found. Falling back to regular AdamW.")
		except AttributeError:
			 logger.warning("bitsandbytes 8-bit AdamW not found. Falling back to regular AdamW.")
	else:
		logger.info("Using regular AdamW optimizer.")

	optimizer = optimizer_cls(
		params_to_optimize,
		lr=args.learning_rate,
		betas=(args.adam_beta1, args.adam_beta2),
		weight_decay=args.adam_weight_decay,
		eps=args.adam_epsilon,
	)

	# --- 데이터셋 및 데이터로더 ---
	try:
		logger.info(f"Loading dataset from: {args.dataset_path}")
		train_dataset = DPOPreferenceDataset(args.dataset_path) # Use the updated dataset class
		train_dataloader = DataLoader(
			train_dataset,
			batch_size=args.train_batch_size,
			shuffle=True,
			collate_fn=lambda batch: { # Collate prompt and score
				"prompt": [item["prompt"] for item in batch],
				# Ensure score is float32 tensor
				"score": torch.tensor([item["score"] for item in batch], dtype=torch.float32)
			},
			num_workers=args.dataloader_num_workers
		)
		logger.info(f"Dataset loaded with {len(train_dataset)} preference samples.")
	except Exception as e:
		 logger.error(f"데이터셋 로딩 실패: {e}")
		 sys.exit(1)

	# --- 레퍼런스 모델 준비 (베이스 + 원본 LoRA, 동결) ---
	logger.info("Creating Reference Model (Base Arch + Original LoRA)...")
	try:
		# ref_pipe: 비교 대상 레퍼런스 모델
		ref_pipe = StableDiffusionXLPipeline.from_single_file(
			args.pretrained_checkpoint_path, # 동일한 베이스 체크포인트 사용
			torch_dtype=torch.float16,
			variant="fp16", use_safetensors=True
		)
		# 레퍼런스 파이프라인에도 원본 LoRA 로드
		ref_pipe.load_lora_weights(
			os.path.dirname(args.pretrained_lora_path), # 동일한 LoRA 경로
			weight_name=os.path.basename(args.pretrained_lora_path),
			adapter_name="original_lora" # 다른 이름 사용 가능
		)
		ref_pipe.unet.requires_grad_(False) # UNet 전체 동결
		ref_pipe.text_encoder.requires_grad_(False)
		ref_pipe.text_encoder_2.requires_grad_(False)
		if ref_pipe.vae: ref_pipe.vae.requires_grad_(False)

		ref_pipe.to(accelerator.device)
		ref_pipe.set_progress_bar_config(disable=True)
		reference_unet = ref_pipe.unet # 레퍼런스 UNet 추출
		logger.info("Reference pipeline with original LoRA created and frozen successfully.")
	except Exception as e:
		logger.error(f"레퍼런스 파이프라인 생성 실패: {e}")
		sys.exit(1)


	# --- Accelerator 준비 ---
	# unet (Policy), optimizer, dataloader 준비
	unet, optimizer, train_dataloader = accelerator.prepare(
		unet, optimizer, train_dataloader
	)
	# 나머지 컴포넌트들도 GPU/분산 환경으로 이동
	text_encoder_one.to(accelerator.device)
	text_encoder_two.to(accelerator.device)
	if vae: vae.to(accelerator.device)
	# reference_unet은 이미 .to(accelerator.device) 완료됨

	# --- 학습 루프 ---
	total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
	max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

	logger.info("***** Running DPO Fine-tuning *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(f"  Num Epochs = {args.num_train_epochs}")
	logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
	logger.info(f"  Total train batch size = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	logger.info(f"  Total optimization steps = {max_train_steps}")
	logger.info(f"  Training Resolution = {args.resolution_width}x{args.resolution_height}") # <<< Added log

	global_step = 0
	progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
	progress_bar.set_description("Steps")

	beta = args.dpo_beta # DPO 하이퍼파라미터

	for epoch in range(args.num_train_epochs):
		epoch_loss = 0.0
		epoch_policy_logps_mean = 0.0
		epoch_ref_logps_mean = 0.0
		epoch_logits_mean = 0.0

		for step, batch in enumerate(train_dataloader):
			unet.train() # Policy UNet을 학습 모드로 설정

			with accelerator.accumulate(unet):
				prompts = batch["prompt"]
				scores = batch["score"].to(accelerator.device) # 배치 점수 (타겟 선호도)

				# Text Embeddings 생성 (Policy 모델의 Text Encoder 사용)
				with torch.no_grad(): # Text Encoder는 동결 상태
					prompt_embeds_list = []
					for tokenizer, text_encoder in zip([tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]):
						text_inputs = tokenizer(
							prompts, padding="max_length", max_length=tokenizer.model_max_length,
							truncation=True, return_tensors="pt",
						)
						text_input_ids = text_inputs.input_ids.to(accelerator.device)
						prompt_embeds_out = text_encoder(text_input_ids, output_hidden_states=True)
						pooled_prompt_embeds = prompt_embeds_out[0] # CLIP-L Pooler Output
						prompt_embeds = prompt_embeds_out.hidden_states[-2] # Penultimate Hidden State
						prompt_embeds_list.append(prompt_embeds)

					prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
					# pooled_prompt_embeds 는 text_encoder_two의 것 사용

					# SDXL 시간/크기 임베딩 추가
					# <<< UPDATED original_size and target_size using width/height
					original_size = (args.resolution_height, args.resolution_width)
					crops_coords_top_left = (0, 0) # Assuming no cropping
					target_size = (args.resolution_height, args.resolution_width) # Assuming target size is the same
					add_time_ids = list(original_size + crops_coords_top_left + target_size)
					add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype).to(accelerator.device)
					add_time_ids = add_time_ids.repeat(len(prompts), 1)

					unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds.to(unet.dtype)}
					prompt_embeds = prompt_embeds.to(unet.dtype)


				# 노이즈 및 타임스텝 준비 (Policy/Reference 동일하게 사용)
				# VAE는 스케일링 팩터에 사용될 수 있으므로 로드 필요
				if vae:
					vae_scale_factor = 2**(len(vae.config.block_out_channels) - 1)
				else:
					vae_scale_factor = 8 # 기본값

				# <<< UPDATED latent_shape using width/height
				latent_shape = (len(prompts), unet.config.in_channels, args.resolution_height // vae_scale_factor, args.resolution_width // vae_scale_factor)

				# bsz = prompt_embeds.shape[0] # Use actual batch size after dataloader
				# latent_shape = (bsz, unet.config.in_channels, args.resolution_height // vae_scale_factor, args.resolution_width // vae_scale_factor)

				# 깨끗한 latent 대신 바로 noise 샘플링 (일반적인 diffusion 학습 방식)
				noise = torch.randn(latent_shape, device=accelerator.device, dtype=unet.dtype)
				timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latent_shape[0],), device=accelerator.device).long()

				# 노이즈 추가된 latent 생성 (이론상 필요 없으나, 원래 latent 필요시 사용)
				# latents = torch.randn(latent_shape, device=accelerator.device, dtype=unet.dtype) * scheduler.init_noise_sigma
				# noisy_latents = scheduler.add_noise(latents, noise, timesteps)

				# 실제로는 noisy_latents 대신 noise를 직접 예측하는 경우가 많음
				# 여기서는 UNet 입력으로 noisy_latents 대신 그냥 noise를 사용하고
				# target도 noise로 설정하는 방식은 아님. UNet은 noisy_latent를 입력받음
				# 따라서 add_noise를 통해 noisy_latents를 만들어야 함.
				# 단, latents 를 0으로 가정하고 noisy_latents = scheduler.add_noise(zeros, noise, timesteps) 형태로 사용하기도 함.
				# 여기서는 표준적인 방식대로 깨끗한 latent에서 시작하는 것처럼 가정.
				latents = torch.randn(latent_shape, device=accelerator.device, dtype=unet.dtype) # 임의의 latent 샘플링
				latents = latents * scheduler.init_noise_sigma # 스케일링 (필요시)
				noisy_latents = scheduler.add_noise(latents, noise, timesteps) # 노이즈 추가

				# --- DPO Loss 계산 ---
				# 1. Policy 모델 예측 (학습 대상)
				policy_noise_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample
				policy_logps = calculate_log_probs(policy_noise_pred, noise)

				# 2. Reference 모델 예측 (동결)
				with torch.no_grad():
					reference_unet.eval()
					ref_noise_pred = reference_unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample
					ref_logps = calculate_log_probs(ref_noise_pred, noise)

				# 3. DPO Loss 계산 (Score 기반 근사 - USING +1/-1/0 SCORES)
				logits = policy_logps - ref_logps
				# target_prob: 0 (Img1 preferred), 1 (Img0 preferred), 0.5 (Equal)
				target_prob = (scores.float() + 1) / 2 # Maps [-1, 1] to [0, 1]

				# Stable BCE with Logits Loss calculation (REMAINS THE SAME!)
				loss_per_item = - (target_prob * F.logsigmoid(beta * logits) + \
								 (1 - target_prob) * F.logsigmoid(-beta * logits))
				loss = loss_per_item.mean()

				# 통계 기록
				# Need to adjust aggregation if batch size is not uniform due to accelerator
				# For standard use, this is usually fine if batch sizes are consistent per device
				avg_loss = accelerator.gather(loss.repeat(batch["prompt"].__len__())).mean() # Use actual batch size
				avg_policy_logps = accelerator.gather(policy_logps.repeat(batch["prompt"].__len__())).mean()
				avg_ref_logps = accelerator.gather(ref_logps.repeat(batch["prompt"].__len__())).mean()
				avg_logits = accelerator.gather(logits.repeat(batch["prompt"].__len__())).mean()


				epoch_loss += avg_loss.item() / args.gradient_accumulation_steps
				epoch_policy_logps_mean += avg_policy_logps.item() / args.gradient_accumulation_steps
				epoch_ref_logps_mean += avg_ref_logps.item() / args.gradient_accumulation_steps
				epoch_logits_mean += avg_logits.item() / args.gradient_accumulation_steps

				# --- 역전파 및 옵티마이저 스텝 ---
				accelerator.backward(loss)

				# --- Manual Gradient Clipping (inside accumulate, before optimizer.step) ---
				# 그래디언트 클리핑은 누적된 그래디언트가 동기화될 때만 수행
				if accelerator.sync_gradients:
					if args.max_grad_norm is not None:
						# Policy UNet의 학습 가능한 파라미터(LoRA)에 대해 클리핑 수행
						accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
				# -------------------------------------------------------------------------

				if accelerator.sync_gradients:
					optimizer.step()
					optimizer.zero_grad()

					# 로그 기록 및 진행률 업데이트
					if accelerator.is_main_process:
						logs = {
							"loss": avg_loss.item(),
							"policy_logps": avg_policy_logps.item(),
							"ref_logps": avg_logps.item(), # Should this be avg_ref_logps? Yes. Fixing.
							# Corrected line:
							"ref_logps": avg_ref_logps.item(),
							"logits": avg_logits.item(),
							"lr": optimizer.param_groups[0]['lr']
						}
						progress_bar.set_postfix(**logs)
						accelerator.log(logs, step=global_step)

					progress_bar.update(1)
					global_step += 1


				# --- 체크포인트 저장 ---
				# Note: Checkpointing logic might need adjustment if saving is very frequent
				# relative to epoch size and gradient accumulation steps.
				# Current logic saves based on global_step after sync_gradients.
				if accelerator.is_main_process and global_step > 0 and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
					save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
					os.makedirs(save_path, exist_ok=True)
					try:
						unwrapped_unet = accelerator.unwrap_model(unet) # Policy UNet
						# save_lora_weights는 unet의 LoRA 레이어만 저장
						# We are saving the *policy* model's LoRA weights
						pipe.save_lora_weights(
							save_directory=save_path,
							unet_lora_layers=unwrapped_unet.attn_processors, # LoRA 레이어 직접 전달
							safe_serialization=True
						)
						logger.info(f"Saved LoRA checkpoint to {save_path}")
					except Exception as e:
						logger.error(f"체크포인트 저장 실패 (Step {global_step}): {e}")


			if global_step >= max_train_steps:
				logger.info("Max training steps reached. Finishing training.")
				break

		# --- 에포크 종료 로그 ---
		if accelerator.is_main_process:
			 num_steps_in_epoch = (step + 1) # 실제 에포크 내 스텝 수
			 # Avoid division by zero if epoch has no steps (shouldn't happen with valid dataset)
			 if num_steps_in_epoch > 0:
				 # Use the accumulated values from the loop for average
				 epoch_avg_loss = epoch_loss / num_update_steps_per_epoch # Correct division for epoch average
				 epoch_avg_policy_logps = epoch_policy_logps_mean / num_update_steps_per_epoch
				 epoch_avg_ref_logps = epoch_ref_logps_mean / num_update_steps_per_epoch
				 epoch_avg_logits = epoch_logits_mean / num_update_steps_per_epoch

				 logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} finished. Avg Loss: {epoch_avg_loss:.4f}, "
							 f"Avg Policy LogP: {epoch_avg_policy_logps:.4f}, Avg Ref LogP: {epoch_avg_ref_logps:.4f}, "
							 f"Avg Logits: {epoch_avg_logits:.4f}")
			 else:
				 logger.warning(f"Epoch {epoch+1}/{args.num_train_epochs} finished with no training steps.")


	# --- 학습 종료 후 최종 모델 저장 ---
	accelerator.wait_for_everyone()
	if accelerator.is_main_process:
		final_save_path = os.path.join(args.output_dir, "final_lora")
		os.makedirs(final_save_path, exist_ok=True)
		try:
			unwrapped_unet = accelerator.unwrap_model(unet)
			# 최종 LoRA 저장 (파이프라인 객체 이용)
			pipe.save_lora_weights(
				save_directory=final_save_path,
				unet_lora_layers=unwrapped_unet.attn_processors, # 또는 None으로 두면 알아서 찾음
				safe_serialization=True
			)
			logger.info(f"Final fine-tuned LoRA weights saved to {final_save_path}")

			# 설정값 저장
			# Convert Namespace to dict and remove non-serializable items if any
			final_config = vars(args)
			# Example of removing a non-serializable item if necessary
			# if 'some_object' in final_config:
			#     del final_config['some_object']
			with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
				 # Ensure simple types for JSON serialization
				 serializable_config = {k: v for k, v in final_config.items() if isinstance(v, (int, float, str, bool, list, dict)) or v is None}
				 json.dump(serializable_config, f, indent=2, ensure_ascii=False)
			logger.info(f"Training arguments saved to {os.path.join(args.output_dir, 'training_args.json')}")

		except Exception as e:
			 logger.error(f"최종 LoRA 저장 실패: {e}")

	accelerator.end_training()
	logger.info("Training finished.")


# --- 스크립트 실행 부분 (argparse) ---
if __name__ == "__main__":
	# ... (Argparse setup remains the same, ensure help text for --dataset_path is updated if needed) ...
    # parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset file (containing 'prompt_text_reference' and 'comparison_score' [-1, 0, 1]).")
	parser = argparse.ArgumentParser(description="Fine-tune existing SDXL LoRA with DPO (approximated using scores)")

	# 경로 관련 인자
	parser.add_argument("--pretrained_checkpoint_path", type=str, required=True, help="Path to the single base model checkpoint file (.safetensors or .ckpt).")
	parser.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Compatible base model ID or directory path for config/tokenizer.")
	parser.add_argument("--pretrained_lora_path", type=str, required=True, help="Path to the existing LoRA file (.safetensors or .bin) to fine-tune.")
	parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset file (containing 'prompt' and 'score' between -1 and 1).")
	parser.add_argument("--output_dir", type=str, default="sdxl-lora-dpo-finetuned", help="Directory to save fine-tuned LoRA weights and logs.")

	# 학습 파라미터
	parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
	parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device during training.")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients before optimizing.")
	parser.add_argument("--learning_rate", type=float, default=5e-7, help="Initial learning rate for fine-tuning.")
	parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
	parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2.")
	parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
	parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="AdamW epsilon.")
	parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
	parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses for data loading.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
	parser.add_argument("--resolution_width", type=int, default=1024, help="Image width for training.")
	parser.add_argument("--resolution_height", type=int, default=1024, help="Image height for training.")


	# DPO 파라미터
	parser.add_argument("--dpo_beta", type=float, default=0.1, help="Beta parameter for DPO loss, controls divergence sensitivity.") # <<< UPDATED help text

	# 최적화 및 하드웨어 설정
	parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training.")
	parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit AdamW optimizer.")
	parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")

	# 로깅 및 저장
	parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint every X steps.")

	args = parser.parse_args()
	main(args)
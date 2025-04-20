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
from diffusers.loaders import LoraLoaderMixin, AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
# <<< ADDED LR Scheduler import >>>
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import bitsandbytes as bnb
from peft import PeftModel
import warnings # For xformers warning

# 로거 설정
logger = get_logger(__name__)

# --- 데이터셋 클래스 정의 --- (변경 없음)
class DPOPreferenceDataset(Dataset):
	def __init__(self, jsonl_path):
		self.data = []
		try:
			with open(jsonl_path, 'r', encoding='utf-8') as f:
				for line_num, line in enumerate(f):
					try:
						entry = json.loads(line.strip())
						prompt = entry.get("prompt_text_reference")
						score = entry.get("comparison_score")

						if prompt is not None and score is not None:
							try:
								score_float = float(score)
								if -1.0 <= score_float <= 1.0:
									self.data.append({"prompt": prompt, "score": score_float})
								else:
									logger.warning(f"L{line_num+1}: Skipping entry with score outside [-1, 1]: {score_float} in {jsonl_path}")
							except (ValueError, TypeError):
								logger.warning(f"L{line_num+1}: Skipping entry with non-numeric score: {score} in {jsonl_path}")
						else:
							missing = []
							if prompt is None: missing.append("'prompt_text_reference'")
							if score is None: missing.append("'comparison_score'")
							logger.warning(f"L{line_num+1}: Skipping entry with missing fields: {', '.join(missing)} in {jsonl_path}")

					except json.JSONDecodeError:
						logger.warning(f"L{line_num+1}: Skipping invalid JSON line in {jsonl_path}: {line.strip()}")
					except Exception as e:
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
		return self.data[idx]

# --- Log Probability 계산 함수 (MSE 기반 근사) --- (변경 없음)
def calculate_log_probs(predicted_noise: torch.Tensor, actual_noise: torch.Tensor) -> torch.Tensor:
	mse = F.mse_loss(predicted_noise.float(), actual_noise.float(), reduction="none")
	mse_per_item = torch.mean(mse, dim=list(range(1, mse.ndim)))
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
	)
	logger.info(f"Accelerator state initialized. Device: {accelerator.device}, Distributed: {accelerator.distributed_type}, Mixed Precision: {args.mixed_precision}")
	logger.warning("This script uses scores to approximate DPO loss. For best results, use a dataset with chosen/rejected pairs.")

	# *** 2. 경로 및 기본 검사 ***
	if not os.path.isfile(args.pretrained_checkpoint_path):
		logger.error(f"오류: 지정된 체크포인트 파일을 찾을 수 없습니다: {args.pretrained_checkpoint_path}")
		sys.exit(1)
	if not os.path.exists(args.pretrained_lora_path):
		 logger.error(f"오류: 지정된 기존 LoRA 파일을 찾을 수 없습니다: {args.pretrained_lora_path}")
		 sys.exit(1)
	if args.resolution_width % 8 != 0 or args.resolution_height % 8 != 0:
		logger.error(f"오류: 해상도 너비 ({args.resolution_width})와 높이 ({args.resolution_height})는 8의 배수여야 합니다.")
		sys.exit(1)
	if args.resolution_width <= 0 or args.resolution_height <= 0:
		 logger.error(f"오류: 해상도 너비 ({args.resolution_width})와 높이 ({args.resolution_height})는 0보다 커야 합니다.")
		 sys.exit(1)
	logger.info(f"Using training resolution: {args.resolution_width}x{args.resolution_height}")
	logger.info(f"Max token length: {args.max_token_length}") # <<< ADDED Log
	if args.noise_offset > 0: # <<< ADDED Log
		logger.info(f"Using noise offset: {args.noise_offset}")

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
		# Determine torch dtype based on mixed precision setting
		weight_dtype = torch.float32
		if args.mixed_precision == "fp16":
			weight_dtype = torch.float16
			logger.info("Using float16 for model weights.")
		elif args.mixed_precision == "bf16":
			weight_dtype = torch.bfloat16
			logger.info("Using bfloat16 for model weights.")
		else:
			logger.info("Using float32 for model weights.")

		# pipe: 학습 대상 모델 (Policy)
		pipe = StableDiffusionXLPipeline.from_single_file(
			args.pretrained_checkpoint_path,
			torch_dtype=weight_dtype, # Use determined dtype
			# variant=args.mixed_precision if args.mixed_precision != "no" else None, # variant is often for fp16 branches
			use_safetensors=True,
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

	# --- XFormers 활성화 (선택 사항) ---
	if args.use_xformers:
		try:
			import xformers
			unet.enable_xformers_memory_efficient_attention()
			logger.info("Enabled XFormers for Policy UNet.")
		except ImportError:
			warnings.warn("XFormers not installed. Proceeding without memory-efficient attention.")
		except Exception as e:
			warnings.warn(f"Could not enable XFormers: {e}. Proceeding without memory-efficient attention.")


	# --- 모델 가중치 동결 (LoRA 파라미터 제외) ---
	unet.requires_grad_(False)
	text_encoder_one.requires_grad_(False)
	text_encoder_two.requires_grad_(False)
	if vae: vae.requires_grad_(False)

	# --- 기존 LoRA 가중치 로드 (Policy 모델에) ---
	logger.info(f"Loading existing LoRA weights into Policy UNet: {args.pretrained_lora_path}")
	try:
		# Load LoRA weights - use load_lora_weights for diffusers >= 0.17.0
		# Make sure the adapter name is consistent if you need to refer to it later
		pipe.load_lora_weights(
			os.path.dirname(args.pretrained_lora_path),
			weight_name=os.path.basename(args.pretrained_lora_path),
			adapter_name="dpo_lora" # Name this adapter
		)
		# Set the loaded adapter as active
		pipe.set_adapters(["dpo_lora"], adapter_weights=[1.0])
		logger.info(f"Successfully loaded and set LoRA weights adapter 'dpo_lora' into Policy UNet.")

		# --- 중요: LoRA 레이어만 학습 가능하도록 설정 ---
		# Freeze everything first (already done above)
		# Then, unfreeze only the parameters associated with the loaded LoRA adapter
		lora_params = []
		for name, param in unet.named_parameters():
			# Check if the parameter name contains indications of being part of the loaded LoRA layers
			# This might need adjustment based on how load_lora_weights names parameters
			# A common pattern includes 'lora' or the adapter name 'dpo_lora'
			# A safer way is to check requires_grad status *after* potential unfreezing by peft/load_lora
			# However, load_lora_weights itself *doesn't* unfreeze, it just loads weights.
			# We need to identify the LoRA parameters *manually* or rely on their names.
			# Let's assume standard LoRA naming convention within AttnProcs.
			if "lora" in name: # This is a common convention
				param.requires_grad_(True)
				lora_params.append(param)

		# Verify we found parameters
		if not lora_params:
			# Alternative check: Use the structure of LoRAAttnProcessor
			lora_params = list(filter(lambda p: p.requires_grad, unet.parameters())) # Check requires_grad *after* loading

		if not lora_params:
			logger.warning("Could not automatically identify LoRA parameters to unfreeze based on name 'lora'.")
			# Fallback: Check for parameters within LoRA Attention Processors
			lora_params = []
			for attn_processor in unet.attn_processors.values():
				if hasattr(attn_processor, 'parameters'):
					for param in attn_processor.parameters():
						param.requires_grad_(True)
						lora_params.append(param)

		if not lora_params:
			logger.error("오류: LoRA 가중치를 로드했으나 학습 가능한 LoRA 파라미터를 찾거나 활성화할 수 없습니다.")
			sys.exit(1)
		else:
			logger.info(f"Successfully unfrozen {len(lora_params)} LoRA parameters in Policy UNet for training.")

	except Exception as e:
		 logger.error(f"오류: Policy 모델 LoRA 로딩 또는 활성화 실패: {e}")
		 import traceback
		 traceback.print_exc()
		 sys.exit(1)

	# --- 학습 대상 파라미터 설정 (다시 확인) ---
	params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
	logger.info(f"Final check: Number of trainable parameters in Policy UNet: {len(params_to_optimize)}")
	if not params_to_optimize:
		 logger.error("오류: 학습 가능한 파라미터를 찾을 수 없습니다.")
		 sys.exit(1)

	# --- 그래디언트 체크포인팅 ---
	if args.gradient_checkpointing:
		unet.enable_gradient_checkpointing()
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
		train_dataset = DPOPreferenceDataset(args.dataset_path)
		train_dataloader = DataLoader(
			train_dataset,
			batch_size=args.train_batch_size,
			shuffle=True,
			collate_fn=lambda batch: {
				"prompt": [item["prompt"] for item in batch],
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
		ref_pipe = StableDiffusionXLPipeline.from_single_file(
			args.pretrained_checkpoint_path,
			torch_dtype=weight_dtype, # Use same dtype as policy model
			# variant=args.mixed_precision if args.mixed_precision != "no" else None,
			use_safetensors=True
		)
		# 레퍼런스 파이프라인에도 원본 LoRA 로드 (다른 이름으로)
		ref_pipe.load_lora_weights(
			os.path.dirname(args.pretrained_lora_path),
			weight_name=os.path.basename(args.pretrained_lora_path),
			adapter_name="original_lora" # Different name for clarity
		)
		ref_pipe.set_adapters(["original_lora"]) # Activate the reference LoRA

		ref_pipe.unet.requires_grad_(False)
		ref_pipe.text_encoder.requires_grad_(False)
		ref_pipe.text_encoder_2.requires_grad_(False)
		if ref_pipe.vae: ref_pipe.vae.requires_grad_(False)

		# <<< ADDED XFormers for Reference UNet >>>
		if args.use_xformers:
			try:
				ref_pipe.unet.enable_xformers_memory_efficient_attention()
				logger.info("Enabled XFormers for Reference UNet.")
			except Exception as e:
				warnings.warn(f"Could not enable XFormers for Reference UNet: {e}")

		ref_pipe.to(accelerator.device)
		ref_pipe.set_progress_bar_config(disable=True)
		reference_unet = ref_pipe.unet
		logger.info("Reference pipeline with original LoRA created and frozen successfully.")
	except Exception as e:
		logger.error(f"레퍼런스 파이프라인 생성 실패: {e}")
		sys.exit(1)


	# --- 학습 루프 계산 ---
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
	max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
	total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

	# --- LR 스케줄러 설정 ---
	logger.info(f"Configuring LR Scheduler: {args.lr_scheduler}")
	lr_scheduler = get_scheduler(
		args.lr_scheduler,
		optimizer=optimizer,
		num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, # Scale warmup steps
		num_training_steps=max_train_steps * accelerator.num_processes, # Scale total steps
        # Pass scheduler specific args if applicable (adjust names if get_scheduler expects different ones)
        num_cycles=args.lr_scheduler_num_cycles, # For cosine_with_restarts
        power=args.lr_scheduler_power,           # For polynomial
	)

	# --- Accelerator 준비 ---
	# Policy UNet, optimizer, dataloader, scheduler 준비
	unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
		unet, optimizer, train_dataloader, lr_scheduler
	)
	# 나머지 컴포넌트들도 GPU/분산 환경으로 이동
	text_encoder_one.to(accelerator.device)
	text_encoder_two.to(accelerator.device)
	if vae: vae.to(accelerator.device)
	# reference_unet은 이미 .to(accelerator.device) 완료됨

	# --- 학습 루프 시작 ---
	logger.info("***** Running DPO Fine-tuning *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(f"  Num Epochs = {args.num_train_epochs}")
	logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
	logger.info(f"  Total train batch size (w. parallel, distributed, accum) = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	logger.info(f"  Total optimization steps = {max_train_steps}")
	logger.info(f"  Learning Rate Scheduler = {args.lr_scheduler}") # <<< ADDED Log
	logger.info(f"  LR Warmup Steps = {args.lr_warmup_steps}") # <<< ADDED Log

	global_step = 0
	progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
	progress_bar.set_description("Steps")

	beta = args.dpo_beta

	for epoch in range(args.num_train_epochs):
		epoch_loss = 0.0
		epoch_policy_logps_mean = 0.0
		epoch_ref_logps_mean = 0.0
		epoch_logits_mean = 0.0
		unet.train() # Ensure policy model is in train mode

		for step, batch in enumerate(train_dataloader):
			with accelerator.accumulate(unet):
				prompts = batch["prompt"]
				scores = batch["score"].to(accelerator.device)

				# Text Embeddings 생성
				with torch.no_grad():
					prompt_embeds_list = []
					# <<< UPDATED Tokenizer call with max_token_length >>>
					for tokenizer, text_encoder in zip([tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]):
						text_inputs = tokenizer(
							prompts, padding="max_length", max_length=args.max_token_length, # Use arg
							truncation=True, return_tensors="pt",
						)
						text_input_ids = text_inputs.input_ids.to(accelerator.device)
						prompt_embeds_out = text_encoder(text_input_ids, output_hidden_states=True)
						pooled_prompt_embeds = prompt_embeds_out[0]
						prompt_embeds = prompt_embeds_out.hidden_states[-2]
						prompt_embeds_list.append(prompt_embeds)

					prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
					# SDXL 시간/크기 임베딩 추가
					original_size = (args.resolution_height, args.resolution_width)
					crops_coords_top_left = (0, 0)
					target_size = (args.resolution_height, args.resolution_width)
					add_time_ids = list(original_size + crops_coords_top_left + target_size)
					add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype).to(accelerator.device)
					add_time_ids = add_time_ids.repeat(len(prompts), 1)
					unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds.to(unet.dtype)}
					prompt_embeds = prompt_embeds.to(unet.dtype)


				# 노이즈 및 타임스텝 준비
				if vae:
					vae_scale_factor = 2**(len(vae.config.block_out_channels) - 1)
				else:
					vae_scale_factor = 8 # Default for SDXL

				latent_shape = (len(prompts), unet.config.in_channels, args.resolution_height // vae_scale_factor, args.resolution_width // vae_scale_factor)

				# Sample noise
				noise = torch.randn(latent_shape, device=accelerator.device, dtype=unet.dtype)

                # <<< ADDED Noise Offset (if enabled) >>>
				# Add noise offset similar to diffusers train_text_to_image_sdxl.py
				if args.noise_offset > 0:
					noise = noise + args.noise_offset * torch.randn(
						latent_shape, device=accelerator.device, dtype=unet.dtype
					)

				timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latent_shape[0],), device=accelerator.device).long()

				# Add noise to latents
				latents = torch.randn(latent_shape, device=accelerator.device, dtype=unet.dtype)
				latents = latents * scheduler.init_noise_sigma
				noisy_latents = scheduler.add_noise(latents, noise, timesteps)

				# --- DPO Loss 계산 ---
				# 1. Policy 모델 예측
				policy_noise_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample
				policy_logps = calculate_log_probs(policy_noise_pred, noise)

				# 2. Reference 모델 예측
				with torch.no_grad():
					reference_unet.eval() # Ensure reference model is in eval mode
					ref_noise_pred = reference_unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample
					ref_logps = calculate_log_probs(ref_noise_pred, noise)

				# 3. DPO Loss 계산 (Score 기반 근사)
				logits = policy_logps - ref_logps
				target_prob = (scores.float() + 1) / 2 # [-1, 1] -> [0, 1]
				loss_per_item = - (target_prob * F.logsigmoid(beta * logits) + \
								 (1 - target_prob) * F.logsigmoid(-beta * logits))
				loss = loss_per_item.mean()

				# 통계 기록
				current_batch_size = batch["prompt"].__len__() # Use actual batch size for gather
				avg_loss = accelerator.gather(loss.repeat(current_batch_size)).mean()
				avg_policy_logps = accelerator.gather(policy_logps).mean() # Already per-item
				avg_ref_logps = accelerator.gather(ref_logps).mean() # Already per-item
				avg_logits = accelerator.gather(logits).mean() # Already per-item

				# Accumulate epoch stats (divide by accum steps later if needed, or log per-step)
				# Let's log per-step average for clarity
				epoch_loss += avg_loss.item() # Track total loss sum for epoch average later
				# Track other means if needed for epoch average

				# --- 역전파 및 옵티마이저 스텝 ---
				accelerator.backward(loss)

				if accelerator.sync_gradients:
					# Gradient clipping applied only to trainable parameters
					if args.max_grad_norm is not None and args.max_grad_norm > 0:
						accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

					optimizer.step()
					lr_scheduler.step() # <<< MOVED LR Scheduler Step AFTER optimizer step >>>
					optimizer.zero_grad()

					# 로그 기록 및 진행률 업데이트
					if accelerator.is_main_process:
						logs = {
							"loss": avg_loss.item(),
							"policy_logps": avg_policy_logps.item(),
							"ref_logps": avg_ref_logps.item(),
							"logits": avg_logits.item(),
							"lr": lr_scheduler.get_last_lr()[0] # Get current LR from scheduler
						}
						progress_bar.set_postfix(**logs)
						accelerator.log(logs, step=global_step)

					progress_bar.update(1)
					global_step += 1


				# --- 체크포인트 저장 ---
				if accelerator.is_main_process and global_step > 0 and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
					save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
					os.makedirs(save_path, exist_ok=True)
					try:
						unwrapped_unet = accelerator.unwrap_model(unet)
						# Use pipeline's save_lora_weights for convenience
						# Need to ensure the pipeline has the correct UNet reference
						# If unet was prepared, pipe.unet might not be the prepared one.
						# It's safer to extract LoRA layers from unwrapped_unet
						policy_pipe = pipe # Assuming pipe holds the prepared unet via accelerator
						policy_pipe.save_lora_weights(
							save_directory=save_path,
							# Specify the adapter name saved during loading/training
							adapter_name="dpo_lora",
							# Or pass the layers directly from the unwrapped model if needed
							# unet_lora_layers=unwrapped_unet.attn_processors,
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
			 avg_epoch_loss = epoch_loss / num_update_steps_per_epoch # Calculate average loss for the epoch
			 logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} finished. Avg Loss: {avg_epoch_loss:.4f}")
             # Add other averaged epoch stats if needed


	# --- 학습 종료 후 최종 모델 저장 ---
	accelerator.wait_for_everyone()
	if accelerator.is_main_process:
		final_save_path = os.path.join(args.output_dir, "final_lora")
		os.makedirs(final_save_path, exist_ok=True)
		try:
			unwrapped_unet = accelerator.unwrap_model(unet)
			policy_pipe = pipe # Use the main pipeline object
			policy_pipe.save_lora_weights(
				save_directory=final_save_path,
				adapter_name="dpo_lora", # Save the trained adapter
				safe_serialization=True
			)
			logger.info(f"Final fine-tuned LoRA weights saved to {final_save_path}")

			# 설정값 저장
			final_config = vars(args)
			with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
				 serializable_config = {k: v for k, v in final_config.items() if isinstance(v, (int, float, str, bool, list, dict)) or v is None}
				 json.dump(serializable_config, f, indent=2, ensure_ascii=False)
			logger.info(f"Training arguments saved to {os.path.join(args.output_dir, 'training_args.json')}")

		except Exception as e:
			 logger.error(f"최종 LoRA 저장 실패: {e}")

	accelerator.end_training()
	logger.info("Training finished.")


# --- 스크립트 실행 부분 (argparse) ---
if __name__ == "__main__":
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
	parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (0 to disable).")
	parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses for data loading.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
	parser.add_argument("--resolution_width", type=int, default=1024, help="Image width for training.")
	parser.add_argument("--resolution_height", type=int, default=1024, help="Image height for training.")
    # <<< ADDED Kohya-inspired args >>>
	parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", help="Learning rate scheduler type (e.g., 'linear', 'cosine', 'constant').")
	parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Number of steps for the learning rate warmup.")
	parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1, help="Number of cycles for cosine with restarts scheduler.")
	parser.add_argument("--lr_scheduler_power", type=float, default=1.0, help="Power factor for polynomial scheduler.")
	parser.add_argument("--max_token_length", type=int, default=77, help="Maximum sequence length for tokenizers.") # Default 77, user can increase for SDXL if needed
	parser.add_argument("--noise_offset", type=float, default=0.0, help="Amount of noise offset to add during training (0 to disable).")


	# DPO 파라미터
	parser.add_argument("--dpo_beta", type=float, default=0.1, help="Beta parameter for DPO loss.")

	# 최적화 및 하드웨어 설정
	parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training.")
	parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit AdamW optimizer.")
	parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    # <<< ADDED Kohya-inspired flag >>>
	parser.add_argument("--use_xformers", action="store_true", help="Enable xformers memory efficient attention (if available).")

	# 로깅 및 저장
	parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint every X steps (0 to disable).")

	args = parser.parse_args()
	main(args)
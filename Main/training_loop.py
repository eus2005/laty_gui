import logging
import math
import os
import shutil
import glob
import contextlib

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Helper 함수 임포트 (필요한 경우)
try:
	from dpo_helpers import encode_prompt, log_validation, VALIDATION_PROMPTS
except ImportError:
	print("Warning: Could not import from dpo_helpers.py in training_loop.py")
	# 필요한 함수가 없으면 오류 발생 가능

logger = logging.getLogger(__name__)

# --- Time ID 계산 함수 ---
# weight_dtype은 training_loop 호출 시 인자로 받아야 함
def compute_time_ids(original_size, crops_coords_top_left, target_resolution, device, dtype):
	if not isinstance(target_resolution, (tuple, list)) or len(target_resolution) != 2:
		raise ValueError(f"target_resolution must be a tuple/list of (height, width), got {target_resolution}")
	target_size = (target_resolution[0], target_resolution[1])
	add_time_ids = list(original_size) + list(crops_coords_top_left) + list(target_size)
	return torch.tensor([add_time_ids], device=device, dtype=dtype)

# --- 학습 루프 함수 ---
def training_loop(
	args,
	accelerator,
	unet, text_encoder_one, text_encoder_two, noise_scheduler,
	optimizer, params_to_optimize, lr_scheduler,
	train_dataloader,
	weight_dtype, # <<< weight_dtype 추가
	initial_global_step, first_epoch,
	num_update_steps_per_epoch, # 재계산된 값 전달
):
	global_step = initial_global_step

	progress_bar = tqdm(
		range(initial_global_step, args.max_train_steps),
		desc="Steps",
		disable=not accelerator.is_local_main_process,
	)

	accelerator.print("***** Starting training loop *****")
	unet.eval() # UNet 평가 모드

	effective_epochs = args.num_train_epochs if num_update_steps_per_epoch != float('inf') else int(1e6)

	for epoch in range(first_epoch, effective_epochs):
		text_encoder_one.train(); text_encoder_two.train()
		train_loss = 0.0
		num_steps_in_epoch = 0

		epoch_desc = f"Epoch {epoch+1}/{args.num_train_epochs}" if num_update_steps_per_epoch != float('inf') else f"Epoch {epoch+1} (Streaming)"
		progress_bar.set_description(epoch_desc)

		for step, batch in enumerate(train_dataloader):
			if batch is None or "input_ids_one" not in batch or batch["input_ids_one"].shape[0] == 0:
				logger.warning(f"Skipping empty batch at step {step}."); continue

			with accelerator.accumulate():
				# --- 데이터 준비 ---
				try:
					latents_w = batch["latents_w"].to(accelerator.device, dtype=weight_dtype)
					latents_l = batch["latents_l"].to(accelerator.device, dtype=weight_dtype)
					latents = torch.cat([latents_w, latents_l], dim=0)
					input_ids_one = batch["input_ids_one"].to(accelerator.device)
					input_ids_two = batch["input_ids_two"].to(accelerator.device)
					original_sizes_w = batch["original_sizes_w"]
					crop_top_lefts_w = batch["crop_top_lefts_w"]
				except KeyError as e: logger.warning(f"Batch missing key {e}. Skipping."); continue
				except Exception as e: logger.error(f"Error preparing data: {e}", exc_info=True); continue

				# --- 노이즈 및 타임스텝 ---
				noise = torch.randn_like(latents)
				bs = latents.shape[0] // 2
				timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long().repeat(2)
				noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

				# --- Time IDs 계산 ---
				target_resolution_tuple = (args.resolution_height, args.resolution_width)
				try:
					# <<< compute_time_ids 호출 시 device 및 dtype 전달 >>>
					add_time_ids = torch.cat([
						compute_time_ids(s, c, target_resolution_tuple, device=accelerator.device, dtype=weight_dtype)
						for s, c in zip(original_sizes_w, crop_top_lefts_w)
					], dim=0).repeat(2, 1)
				except Exception as e: logger.error(f"Error computing Time IDs: {e}", exc_info=True); continue

				# --- 임베딩 계산 ---
				try:
					prompt_embeds, pooled_prompt_embeds = encode_prompt([text_encoder_one, text_encoder_two], [input_ids_one, input_ids_two])
					prompt_embeds_policy = prompt_embeds.repeat(2, 1, 1)
					pooled_prompt_embeds_policy = pooled_prompt_embeds.repeat(2, 1)

					adapter_context1 = contextlib.nullcontext(); adapter_context2 = contextlib.nullcontext()
					if hasattr(text_encoder_one, 'disable_adapter') and hasattr(text_encoder_two, 'disable_adapter'):
						adapter_context1 = text_encoder_one.disable_adapter(); adapter_context2 = text_encoder_two.disable_adapter()
					with adapter_context1, adapter_context2, torch.no_grad():
						 ref_prompt_embeds, ref_pooled_prompt_embeds = encode_prompt([text_encoder_one, text_encoder_two], [input_ids_one, input_ids_two])
					prompt_embeds_ref = ref_prompt_embeds.repeat(2, 1, 1)
					pooled_prompt_embeds_ref = ref_pooled_prompt_embeds.repeat(2, 1)
				except Exception as e: logger.warning(f"Error encoding prompts: {e}", exc_info=True); continue

				# --- UNet 예측 및 손실 계산 ---
				unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds_policy}
				ref_unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds_ref}
				try:
					model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states=prompt_embeds_policy, added_cond_kwargs=unet_added_conditions).sample
					with torch.no_grad(): ref_pred = unet(noisy_model_input, timesteps, encoder_hidden_states=prompt_embeds_ref, added_cond_kwargs=ref_unet_added_conditions).sample

					if noise_scheduler.config.prediction_type == "epsilon": target = noise
					elif noise_scheduler.config.prediction_type == "v_prediction": target = noise_scheduler.get_velocity(latents, noise, timesteps)
					else: raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

					model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
					ref_losses = F.mse_loss(ref_pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])

					model_losses_w, model_losses_l = model_losses.chunk(2)
					ref_losses_w, ref_losses_l = ref_losses.chunk(2)
					model_diff = model_losses_l - model_losses_w
					ref_diff = ref_losses_l - ref_losses_w
					scale_term = args.beta_dpo
					inside_term = scale_term * (model_diff - ref_diff)
					loss = -F.logsigmoid(inside_term).mean()
				except Exception as e: logger.warning(f"Error in UNet/loss calc: {e}", exc_info=True); continue

				# 로그용 값
				with torch.no_grad():
					 raw_model_loss = 0.5 * (model_losses_w + model_losses_l).mean()
					 raw_ref_loss = 0.5 * (ref_losses_w + ref_losses_l).mean()
					 implicit_acc_adjusted = (inside_term > 0).float().mean() + 0.5 * (inside_term == 0).float().mean()

				# 역전파 및 업데이트
				avg_loss = accelerator.gather(loss.detach()).mean()
				train_loss += avg_loss.item(); num_steps_in_epoch += 1
				try:
					accelerator.backward(loss)
					if accelerator.sync_gradients:
						accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
						optimizer.step()
						lr_scheduler.step()
						optimizer.zero_grad(set_to_none=True)
				except Exception as e: logger.error(f"Error in backward/step: {e}", exc_info=True); return # 루프 종료

			# 로깅 및 체크포인트
			if accelerator.sync_gradients:
				progress_bar.update(1); global_step += 1
				logs = {"loss": avg_loss.item(), "model_loss": raw_model_loss.item(), "ref_loss": raw_ref_loss.item(), "acc": implicit_acc_adjusted.item(), "lr": lr_scheduler.get_last_lr()[0]}
				progress_bar.set_postfix(**{k: f"{v:.4g}" for k, v in logs.items()})
				accelerator.log(logs, step=global_step)

				if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
					save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
					try:
						accelerator.save_state(save_path, safe_serialization=True)
						logger.info(f"Saved state to {save_path}")
						if args.checkpoints_total_limit is not None and args.checkpoints_total_limit > 0:
								checkpoints = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), key=lambda x: int(os.path.basename(x).split('-')[1]))
								if len(checkpoints) > args.checkpoints_total_limit:
									for ckpt_to_remove in checkpoints[:len(checkpoints) - args.checkpoints_total_limit]:
										logger.info(f"Deleting old checkpoint: {ckpt_to_remove}"); shutil.rmtree(ckpt_to_remove, ignore_errors=True)
					except Exception as e: logger.warning(f"Could not save checkpoint: {e}")

				if args.run_validation and accelerator.is_main_process and global_step % args.validation_steps == 0:
					logger.info(f"Running validation at step {global_step}...")
					try: log_validation(args, text_encoder_one, text_encoder_two, unet, accelerator, weight_dtype, global_step, False, logger, VALIDATION_PROMPTS)
					except Exception as e: logger.error(f"Validation failed: {e}", exc_info=True)

				if global_step >= args.max_train_steps: break

		# 에포크 종료 로깅
		avg_epoch_loss = train_loss / num_steps_in_epoch if num_steps_in_epoch > 0 else 0
		accelerator.log({"epoch_loss": avg_epoch_loss, "epoch": epoch + 1}, step=global_step)
		accelerator.print(f"{epoch_desc} finished. Average Loss: {avg_epoch_loss:.4f}")

		if global_step >= args.max_train_steps: break

	accelerator.print("Training loop finished.")
	# 최종 global_step 반환 (검증 등에서 사용 가능)
	return global_step

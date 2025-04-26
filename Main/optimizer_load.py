import logging
import math
import torch
import argparse
from accelerate import Accelerator
from diffusers.optimization import get_scheduler

logger = logging.getLogger(__name__)

def create_optimizer_and_scheduler(
	args: argparse.Namespace,
	params_to_optimize: list,
	accelerator: Accelerator = None,
	num_update_steps_per_epoch: float = float('inf'), # 스케줄러 계산용
):
	"""옵티마이저와 LR 스케줄러를 생성합니다."""
	print_fn = accelerator.print if accelerator else print
	optimizer = None
	lr_scheduler = None

	# --- 옵티마이저 생성 ---
	if args.use_8bit_adam:
		try:
			import bitsandbytes as bnb
			optimizer_class = bnb.optim.AdamW8bit
			print_fn(f"Using {optimizer_class.__name__} optimizer.")
		except ImportError:
			print_fn("[ERROR] bitsandbytes not found. Cannot use 8-bit Adam."); return None, None
	else:
		optimizer_class = torch.optim.AdamW
		print_fn(f"Using {optimizer_class.__name__} optimizer.")

	try:
		optimizer = optimizer_class(
			params_to_optimize, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
			weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
		)
	except Exception as e:
		print_fn(f"[ERROR] Failed to create optimizer: {e}")
		logger.error("Optimizer creation failed", exc_info=True)
		return None, None

	# --- 스케줄러 계산 및 생성 ---
	# num_training_steps 계산
	if args.max_train_steps is None:
		if num_update_steps_per_epoch == float('inf'):
			print_fn("[ERROR] max_train_steps must be specified for streaming dataset or unknown dataloader length.")
			return None, None
		calculated_max_steps = args.num_train_epochs * num_update_steps_per_epoch
		print_fn(f"Calculated max_train_steps for scheduler: {calculated_max_steps}")
	else:
		calculated_max_steps = args.max_train_steps # 사용자가 지정한 값 사용
		print_fn(f"Using provided max_train_steps for scheduler: {calculated_max_steps}")

	# num_training_steps 에 gradient accumulation 반영 (Accelerate 0.17.0+ 기준)
	num_training_steps_for_scheduler = calculated_max_steps * args.gradient_accumulation_steps

	try:
		lr_scheduler = get_scheduler(
			args.lr_scheduler, optimizer=optimizer,
			num_warmup_steps=args.lr_warmup_steps, # * accelerator.num_processes 제거
			num_training_steps=num_training_steps_for_scheduler,
			num_cycles=args.lr_scheduler_num_cycles,
			power=args.lr_scheduler_power,
		)
		print_fn(f"LR Scheduler '{args.lr_scheduler}' created.")
	except Exception as e:
		print_fn(f"[ERROR] Failed to create LR scheduler: {e}")
		logger.error("LR Scheduler creation failed", exc_info=True)
		return optimizer, None # 옵티마이저는 생성되었을 수 있으므로 반환

	return optimizer, lr_scheduler

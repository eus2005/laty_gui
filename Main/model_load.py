import torch
import gc
import logging
import argparse
import os
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionXLPipeline
from diffusers.loaders import LoraLoaderMixin # LoRA 로딩 위해 필요
from accelerate import Accelerator

logger = logging.getLogger(__name__)

# <<< LoRA 로딩 및 설정 함수 추가 >>>
def setup_lora_for_text_encoders(text_encoder_one, text_encoder_two, pretrained_lora_path, accelerator):
	print_fn = accelerator.print if accelerator else print
	try:
		print_fn(f"Loading pretrained TE LoRA from: {pretrained_lora_path}")
		# ... (LoRA 파일 존재 확인) ...

		# --- LoRA 로드 ---
		lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(pretrained_lora_path)
		# <<< 로드 전 requires_grad 상태 확인 (디버깅용) >>>
		print_fn("Requires_grad status BEFORE loading LoRA:")
		for i, te in enumerate([text_encoder_one, text_encoder_two]):
			count_true = sum(1 for p in te.parameters() if p.requires_grad)
			print_fn(f"  TE{i+1}: {count_true} / {len(list(te.parameters()))} params require grad.")

		# <<< LoRA 로드 실행 >>>
		LoraLoaderMixin.load_lora_into_text_encoder(lora_state_dict, network_alphas, text_encoder_one, "text_encoder", 1.0)
		print_fn("Loaded LoRA into TE1.")
		LoraLoaderMixin.load_lora_into_text_encoder(lora_state_dict, network_alphas, text_encoder_two, "text_encoder_2", 1.0)
		print_fn("Loaded LoRA into TE2.")
		del lora_state_dict

		# --- LoRA 파라미터 학습 가능하도록 설정 및 확인 ---
		print_fn("Setting requires_grad=True for LoRA parameters...")
		lora_layers_found = False
		trainable_params_count = 0
		total_params = 0
		trainable_names = [] # 학습 가능한 파라미터 이름 로깅용

		for i, te in enumerate([text_encoder_one, text_encoder_two]):
			total_params += sum(p.numel() for p in te.parameters())
			# <<< PEFT 모델 여부 확인 및 처리 >>>
			# load_lora_into_text_encoder 가 PEFT 모델로 변환할 수 있음
			# PEFT 모델의 경우, lora 파라미터만 trainable로 설정하는 것이 일반적
			# if isinstance(te, peft.PeftModel): # PEFT 모델인지 확인
			# 	te.base_model.requires_grad_(False) # 베이스 모델 고정 확실히
			# 	# PEFT 모델은 LoRA 레이어 requires_grad 를 자동으로 처리할 수 있음
			# 	# 또는 명시적으로 설정 필요 시 아래 루프 사용

			for name, param in te.named_parameters():
				# <<< LoRA 파라미터 식별 조건 강화 >>>
				# diffusers에서 로드 시 'lora_A', 'lora_B' 같은 이름 포함 가능성
				is_lora_param = "lora" in name.lower()
				if is_lora_param:
					param.requires_grad_(True) # 명시적으로 True 설정
					lora_layers_found = True
					trainable_params_count += param.numel()
					trainable_names.append(f"TE{i+1}: {name}") # 학습 대상 이름 추가
				elif param.requires_grad: # LoRA가 아닌데 True로 되어있으면 False로
					param.requires_grad_(False)
					logger.warning(f"Non-LoRA param {name} was requires_grad=True. Setting to False.")

		if not lora_layers_found:
			raise ValueError("Could not find any LoRA parameters ('lora' in name) after loading.")

		# 최종 확인
		final_trainable_count = sum(p.numel() for te in [text_encoder_one, text_encoder_two] for p in te.parameters() if p.requires_grad)
		print_fn(f"Final check: Found {final_trainable_count} trainable parameters.")
		if trainable_params_count != final_trainable_count:
			logger.warning("Mismatch in trainable parameter count during setup!")
		if final_trainable_count == 0:
			raise ValueError("No parameters left trainable after setup!")

		# 학습 가능한 파라미터 이름 일부 출력 (디버깅용)
		print_fn(f"Top 5 trainable parameter names: {trainable_names[:5]}")
		print_fn(f"Trainable params count: {final_trainable_count} ({final_trainable_count/total_params*100:.4f}%)")

		return True, final_trainable_count, total_params

	except Exception as e:
		# ... (기존 에러 처리) ...
		print_fn(f"[ERROR] Failed to setup LoRA: {e}")
		logger.error("Failed to setup LoRA", exc_info=True)
		return False, 0, 0

# --- load_models_and_configure 함수 (setup_lora_for_text_encoders 호출 부분 수정 없음) ---
def load_models_and_configure(args, weight_dtype, device, accelerator):
	# ... (모델 로드, 파라미터 고정) ...
	unet, text_encoder_one, text_encoder_two, noise_scheduler = None, None, None, None
	params_to_optimize = []
	try: # 파이프라인 로딩
		print_fn = accelerator.print if accelerator else print
		print_fn(f"Attempting to load pipeline from single file: {args.pretrained_model_name_or_path}")
		pipeline = StableDiffusionXLPipeline.from_single_file(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)
		unet = pipeline.unet.to(device); text_encoder_one = pipeline.text_encoder.to(device)
		text_encoder_two = pipeline.text_encoder_2.to(device); noise_scheduler = pipeline.scheduler
		del pipeline; gc.collect(); torch.cuda.empty_cache()
		print_fn("Components extracted.")
	except Exception as e_pipe: print_fn(f"[ERROR] Pipeline loading failed: {e_pipe}"); return None, None, None, None, None

	unet.requires_grad_(False); text_encoder_one.requires_grad_(False); text_encoder_two.requires_grad_(False)
	print_fn("Base models frozen.")

	# LoRA 설정 호출
	lora_success, _, _ = setup_lora_for_text_encoders(text_encoder_one, text_encoder_two, args.pretrained_lora_path, accelerator)
	if not lora_success: return None, None, None, None, None

	# params_to_optimize 다시 생성 (setup_lora 이후 상태 기준)
	params_to_optimize = [p for te in [text_encoder_one, text_encoder_two] for p in te.parameters() if p.requires_grad]
	if not params_to_optimize: logger.error("No trainable params after LoRA setup!"); return None, None, None, None, None
	accelerator.print(f"Identified {len(params_to_optimize)} parameter tensors to optimize after LoRA setup.")

	# Gradient Checkpointing
	if args.gradient_checkpointing:
		try:
			text_encoder_one.gradient_checkpointing_enable(); text_encoder_two.gradient_checkpointing_enable()
			print_fn("Enabled gradient checkpointing.")
		except Exception as e: print_fn(f"[WARN] Could not enable gradient checkpointing: {e}")

	return unet, text_encoder_one, text_encoder_two, noise_scheduler, params_to_optimize

def load_models_and_configure(
	args: argparse.Namespace, # args 전체 전달 (LoRA 경로 등 필요)
	weight_dtype: torch.dtype,
	device: torch.device,
	accelerator: Accelerator = None,
) -> tuple:
	"""
	모델 로드, 파라미터 고정, LoRA 설정을 수행합니다.

	Returns:
		tuple: (unet, text_encoder_one, text_encoder_two, noise_scheduler, params_to_optimize)
			   오류 발생 시 None 반환 요소 포함.
	"""
	print_fn = accelerator.print if accelerator else print
	unet, text_encoder_one, text_encoder_two, noise_scheduler = None, None, None, None
	params_to_optimize = []

	# --- 1. 모델 로드 (파이프라인 방식 유지) ---
	try:
		print_fn(f"Attempting to load pipeline from single file: {args.pretrained_model_name_or_path}")
		pipeline = StableDiffusionXLPipeline.from_single_file(
			args.pretrained_model_name_or_path, torch_dtype=weight_dtype
		)
		print_fn("Pipeline loaded successfully.")

		unet = pipeline.unet.to(device)
		text_encoder_one = pipeline.text_encoder.to(device)
		text_encoder_two = pipeline.text_encoder_2.to(device)
		noise_scheduler = pipeline.scheduler

		del pipeline; gc.collect(); torch.cuda.empty_cache()
		print_fn("Components extracted and pipeline object deleted.")

	except Exception as e_pipe:
		print_fn(f"[ERROR] Failed to load pipeline directly: {e_pipe}")
		logger.error("Pipeline loading failed.", exc_info=True)
		return None, None, None, None, None

	# --- 2. 모델 파라미터 고정 ---
	unet.requires_grad_(False)
	text_encoder_one.requires_grad_(False)
	text_encoder_two.requires_grad_(False)
	print_fn("Base model parameters frozen.")

	# --- 3. LoRA 로딩 및 설정 ---
	lora_success, trainable_count, total_te_params = setup_lora_for_text_encoders(
		text_encoder_one, text_encoder_two, args.pretrained_lora_path, accelerator
	)
	if not lora_success:
		return None, None, None, None, None

	# 학습 대상 파라미터 리스트 생성
	params_to_optimize = [p for te in [text_encoder_one, text_encoder_two] for p in te.parameters() if p.requires_grad]
	accelerator.print(f"Identified {len(params_to_optimize)} parameter tensors to optimize.")

	# --- 4. Gradient Checkpointing (선택적) ---
	if args.gradient_checkpointing:
		try:
			text_encoder_one.gradient_checkpointing_enable()
			text_encoder_two.gradient_checkpointing_enable()
			print_fn("Enabled gradient checkpointing for Text Encoders.")
		except Exception as e:
			print_fn(f"[WARN] Could not enable gradient checkpointing: {e}")

	return unet, text_encoder_one, text_encoder_two, noise_scheduler, params_to_optimize

# --- END OF FILE model_load.py ---
import os
import torch
import logging
import contextlib
import numpy as np
from PIL import Image
from diffusers.loaders import LoraLoaderMixin
from typing import Tuple 
import argparse
import peft
import time

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.loaders import LoraLoaderMixin
from diffusers import AutoencoderKL
from torchvision import transforms
import json
from tqdm import tqdm
from functools import partial
import transformers

try: # Wandb 임포트
	import wandb
	has_wandb = True
except ImportError:
	has_wandb = False

logger = logging.getLogger(__name__)

VALIDATION_PROMPTS = [
	"portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
	"Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
	"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
	"A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

# --- 헬퍼 함수 정의 ---
def check_gpu_compatibility(accelerator):
	"""설치된 PyTorch 버전과 사용 가능한 GPU의 호환성을 확인합니다. (수정됨: 상위 CC에 대해 경고만 발생)"""
	if not torch.cuda.is_available():
		accelerator.print("[WARN] CUDA not available. Skipping GPU compatibility check.")
		return True # CPU 환경 등에서는 통과

	try:
		gpu_properties = torch.cuda.get_device_properties(accelerator.device)
		cc_major = gpu_properties.major
		cc_minor = gpu_properties.minor
		current_cc = float(f"{cc_major}.{cc_minor}")

		# 현재 PyTorch 빌드가 지원하는 최대 Compute Capability 확인 시도
		supported_archs = torch.cuda.get_arch_list()
		# sm_XX 형태를 float 숫자로 변환 (예: 'sm_86' -> 8.6)
		supported_ccs = [float(arch.split('_')[1][:1] + '.' + arch.split('_')[1][1:]) for arch in supported_archs if arch.startswith('sm_')]

		max_supported_cc = max(supported_ccs) if supported_ccs else 0.0 # 지원 목록 없으면 0.0

		accelerator.print(f"Detected GPU: {gpu_properties.name} with Compute Capability {current_cc:.1f}")
		accelerator.print(f"Current PyTorch installation supports Compute Capabilities up to: {max_supported_cc:.1f} (based on arch list: {supported_archs})")

		# --- 수정된 호환성 체크 ---
		if current_cc > max_supported_cc and max_supported_cc > 0: # 지원 목록이 있고 현재 CC가 더 높으면
			 # 에러 대신 경고 출력
			 accelerator.print(f"[WARN] GPU Compute Capability ({current_cc:.1f}) is higher than the maximum explicitly supported by this PyTorch build ({max_supported_cc:.1f}).")
			 accelerator.print(f"[WARN] Proceeding, but unexpected behavior, errors during kernel execution, or performance issues might occur.")
			 accelerator.print(f"[WARN] Consider updating PyTorch if problems arise: https://pytorch.org/get-started/locally/")
			 return True # <<< 에러 대신 True를 반환하여 계속 진행
		elif current_cc < 3.7: # 너무 낮은 버전 (예시, 이 값은 조정 가능)
			 accelerator.print(f"[WARN] GPU Compute Capability ({current_cc:.1f}) might be low for optimal performance with some operations.")
			 return True # 경고만 하고 일단 진행
		else:
			 # 지원 범위 내에 있거나, 지원 목록이 없는 경우 (일단 통과)
			 accelerator.print("GPU compatibility check passed (or check inconclusive).")
			 return True

	except Exception as e:
		accelerator.print(f"[WARN] Failed to perform GPU compatibility check: {e}")
		return True # 확인 실패 시 일단 진행 (다른 에러가 발생할 수 있음


def check_and_prepare_lora(text_encoder_one, text_encoder_two, lora_path, logger_instance):
	"""기존 Text Encoder LoRA 가중치를 로드합니다."""
	logger_instance.info(f"Loading pretrained Text Encoder LoRA from: {lora_path}")
	if not os.path.exists(lora_path):
		raise FileNotFoundError(f"LoRA file not found at {lora_path}")

	try:
		lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora_path)
		LoraLoaderMixin.load_lora_into_text_encoder(
			lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one,
			prefix="text_encoder", lora_scale=1.0
		)
		logger_instance.info("Loaded LoRA into Text Encoder 1.")
		LoraLoaderMixin.load_lora_into_text_encoder(
			lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two,
			prefix="text_encoder_2", lora_scale=1.0
		)
		logger_instance.info("Loaded LoRA into Text Encoder 2.")
		del lora_state_dict
		return text_encoder_one, text_encoder_two
	except Exception as e:
		logger_instance.error(f"Could not load Text Encoder LoRA weights from {lora_path}: {e}")
		raise

from peft import get_peft_model_state_dict	

def extract_lora_weights(text_encoder_one, text_encoder_two, logger_instance):
	"""학습된 Text Encoder LoRA 가중치만 추출 (PEFT 기반)."""
	logger_instance.info("Extracting Text Encoder LoRA weights using PEFT...")
	try:
		state_dict_te1 = get_peft_model_state_dict(text_encoder_one)
		state_dict_te2 = get_peft_model_state_dict(text_encoder_two)
		final_state_dict = {}
		key_map_warnings = set()

		for prefix, state_dict in [("text_encoder", state_dict_te1), ("text_encoder_2", state_dict_te2)]:
			for key, value in state_dict.items():
				new_key = key.replace("base_model.model.", f"{prefix}.", 1)
				final_state_dict[new_key] = value.cpu().detach().clone()
				if new_key == key and "lora" in key and prefix not in key_map_warnings:
					logger_instance.warning(f"Potential key mapping issue for {prefix} LoRA: {key}")
					key_map_warnings.add(prefix)

		if not final_state_dict:
			 logger_instance.warning("No LoRA weights extracted using get_peft_model_state_dict.")
			 return {}
		logger_instance.info(f"Extracted {len(final_state_dict)} TE LoRA parameters using PEFT.")
		return final_state_dict
	except Exception as e:
		logger_instance.error(f"Error extracting LoRA weights using PEFT: {e}.")
		return {}

@torch.no_grad()
def log_validation(args: argparse.Namespace, text_encoder_one, text_encoder_two, unet, accelerator, weight_dtype, epoch, is_final_validation=False, logger_instance=None, validation_prompts=None):
	""" DPO 학습 검증 이미지를 로깅합니다. """
	if logger_instance is None: logger_instance = logger
	if validation_prompts is None: validation_prompts = VALIDATION_PROMPTS
	if not validation_prompts: return # 프롬프트 없으면 종료

	logger_instance.info(f"Running validation... Generating {len(validation_prompts)} images.")

	# VAE 로드 (검증에만 필요, 메모리 주의)
	vae = None
	try:
		# <<< VAE 경로 가져오기 수정 >>>
		# args에 pretrained_vae_model_name_or_path 가 있는지 확인
		vae_path_for_val = args.pretrained_vae_model_name_or_path
		if not vae_path_for_val:
			# 없으면 모델 경로에서 VAE 로드 시도 (예: diffusers 모델 디렉토리)
			# 또는 기본 SDXL VAE 사용
			# default_vae_id = "madebyollin/sdxl-vae-fp16-fix"
			default_vae_id = "stabilityai/sdxl-vae" # 안정적인 기본값
			logger_instance.warning(f"No VAE path specified for validation, using default: {default_vae_id}")
			vae_path_for_val = default_vae_id

		# <<< VAE 로딩 (FP16 권장) >>>
		vae_dtype = torch.float16 # 검증에는 FP16 사용
		vae = AutoencoderKL.from_pretrained(vae_path_for_val, torch_dtype=vae_dtype).to(accelerator.device)
		vae.eval()
		logger_instance.info(f"VAE loaded for validation from {vae_path_for_val}")
	except Exception as e:
		 logger_instance.error(f"Failed to load VAE for validation: {e}")
		 return # VAE 없으면 검증 불가

	# 파이프라인 생성 (Tokenizer는 필요 없음)
	# <<< unwrap_model 사용 필요 여부 확인 >>>
	# training_loop 에서 모델이 accelerator.prepare 되지 않았으므로 unwrap 불필요
	pipeline = StableDiffusionXLPipeline(
		vae=vae,
		text_encoder=text_encoder_one, # 이미 GPU에 있음
		text_encoder_2=text_encoder_two, # 이미 GPU에 있음
		unet=unet, # 이미 GPU에 있음
		tokenizer=None,
		tokenizer_2=None,
		scheduler=DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"), # 스케줄러는 새로 로드
	).to(accelerator.device) # 파이프라인 전체를 디바이스로
	pipeline.set_progress_bar_config(disable=True)

	# <<< LoRA 어댑터 상태 확인 >>>
	# LoRA가 적용된 상태에서 이미지 생성
	lora_loaded_for_val = hasattr(text_encoder_one, "lora_layer") or hasattr(text_encoder_two, "lora_layer") # PEFT 적용 여부 간이 확인

	images = []
	generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

	for i, prompt in enumerate(validation_prompts):
		with torch.cuda.amp.autocast(enabled=accelerator.mixed_precision == "fp16"): # AMP 사용
			try:
				image = pipeline(
					prompt,
					num_inference_steps=30, # 적절한 추론 스텝
					generator=generator,
					height=args.resolution_height, # 학습 시 사용한 해상도
					width=args.resolution_width,
					guidance_scale=7.0 # 일반적인 값
				).images[0]
				images.append(image)
			except Exception as e:
				logger_instance.warning(f"Validation image {i} generation failed for prompt '{prompt}': {e}")
				# 실패 시 빈 이미지 추가
				images.append(Image.new('RGB', (args.resolution_width, args.resolution_height), color = 'grey'))

	# 로깅
	log_prefix = "validation_final" if is_final_validation else f"validation_epoch_{epoch}"
	for tracker in accelerator.trackers:
		if tracker.name == "tensorboard":
			try:
				np_images = np.stack([np.asarray(img) for img in images])
				tracker.writer.add_images(log_prefix, np_images, epoch, dataformats="NHWC")
			except Exception as e: logger_instance.warning(f"Failed to log validation images to TensorBoard: {e}")
		if tracker.name == "wandb" and has_wandb:
			try:
				tracker.log({
					log_prefix: [
						wandb.Image(image, caption=f"{i}: {validation_prompts[i]}")
						for i, image in enumerate(images)
					]
				}, step=epoch) # step을 epoch 또는 global_step으로
			except Exception as e: logger_instance.warning(f"Failed to log validation images to wandb: {e}")

	# 비교 이미지 생성 (최종 검증 시) - LoRA 비활성화 후 생성
	if is_final_validation and lora_loaded_for_val:
		logger_instance.info("Generating comparison images without TE LoRA...")
		no_lora_images = []
		# <<< 어댑터 비활성화 시도 >>>
		can_disable_adapters = hasattr(pipeline.text_encoder, 'disable_adapter') and hasattr(pipeline.text_encoder_2, 'disable_adapter')
		if can_disable_adapters:
			try:
				pipeline.text_encoder.disable_adapter()
				pipeline.text_encoder_2.disable_adapter()
				logger_instance.info("TE adapters disabled for comparison.")

				generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None # 제너레이터 리셋
				for i, prompt in enumerate(validation_prompts):
					with torch.cuda.amp.autocast(enabled=accelerator.mixed_precision == "fp16"):
						try:
							 image = pipeline(prompt, num_inference_steps=30, generator=generator, height=args.resolution_height, width=args.resolution_width, guidance_scale=7.0).images[0]
							 no_lora_images.append(image)
						except Exception as e:
							 logger_instance.warning(f"Comparison image {i} generation failed: {e}")
							 no_lora_images.append(Image.new('RGB', (args.resolution_width, args.resolution_height), color = 'darkgrey'))

				# 비교 이미지 로깅
				for tracker in accelerator.trackers:
					if tracker.name == "tensorboard":
						try:
							np_no_lora_images = np.stack([np.asarray(img) for img in no_lora_images])
							tracker.writer.add_images("comparison_without_lora", np_no_lora_images, epoch, dataformats="NHWC")
						except Exception as e: logger_instance.warning(f"Failed to log comparison images to TensorBoard: {e}")
					if tracker.name == "wandb" and has_wandb:
						try:
							tracker.log({
								"comparison_without_lora": [
									wandb.Image(image, caption=f"{i}: {validation_prompts[i]}")
									for i, image in enumerate(no_lora_images)
								]
							}, step=epoch)
						except Exception as e: logger_instance.warning(f"Failed to log comparison images to wandb: {e}")

				# <<< 어댑터 다시 활성화 >>>
				if hasattr(pipeline.text_encoder, 'enable_adapter') and hasattr(pipeline.text_encoder_2, 'enable_adapter'):
					pipeline.text_encoder.enable_adapter()
					pipeline.text_encoder_2.enable_adapter()
					logger_instance.info("TE adapters re-enabled.")
				else:
					logger_instance.warning("Could not re-enable adapters automatically.")

			except Exception as e:
				logger_instance.warning(f"Could not disable/enable TE adapters or generate comparison images: {e}")
		else:
			logger_instance.warning("Text encoders do not support adapter disabling, skipping comparison image generation.")


	# 파이프라인 및 VAE 메모리 정리
	del pipeline, vae # <<< vae 도 여기서 삭제
	gc.collect()
	torch.cuda.empty_cache()
	logger_instance.info("Validation finished and resources released.")

# --- 프롬프트 인코딩 함수 ---
def encode_prompt(text_encoders: list, text_input_ids_list: list[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
	"""텍스트 인코더를 사용하여 프롬프트 임베딩을 계산합니다."""
	prompt_embeds_list = []
	pooled_prompt_embeds = None # pooled는 마지막 TE에서만 사용

	for i, text_encoder in enumerate(text_encoders):
		text_input_ids = text_input_ids_list[i]
		prompt_embeds_out = text_encoder(
			text_input_ids.to(text_encoder.device),
			output_hidden_states=True,
		)
		# We are only ALWAYS interested in the pooled output of the final text encoder (CLIPTextModelWithProjection)
		if isinstance(text_encoder.config, transformers.CLIPTextConfig) and text_encoder.config.output_hidden_states:
			 prompt_embeds = prompt_embeds_out.hidden_states[-2] # penultimate hidden state
		else: # 다른 모델 구조 대비
			 prompt_embeds = prompt_embeds_out.last_hidden_state

		if isinstance(text_encoder.config, transformers.CLIPVisionConfig) or \
		   (hasattr(text_encoder.config, 'projection_dim') and text_encoder.config.projection_dim > 0) or \
		   i == len(text_encoders) - 1: # 마지막 인코더 또는 Projection 있는 경우
			   pooled_prompt_embeds = prompt_embeds_out[0] # pooled output

		bs_embed, seq_len, _ = prompt_embeds.shape
		prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
		prompt_embeds_list.append(prompt_embeds)

	prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
	if pooled_prompt_embeds is not None:
		pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
	else: # 예외 처리: pooled output 없는 경우 (이론상 SDXL에선 발생 안 함)
		 pooled_prompt_embeds = torch.zeros((bs_embed, prompt_embeds.shape[-1]), device=prompt_embeds.device, dtype=prompt_embeds.dtype)


	return prompt_embeds, pooled_prompt_embeds

def check_positive(value):
	"""argparse 인자가 양의 정수인지 확인합니다."""
	try:
		ivalue = int(value)
		if ivalue <= 0:
			raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value. Must be >= 1.")
		return ivalue
	except ValueError:
		raise argparse.ArgumentTypeError(f"{value} is not an integer.")
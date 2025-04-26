import os
import torch
import logging
import numpy as np
from PIL import Image
from torchvision import transforms
import json
from tqdm import tqdm
import torch.multiprocessing as mp
from functools import partial
import argparse # argparse 임포트
import gc # gc 임포트

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL

# 멀티프로세싱 설정 (기존과 동일)
try:
	mp.set_start_method('spawn', force=True)
	print("[INFO] Set multiprocessing start method to 'spawn'.")
except RuntimeError as e:
	print(f"[WARN] Could not set multiprocessing start method to 'spawn': {e}")

# 로거 설정
logger = logging.getLogger(__name__)

# --- GPU 호환성 체크 함수 (dpo_helpers.py 에서 가져옴) ---
def check_gpu_compatibility(accelerator):
	if not torch.cuda.is_available():
		accelerator.print("[WARN] CUDA not available. Skipping GPU compatibility check.")
		return True
	try:
		gpu_properties = torch.cuda.get_device_properties(accelerator.device)
		current_cc = float(f"{gpu_properties.major}.{gpu_properties.minor}")
		supported_archs = torch.cuda.get_arch_list()
		supported_ccs = [float(arch.split('_')[1][:1] + '.' + arch.split('_')[1][1:]) for arch in supported_archs if arch.startswith('sm_')]
		max_supported_cc = max(supported_ccs) if supported_ccs else 0.0

		accelerator.print(f"Detected GPU: {gpu_properties.name} with Compute Capability {current_cc:.1f}")
		accelerator.print(f"Current PyTorch supports CC up to: {max_supported_cc:.1f} ({supported_archs})")

		if current_cc > max_supported_cc and max_supported_cc > 0:
			 accelerator.print(f"[WARN] GPU CC ({current_cc:.1f}) > max supported ({max_supported_cc:.1f}).")
			 accelerator.print(f"[WARN] Proceeding, but expect potential issues.")
			 return True
		elif current_cc < 3.7:
			 accelerator.print(f"[WARN] GPU CC ({current_cc:.1f}) might be low for optimal performance.")
			 return True
		else:
			 accelerator.print("GPU compatibility check passed.")
			 return True
	except Exception as e:
		accelerator.print(f"[WARN] Failed GPU compatibility check: {e}")
		return True

# --- 이미지 변환 함수 ---
def get_vae_transforms(resolution_height, resolution_width):
	resize_size = min(resolution_width, resolution_height)
	return transforms.Compose([
		transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
		transforms.CenterCrop((resolution_height, resolution_width)),
		transforms.ToTensor(), # [0, 1] 범위
	])

# --- 단일 샘플 Latent 생성 함수 ---
def prepare_latent_for_sample(sample_info, args, vae, image_transforms, device):
	img_rel_path = sample_info['img_rel_path']
	sample_index = sample_info['index']
	base_path = os.path.join(args.image_data_root, img_rel_path)
	cache_path = os.path.splitext(base_path)[0] + ".pt"

	if os.path.exists(cache_path): return True

	try:
		image = Image.open(base_path).convert("RGB")
		original_size = (image.height, image.width)
		pixel_values = image_transforms(image).unsqueeze(0).to(device, dtype=torch.float16)

		# Crop 정보 계산
		temp_pv = transforms.ToTensor()(image)
		resized_pv = transforms.Resize(min(args.resolution_width, args.resolution_height), interpolation=transforms.InterpolationMode.BILINEAR)(temp_pv)
		img_h, img_w = resized_pv.shape[1], resized_pv.shape[2]
		y1 = max(0, int(round((img_h - args.resolution_height) / 2.0)))
		x1 = max(0, int(round((img_w - args.resolution_width) / 2.0)))
		crop_top_left = (y1, x1)

		# VAE 인코딩
		with torch.no_grad():
			latent_dist = vae.encode(pixel_values).latent_dist
			latent = latent_dist.sample().detach()
			# <<< VAE 스케일링 계수 사용 확인 >>>
			# SDXL VAE는 config에 scaling_factor가 없을 수 있음. 직접 값 사용 또는 확인 필요
			# scaling_factor = getattr(vae.config, "scaling_factor", 0.18215) # SD 1.5 기준
			scaling_factor = getattr(vae.config, "scaling_factor", 0.13025) # SDXL VAE 기준 추정치 (확인 필요!)
			if scaling_factor != 0.13025:
				logger.warning(f"Using VAE scaling factor: {scaling_factor}. Ensure this is correct for the loaded VAE.")

			latent = (latent * scaling_factor).squeeze(0).to(dtype=torch.float16)
			original_size_tensor = torch.tensor(original_size, dtype=torch.int32)
			crop_top_left_tensor = torch.tensor(crop_top_left, dtype=torch.int32)

		os.makedirs(os.path.dirname(cache_path), exist_ok=True)
		data_to_save = {
			'latent': latent.cpu(), # CPU로 저장
			'original_size': original_size_tensor,
			'crop_top_left': crop_top_left_tensor
		}
		torch.save(data_to_save, cache_path)
		return True

	except FileNotFoundError:
		# 워커 프로세스에서는 logger 대신 print 사용이 더 간단할 수 있음
		print(f"[WARN][Worker] Img not found: {base_path}. Skipping sample {sample_index}.")
		return False
	except Exception as e:
		print(f"[ERROR][Worker] Error processing sample {sample_index} ({img_rel_path}): {e}")
		# traceback.print_exc() # 필요시 traceback 출력
		return False

# --- 전체 Latent 확인 및 생성 메인 함수 ---
def check_and_prepare_all_latents(args, accelerator):
	""" 데이터셋의 모든 이미지에 대한 Latent 캐시를 확인하고 필요시 생성합니다. """
	accelerator.print("--- Latent Cache Preparation ---")
	if not check_gpu_compatibility(accelerator): return False

	dataset_path = args.dataset_path
	image_data_root = args.image_data_root

	# VAE 경로 결정
	vae_path_arg = args.pretrained_vae_model_name_or_path
	# <<< SDXL VAE 기본값 사용 >>>
	default_vae_id = "madebyollin/sdxl-vae-fp16-fix" # 또는 stabilityai/sdxl-vae
	if vae_path_arg: vae_path = vae_path_arg
	else: vae_path = default_vae_id
	accelerator.print(f"Using VAE: {vae_path} for latent preparation.")

	# 필요한 작업 목록 생성
	tasks = []
	all_files_exist = True
	accelerator.print("Scanning dataset for missing latent caches...")
	try:
		line_count = 0 # 진행률 표시용
		# 파일 라인 수 먼저 세기 (선택적, tqdm total 위함)
		try:
			with open(dataset_path, 'r', encoding='utf-8') as f_count:
				total_lines = sum(1 for _ in f_count)
			accelerator.print(f"Scanning {total_lines} lines in dataset...")
		except Exception:
			total_lines = None # 라인 수 세기 실패 시 None
			accelerator.print("Scanning dataset (line count unknown)...")

		with open(dataset_path, 'r', encoding='utf-8') as f:
			# tqdm 추가
			line_iterator = tqdm(f, total=total_lines, desc="Scanning Dataset", disable=not accelerator.is_main_process)
			for i, line in enumerate(line_iterator):
				line_count += 1
				try:
					sample = json.loads(line)
					for key in ["image_0_relative_path", "image_1_relative_path"]:
						img_rel_path = sample.get(key)
						if img_rel_path:
							base_path = os.path.join(image_data_root, img_rel_path)
							cache_path = os.path.splitext(base_path)[0] + ".pt"
							if not os.path.exists(cache_path):
								tasks.append({'img_rel_path': img_rel_path, 'index': i})
								all_files_exist = False
						elif key in sample: # 경로 키는 있는데 값이 None 또는 빈 문자열인 경우
							logger.warning(f"Empty image path for key '{key}' in sample {i}")
							# all_files_exist = False # 빈 경로도 준비 안된 것으로 간주할 수 있음
						# else: 키 자체가 없는 경우는 무시 (선택적)
				except json.JSONDecodeError:
					 logger.warning(f"Skipping invalid JSON line {i+1}")
					 all_files_exist = False # 파싱 실패 시 준비 안됨 간주

		tasks = list({task['img_rel_path']: task for task in tasks}.values()) # 중복 제거

	except FileNotFoundError: logger.error(f"Dataset file not found: {dataset_path}"); return False
	except Exception as e: logger.error(f"Error scanning dataset: {e}"); return False

	if all_files_exist:
		accelerator.print("All required latent caches seem to exist.")
		accelerator.wait_for_everyone(); return True

	accelerator.print(f"Found {len(tasks)} unique images needing latent preparation.")
	if not tasks:
		accelerator.print("[WARN] No missing caches found, but initial scan reported missing. Proceeding cautiously.");
		accelerator.wait_for_everyone(); return True # 상태 불일치 가능성 있지만 진행

	# --- Latent 생성 실행 ---
	num_vae_workers = args.num_vae_workers
	vae = None
	image_transforms = get_vae_transforms(args.resolution_height, args.resolution_width)
	success_count = 0
	fail_count = len(tasks) # 실패 기본값

	# VAE 로딩은 메인 프로세스에서만
	if accelerator.is_main_process:
		try:
			accelerator.print(f"Loading VAE...")
			# <<< VAE 로딩 시 적절한 dtype 사용 >>>
			vae_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
			vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=vae_dtype).to(accelerator.device)
			vae.eval()
			accelerator.print("VAE loaded.")
		except Exception as e:
			accelerator.print(f"[ERROR] Failed to load VAE: {e}")
			accelerator.wait_for_everyone(); return False # VAE 로드 실패 시 종료

		# 병렬 또는 순차 처리
		if num_vae_workers > 0 and len(tasks) > 1: # 워커 수가 있고 작업이 2개 이상일 때만 병렬 처리 의미 있음
			accelerator.print(f"Starting parallel latent preparation ({num_vae_workers} workers)...")
			# <<< partial 함수 생성 시 accelerator.device 사용 >>>
			process_func = partial(prepare_latent_for_sample, args=args, vae=vae, image_transforms=image_transforms, device=accelerator.device)
			results = []
			try:
				# <<< torch.no_grad 컨텍스트 추가 >>>
				with torch.no_grad(), mp.Pool(processes=num_vae_workers) as pool:
					with tqdm(total=len(tasks), desc=f"Preparing Latents ({num_vae_workers} Workers)", disable=not accelerator.is_main_process) as pbar:
						for result in pool.imap_unordered(process_func, tasks):
							results.append(result)
							pbar.update()
				success_count = sum(1 for r in results if r is True)
				fail_count = len(results) - success_count
				accelerator.print(f"Parallel preparation finished. Success: {success_count}, Failed: {fail_count}")
			except Exception as e:
				accelerator.print(f"[ERROR] Parallel preparation error: {e}", exc_info=True)
				success_count = 0; fail_count = len(tasks)
		else: # 순차 처리
			accelerator.print("Starting sequential latent preparation on main process...")
			results = []
			with torch.no_grad(): # 순차 처리도 no_grad 사용
				with tqdm(total=len(tasks), desc="Preparing Latents (Sequential)", disable=not accelerator.is_main_process) as pbar:
					for task in tasks:
						result = prepare_latent_for_sample(task, args=args, vae=vae, image_transforms=image_transforms, device=accelerator.device)
						results.append(result)
						pbar.update()
			success_count = sum(1 for r in results if r is True)
			fail_count = len(results) - success_count
			accelerator.print(f"Sequential preparation finished. Success: {success_count}, Failed: {fail_count}")

		# VAE 메모리 해제
		del vae; gc.collect(); torch.cuda.empty_cache()
		accelerator.print("VAE unloaded.")

	# 모든 프로세스 동기화
	accelerator.wait_for_everyone()

	# 최종 확인 (캐시 생성 후 다시 스캔)
	accelerator.print("Verifying latent cache completeness after preparation...")
	final_check_ok = True
	try:
		# 파일을 다시 열어서 확인
		with open(dataset_path, 'r', encoding='utf-8') as f_verify:
			# tqdm 추가
			verify_iterator = tqdm(f_verify, total=total_lines, desc="Verifying Caches", disable=not accelerator.is_main_process)
			for i, line in enumerate(verify_iterator):
				try:
					sample = json.loads(line)
					for key in ["image_0_relative_path", "image_1_relative_path"]:
						img_rel_path = sample.get(key)
						if img_rel_path:
							cache_path = os.path.splitext(os.path.join(image_data_root, img_rel_path))[0] + ".pt"
							if not os.path.exists(cache_path):
								# 메인 프로세스에서만 에러 로깅
								if accelerator.is_main_process:
									logger.error(f"Verification failed: Cache still missing for sample {i} ({cache_path})")
								final_check_ok = False
								# break # 하나라도 없으면 중단 가능 (선택적)
				except json.JSONDecodeError: pass # 이미 스캔 시 경고함
	except Exception as e:
		if accelerator.is_main_process: logger.error(f"Error during final verification: {e}")
		final_check_ok = False

	# 최종 결과 반환
	if final_check_ok:
		accelerator.print("Latent cache preparation and verification successful.")
		return True
	else:
		accelerator.print("[ERROR] Latent cache verification failed after preparation attempt.")
		return False

# --- 스크립트로 직접 실행될 때의 로직 (선택 사항) ---
if __name__ == "__main__":
	# 이 스크립트를 직접 실행할 경우를 위한 인자 파서
	parser = argparse.ArgumentParser(description="Prepare latent caches for DPO training dataset.")
	parser.add_argument("--dataset_path", type=str, required=True, help="Path to the preference dataset in .jsonl format.")
	parser.add_argument("--image_data_root", type=str, required=True, help="Root directory containing image folders/files.")
	parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None, help="Optional: Path to a specific VAE.")
	parser.add_argument("--resolution_width", type=int, default=1024, help="Target width for VAE encoding.")
	parser.add_argument("--resolution_height", type=int, default=1024, help="Target height for VAE encoding.")
	parser.add_argument("--num_vae_workers", type=int, default=1, help="Number of workers for parallel processing (0 for sequential).")
	parser.add_argument("--mixed_precision", type=str, default="fp16", help="Mixed precision for VAE loading ('no', 'fp16', 'bf16').")
	# Accelerator 초기화에 필요한 최소 인자들
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1) # 사용 안 하지만 Accelerator 초기화에 필요할 수 있음
	parser.add_argument("--log_with", type=str, default="tensorboard") # 사용 안 하지만 Accelerator 초기화에 필요할 수 있음
	parser.add_argument("--project_dir", type=str, default="logs_make_latents") # 임시 로그 디렉토리

	args = parser.parse_args()

	# 로깅 설정
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	# Accelerator 초기화 (간단하게)
	# project_config 에 project_dir 전달
	accelerator = Accelerator(
		mixed_precision=args.mixed_precision,
		log_with=args.log_with,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		project_config=ProjectConfiguration(project_dir=args.project_dir)
	)

	# Latent 생성 함수 호출
	success = check_and_prepare_all_latents(args, accelerator)

	if success:
		accelerator.print("Latent preparation finished successfully.")
		exit(0)
	else:
		accelerator.print("Latent preparation failed.")
		exit(1)

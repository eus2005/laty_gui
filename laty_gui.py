import os
import sys
import subprocess
import platform
import json
import threading
import shutil
import time
import re
import glob

try:
	import gradio as gr
except ImportError as e:
	print(f"오류: Gradio 라이브러리를 찾을 수 없습니다 - {e}")
	sys.exit(1)

# --- 기본 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 스크립트 경로 기본값 (나중에 선택에 따라 변경됨)
# TRAINING_SCRIPT_PATH = os.path.join('Main', 'train_unet_dpo.py') # 기본은 UNet으로 가정

def find_file(directory, extensions):
	"""주어진 디렉토리에서 지정된 확장자를 가진 첫 번째 파일을 찾습니다."""
	if not os.path.isdir(directory):
		print(f"경고: 디렉토리를 찾을 수 없습니다: {directory}")
		return None
	for ext in extensions:
		files = glob.glob(os.path.join(directory, f"*{ext}"))
		if files:
			files.sort()
			print(f"파일 찾음: {files[0]} (in {directory})")
			return files[0]
	print(f"경고: {directory} 에서 {extensions} 확장자를 가진 파일을 찾을 수 없습니다.")
	return None

# --- 고정 경로 설정 ---
APP_DIR = "/workspace/laty_gui"
CONTAINER_MODELS_DIR = os.path.join(APP_DIR, "Models")
CONTAINER_LORA_DIR = os.path.join(APP_DIR, "Lora")
CONTAINER_DATASETS_DIR = os.path.join(APP_DIR, "Datasets")
CONTAINER_OUTPUT_DIR = os.path.join(APP_DIR, "Output")
CONTAINER_CONFIG_DIR = os.path.join(APP_DIR, "Config")

fixed_pretrained_checkpoint_path = find_file(CONTAINER_MODELS_DIR, ['.safetensors', '.ckpt'])
fixed_pretrained_lora_path = find_file(CONTAINER_LORA_DIR, ['.safetensors', '.pt', '.bin'])
fixed_preference_dataset_path = find_file(CONTAINER_DATASETS_DIR, ['.jsonl'])
fixed_model_save_path = CONTAINER_OUTPUT_DIR
fixed_config_path = os.path.join(CONTAINER_CONFIG_DIR, "dpo_lora_finetune_settings.json")

os.makedirs(CONTAINER_CONFIG_DIR, exist_ok=True)

# --- 기본값 설정 ---
default_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
default_lora_target = "UNet" # <<< LoRA 학습 대상 기본값
default_unet_learning_rate = "1e-6" # <<< UNet용 기본 학습률
default_te_learning_rate = "5e-7"  # <<< Text Encoder용 기본 학습률
default_resolution_width = 512
default_resolution_height = 512
default_lr_scheduler = "cosine_with_restarts"
default_lr_warmup_steps = 100
default_optimizer_type = "AdamW8bit"
default_max_token_length = 75
default_noise_offset = 0.0
# default_use_xformers = True
default_use_shared_unet = True # UNet 학습 시에만 의미 있음

# 파일 경로 유효성 검사
# ... (경고 메시지 출력은 이전과 동일) ...
if fixed_pretrained_checkpoint_path is None: print(f"경고: {CONTAINER_MODELS_DIR} 에 모델 파일이 없습니다.")
if fixed_pretrained_lora_path is None: print(f"경고: {CONTAINER_LORA_DIR} 에 LoRA 파일이 없습니다.")
if fixed_preference_dataset_path is None: print(f"경고: {CONTAINER_DATASETS_DIR} 에 데이터셋 파일이 없습니다.")


def strip_ansi_codes(s):
	if isinstance(s, str):
		ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
		return ansi_escape.sub('', s)
	return s

# --- Gradio 앱 로직 함수 ---

def get_ui_settings(
	# 인자 목록에 lora_target 추가
	lora_target, base_model_id,
	num_epochs, batch_size, learning_rate, optimizer_type,
	dpo_beta, seed, gradient_accumulation_steps,
	resolution_width, resolution_height, checkpointing_steps, max_grad_norm,
	use_fp16, gradient_checkpointing, use_shared_unet,
	lr_scheduler, lr_warmup_steps, lr_scheduler_num_cycles, lr_scheduler_power,
	# use_xformers,
	max_token_length, noise_offset
):
	"""Gradio UI에서 설정 값을 수집하여 딕셔너리로 반환합니다."""
	checkpoint_path = fixed_pretrained_checkpoint_path if fixed_pretrained_checkpoint_path else "모델_파일_없음"
	lora_path = fixed_pretrained_lora_path if fixed_pretrained_lora_path else "LoRA_파일_없음"
	dataset_path = fixed_preference_dataset_path if fixed_preference_dataset_path else "데이터셋_파일_없음"

	settings = {
		'loraTarget': lora_target, # <<< LoRA 타겟 추가
		'pretrainedCheckpointPath': checkpoint_path,
		'baseModelId': base_model_id,
		'pretrainedLoraPath': lora_path,
		'preferenceDatasetPath': dataset_path,
		'modelSavePath': fixed_model_save_path,
		'numEpochs': int(num_epochs),
		'batchSize': int(batch_size),
		'learningRate': learning_rate, # UI에서 받은 값 그대로 사용
		'optimizerType': optimizer_type,
		'dpoBeta': float(dpo_beta),
		'seed': int(seed),
		'gradientAccumulationSteps': int(gradient_accumulation_steps),
		'resolutionWidth': int(resolution_width),
		'resolutionHeight': int(resolution_height),
		'checkpointingSteps': int(checkpointing_steps),
		'maxGradNorm': float(max_grad_norm),
		'useFp16': use_fp16,
		'gradientCheckpointing': gradient_checkpointing,
		# use_shared_unet는 UNet 학습 시에만 의미 있으므로 여기서 조건부 처리 가능
		'useSharedUnet': use_shared_unet if lora_target == "UNet" else False,
		'use_8bit_adam': (optimizer_type == 'AdamW8bit'),
		'lrScheduler': lr_scheduler,
		'lrWarmupSteps': int(lr_warmup_steps),
		'lrSchedulerNumCycles': int(lr_scheduler_num_cycles),
		'lrSchedulerPower': float(lr_scheduler_power),
		# 'useXformers': use_xformers,
		'maxTokenLength': int(max_token_length),
		'noiseOffset': float(noise_offset),
	}
	return settings

def apply_settings_to_ui_updates_dict(settings):
	"""설정 딕셔너리에서 값을 읽어 Gradio UI 컴포넌트 업데이트 객체를 반환합니다."""
	updates = {}
	# --- LoRA 타겟 설정 ---
	lora_target_val = settings.get('loraTarget', default_lora_target)
	updates['loraTarget'] = gr.update(value=lora_target_val)

	updates['baseModelId'] = gr.update(value=settings.get('baseModelId', default_base_model_id))
	num_epochs_val = settings.get('numEpochs', 3)
	batch_size_val = settings.get('batchSize', 1)
	# --- 학습률: 타겟에 따라 다른 기본값 적용 ---
	default_lr = default_unet_learning_rate if lora_target_val == "UNet" else default_te_learning_rate
	learning_rate_val = settings.get('learningRate', default_lr)

	optimizer_type_val = settings.get('optimizerType', default_optimizer_type)
	dpo_beta_val = settings.get('dpoBeta', 0.1)
	seed_val = settings.get('seed', 1)
	gradient_accumulation_steps_val = settings.get('gradientAccumulationSteps', 4)
	resolution_width_val = settings.get('resolutionWidth', default_resolution_width)
	resolution_height_val = settings.get('resolutionHeight', default_resolution_height)
	checkpointing_steps_val = settings.get('checkpointingSteps', 200)
	max_grad_norm_val = settings.get('maxGradNorm', 1.0)
	use_fp16_val = settings.get('useFp16', True)
	gradient_checkpointing_val = settings.get('gradientCheckpointing', False)
	# --- use_shared_unet: UNet 타겟일 때만 의미있는 기본값 적용 ---
	use_shared_unet_val = settings.get('useSharedUnet', default_use_shared_unet if lora_target_val == "UNet" else False)

	lr_scheduler_val = settings.get('lrScheduler', default_lr_scheduler)
	lr_warmup_steps_val = settings.get('lrWarmupSteps', default_lr_warmup_steps)
	lr_scheduler_num_cycles_val = settings.get('lrSchedulerNumCycles', 1)
	lr_scheduler_power_val = settings.get('lrSchedulerPower', 1.0)
	# use_xformers_val = settings.get('useXformers', default_use_xformers)
	max_token_length_val = settings.get('maxTokenLength', default_max_token_length)
	noise_offset_val = settings.get('noiseOffset', default_noise_offset)

	if optimizer_type_val is None:
		optimizer_type_val = 'AdamW8bit' if settings.get('use_8bit_adam', False) else 'AdamW'
	elif 'optimizerType' not in settings and settings.get('use_8bit_adam', False):
		optimizer_type_val = 'AdamW8bit'

	updates['numEpochs'] = gr.update(value=num_epochs_val)
	updates['batchSize'] = gr.update(value=batch_size_val)
	updates['learningRate'] = gr.update(value=learning_rate_val)
	updates['optimizerType'] = gr.update(value=optimizer_type_val)
	updates['dpoBeta'] = gr.update(value=dpo_beta_val)
	updates['seed'] = gr.update(value=seed_val)
	updates['gradientAccumulationSteps'] = gr.update(value=gradient_accumulation_steps_val)
	updates['resolutionWidth'] = gr.update(value=resolution_width_val)
	updates['resolutionHeight'] = gr.update(value=resolution_height_val)
	updates['checkpointingSteps'] = gr.update(value=checkpointing_steps_val)
	updates['maxGradNorm'] = gr.update(value=max_grad_norm_val)
	updates['useFp16'] = gr.update(value=use_fp16_val)
	updates['gradientCheckpointing'] = gr.update(value=gradient_checkpointing_val)
	# --- use_shared_unet UI 업데이트 (타겟에 따라 활성화/비활성화 및 값 설정) ---
	updates['useSharedUnet'] = gr.update(value=use_shared_unet_val, interactive=(lora_target_val == "UNet"))
	updates['lrScheduler'] = gr.update(value=lr_scheduler_val)
	updates['lrWarmupSteps'] = gr.update(value=lr_warmup_steps_val)
	updates['lrSchedulerNumCycles'] = gr.update(value=lr_scheduler_num_cycles_val)
	updates['lrSchedulerPower'] = gr.update(value=lr_scheduler_power_val)
	# updates['useXformers'] = gr.update(value=use_xformers_val)
	updates['maxTokenLength'] = gr.update(value=max_token_length_val)
	updates['noiseOffset'] = gr.update(value=noise_offset_val)

	return updates

def save_settings_direct(
	# 인자 목록에 lora_target 추가
	lora_target, base_model_id,
	num_epochs, batch_size, learning_rate, optimizer_type,
	dpo_beta, seed, gradient_accumulation_steps,
	resolution_width, resolution_height, checkpointing_steps, max_grad_norm,
	use_fp16, gradient_checkpointing, use_shared_unet,
	lr_scheduler, lr_warmup_steps, lr_scheduler_num_cycles, lr_scheduler_power,
	# use_xformers,
	max_token_length, noise_offset
):
	"""설정을 JSON 파일로 저장하고 파일 경로를 반환합니다."""
	file_path = fixed_config_path
	try:
		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		settings_dict = get_ui_settings( # get_ui_settings 사용
			lora_target, base_model_id, num_epochs, batch_size, learning_rate, optimizer_type,
			dpo_beta, seed, gradient_accumulation_steps, resolution_width, resolution_height,
			checkpointing_steps, max_grad_norm, use_fp16, gradient_checkpointing, use_shared_unet,
			lr_scheduler, lr_warmup_steps, lr_scheduler_num_cycles, lr_scheduler_power,
			# use_xformers,
			max_token_length, noise_offset
		)
		keys_to_remove = ['pretrainedCheckpointPath', 'pretrainedLoraPath', 'preferenceDatasetPath', 'modelSavePath', 'use_8bit_adam']
		settings_to_save = {k: v for k, v in settings_dict.items() if k not in keys_to_remove}
		with open(file_path, 'w', encoding='utf-8') as f:
			json.dump(settings_to_save, f, indent=2, ensure_ascii=False)
		print(f"설정이 '{os.path.basename(file_path)}' ({file_path})에 성공적으로 저장되었습니다.")
		return file_path
	except Exception as e:
		print(f"설정 저장 중 오류 발생 ({file_path}): {e}")
		return None

def load_settings_gradio_for_list_outputs():
	"""설정 파일을 불러와 Gradio UI 컴포넌트 업데이트 객체 튜플을 반환합니다."""
	output_key_order = [
		'loraTarget', 'baseModelId', # <<< loraTarget 추가
		'numEpochs', 'batchSize', 'learningRate', 'optimizerType',
		'dpoBeta', 'seed', 'gradientAccumulationSteps',
		'resolutionWidth', 'resolutionHeight', 'checkpointingSteps', 'maxGradNorm',
		'useFp16', 'gradientCheckpointing', 'useSharedUnet',
		'lrScheduler', 'lrWarmupSteps', 'lrSchedulerNumCycles', 'lrSchedulerPower',
		'useXformers', 'maxTokenLength', 'noiseOffset',
	]
	updates_dict = apply_settings_to_ui_updates_dict({}) # 기본값으로 시작
	file_path = fixed_config_path
	try:
		if os.path.exists(file_path):
			with open(file_path, 'r', encoding='utf-8') as f:
				settings_from_file = json.load(f)
			# 파일에서 읽은 값으로 업데이트 (apply_settings_to_ui_updates_dict가 loraTarget에 따라 LR, useSharedUnet 처리)
			loaded_settings_updates = apply_settings_to_ui_updates_dict(settings_from_file)
			updates_dict.update(loaded_settings_updates)
			print(f"설정을 '{os.path.basename(file_path)}' ({file_path})에서 성공적으로 불러왔습니다.")
		else:
			print(f"알림: 설정 파일 '{os.path.basename(file_path)}' ({file_path})을(를) 찾을 수 없습니다. 기본값으로 시작합니다.")
	except Exception as e:
		print(f"설정 불러오기 중 오류 발생 ({file_path}): {e}")
	output_list = [updates_dict.get(key, gr.update()) for key in output_key_order]
	return tuple(output_list)

# run_training_gradio 함수 수정
def run_training_gradio(settings_dict):
	print(f"--- run_training_gradio CALLED at {time.time()} ---") # 호출 시점 확인용

	try:
		# ... (파일 경로 유효성 검사는 동일) ...
		required_paths = {'모델': settings_dict.get('pretrainedCheckpointPath'), 'LoRA': settings_dict.get('pretrainedLoraPath'), '데이터셋': settings_dict.get('preferenceDatasetPath')}
		missing_files = [name for name, path in required_paths.items() if path is None or path.endswith("_파일_없음") or not os.path.exists(path)]
		if missing_files:
			error_msg = f"오류: 파일 없음 - {', '.join(missing_files)}. 볼륨 매핑 확인."
			print(error_msg)
			return gr.update(value=error_msg), gr.update(interactive=True)

		lora_target = settings_dict.get('loraTarget', 'UNet') # 설정에서 타겟 읽기
		print(f"--- LoRA 학습 대상: {lora_target} ---")

		# --- 실행할 스크립트 결정 (Text Encoder 선택 시 수정된 스크립트 사용) ---
		if lora_target == "UNet":
			# training_script_name = 'train_unet_dpo.py' # 또는 공식 UNet DPO 스크립트
			print("경고: UNet DPO 학습은 현재 이 GUI에서 지원되지 않습니다.")
			return gr.update(value="오류: UNet DPO 미지원"), gr.update(interactive=True)
		elif lora_target == "Text Encoder":
			# 분리된 새 스크립트 이름으로 변경 (예시)
			training_script_name = 'train_text_encoder_dpo.py' # 분리된 메인 스크립트 파일명
		else:
			error_msg = f"오류: 알 수 없는 LoRA 학습 대상입니다: {lora_target}"
			print(error_msg)
			return gr.update(value=error_msg), gr.update(interactive=True)

		training_script_rel_path = os.path.join('Main', training_script_name)
		training_script_abs_path = os.path.join(APP_DIR, training_script_rel_path)
		print(f"--- 학습 스크립트 경로: {training_script_abs_path} ---")
		if not os.path.exists(training_script_abs_path):
			error_msg = f"오류: 학습 스크립트({training_script_name})를 찾을 수 없습니다."
			print(error_msg)
			return gr.update(value=error_msg), gr.update(interactive=True)

		accelerate_cmd = shutil.which("accelerate")
		if not accelerate_cmd: print("오류: 'accelerate' 없음."); return gr.update(value="오류: accelerate 없음"), gr.update(interactive=True)
		print(f"--- accelerate 경로: {accelerate_cmd} ---")

		# 명령어 생성
		cmd = [accelerate_cmd, 'launch']
		gui_mixed_precision = 'fp16' if settings_dict.get('useFp16') else 'no'
		cmd.extend(['--mixed_precision', gui_mixed_precision])
		script_mixed_precision_arg = gui_mixed_precision
		cmd.append(training_script_abs_path)

		# --- 필수 인수 전달 ---
		cmd.extend(['--pretrained_model_name_or_path', settings_dict['pretrainedCheckpointPath']])
		cmd.extend(['--base_model_id_for_tokenizer', settings_dict['baseModelId']])
		cmd.extend(['--image_data_root', CONTAINER_DATASETS_DIR])
		cmd.extend(['--pretrained_lora_path', settings_dict['pretrainedLoraPath']])
		cmd.extend(['--dataset_path', settings_dict['preferenceDatasetPath']])
		output_path = os.path.join(settings_dict['modelSavePath'], f"{lora_target.lower().replace(' ', '_')}_dpo_output")
		os.makedirs(output_path, exist_ok=True)
		cmd.extend(['--output_dir', output_path])
		 # VAE 경로: 스크립트가 체크포인트에서 로드 시도, 실패 시 base_model_id 사용 가능
		# cmd.extend(['--pretrained_vae_model_name_or_path', ...]) # 필요시 전달
		# 나머지 인수들 (width, height 포함)
		cmd.extend(['--num_train_epochs', str(settings_dict['numEpochs'])])
		cmd.extend(['--train_batch_size', str(settings_dict['batchSize'])])
		cmd.extend(['--learning_rate', str(settings_dict['learningRate'])])
		cmd.extend(['--beta_dpo', str(settings_dict['dpoBeta'])])
		cmd.extend(['--seed', str(settings_dict['seed'])])
		cmd.extend(['--gradient_accumulation_steps', str(settings_dict['gradientAccumulationSteps'])])
		cmd.extend(['--resolution_width', str(settings_dict['resolutionWidth'])]) # width 전달
		cmd.extend(['--resolution_height', str(settings_dict['resolutionHeight'])]) # height 전달
		cmd.extend(['--max_grad_norm', str(settings_dict['maxGradNorm'])])
		cmd.extend(['--checkpointing_steps', str(settings_dict['checkpointingSteps'])])
		cmd.extend(['--lr_scheduler', settings_dict['lrScheduler']])
		cmd.extend(['--lr_warmup_steps', str(settings_dict['lrWarmupSteps'])])
		cmd.extend(['--lr_scheduler_num_cycles', str(settings_dict['lrSchedulerNumCycles'])])
		cmd.extend(['--lr_scheduler_power', str(settings_dict['lrSchedulerPower'])])
		cmd.extend(['--mixed_precision', script_mixed_precision_arg])

		# Boolean/Flag 인자
		if settings_dict.get('gradientCheckpointing'): cmd.append('--gradient_checkpointing')
		if settings_dict.get('optimizerType') == 'AdamW8bit': cmd.append('--use_8bit_adam')
		# if settings_dict.get('useXformers'): cmd.append('--enable_xformers_memory_efficient_attention')

		print(f"--- 최종 실행 명령어:\n{' '.join(cmd)}")
		print("-" * (40 + len(f" {lora_target} LoRA DPO Fine-tuning 시작 ")))
		sys.stdout.flush()

		# Subprocess 실행 및 로그 스트리밍
		process = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
								   text=True, bufsize=1, universal_newlines=True)
		while True:
			stdout_line = process.stdout.readline()
			stderr_line = process.stderr.readline()
			if not stdout_line and not stderr_line and process.poll() is not None: break
			if stdout_line: print(strip_ansi_codes(stdout_line.strip())); sys.stdout.flush()
			if stderr_line: print(f"[STDERR] {strip_ansi_codes(stderr_line.strip())}"); sys.stderr.flush()
			time.sleep(0.01)
		remaining_stdout = process.stdout.read(); remaining_stderr = process.stderr.read()
		if remaining_stdout: print(strip_ansi_codes(remaining_stdout.strip())); sys.stdout.flush()
		if remaining_stderr: print(f"[STDERR] {strip_ansi_codes(remaining_stderr.strip())}"); sys.stderr.flush()
		returncode = process.wait()

		if returncode == 0:
			print("="*20 + f" {lora_target} DPO Fine-tuning 완료 " + "="*20 + "\n")
			return gr.update(value="완료 (터미널/로그 확인)"), gr.update(interactive=True)
		else:
			error_msg = f"오류: {lora_target} DPO 미세조정 스크립트 오류 (Exit code: {returncode})"
			print(error_msg)
			return gr.update(value=f"오류 발생 (종료 코드: {returncode})"), gr.update(interactive=True)

	except Exception as e:
		error_msg = f"오류: Gradio 처리 중 예외 발생: {e}"
		print(error_msg); import traceback; traceback.print_exc()
		return gr.update(value=f"예외 발생: {e}"), gr.update(interactive=True)
	finally:
		print("--- run_training_gradio 함수 종료 ---")
		sys.stdout.flush()


# --- Gradio UI 정의 ---
with gr.Blocks(title="DPO LoRA 미세조정 GUI") as demo:
	gr.Markdown("# DPO 기반 LoRA 미세조정 GUI")
	gr.Markdown(f"""
	**중요:** Docker 환경 실행 가정. 필요한 파일을 지정된 경로에 매핑하세요.
	*   모델: `{CONTAINER_MODELS_DIR}` | LoRA: `{CONTAINER_LORA_DIR}` | 데이터셋: `{CONTAINER_DATASETS_DIR}`
	*   설정: `{fixed_config_path}` | 출력: `{CONTAINER_OUTPUT_DIR}` (타겟별 하위 폴더 생성)
	""")

	settings_state = gr.State()

	with gr.Tab("설정"):
		gr.Markdown(f"### 설정 관리")
		with gr.Row():
			load_settings_button = gr.Button("설정 불러오기")
			save_settings_button = gr.Button("설정 저장")
		download_config_file = gr.File(label="설정 파일 다운로드", interactive=False, value=None)

		# --- LoRA 학습 대상 선택 UI ---
		with gr.Row():
			lora_target_dropdown = gr.Dropdown(
				label="LoRA 학습 대상",
				choices=["UNet", "Text Encoder"],
				value=default_lora_target,
				info="DPO로 개선할 LoRA의 종류를 선택하세요."
			)

		gr.Markdown("### 모델 및 데이터 경로 (자동 감지)")
		base_model_id_textbox = gr.Textbox(label="호환 베이스 ID/경로:", value=default_base_model_id)

		gr.Markdown("### 기본 학습 파라미터")
		with gr.Row():
			num_epochs_number = gr.Number(label="에포크:", value=3, precision=0, minimum=1)
			batch_size_number = gr.Number(label="배치:", value=1, precision=0, minimum=1)
			gradient_accumulation_steps_number = gr.Number(label="축적 스텝:", value=4, precision=0, minimum=1)
		with gr.Row():
			dpo_beta_number = gr.Number(label="DPO Beta:", value=0.1, minimum=0.0)
			seed_number = gr.Number(label="시드:", value=1, precision=0)
			max_grad_norm_number = gr.Number(label="최대 Grad Norm:", value=1.0, minimum=0.0)
		with gr.Row():
			resolution_width_number = gr.Number(label="너비:", value=default_resolution_width, precision=0, minimum=64, step=8)
			resolution_height_number = gr.Number(label="높이:", value=default_resolution_height, precision=0, minimum=64, step=8)
			checkpointing_steps_number = gr.Number(label="체크포인트 주기(스텝):", value=200, precision=0, minimum=0)

		gr.Markdown("### 옵티마이저 및 학습률")
		with gr.Row():
			# 학습률 입력 컴포넌트 (기본값은 LoRA 타겟 변경 시 업데이트됨)
			learning_rate_textbox = gr.Textbox(label="학습률:", value=default_unet_learning_rate, placeholder="예: 1e-6")
			optimizer_type_dropdown = gr.Dropdown(label="옵티마이저:", choices=["AdamW", "AdamW8bit"], value=default_optimizer_type)
		with gr.Row():
			lr_scheduler_dropdown = gr.Dropdown(label="LR 스케줄러:", choices=["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"], value=default_lr_scheduler)
			lr_warmup_steps_number = gr.Number(label="LR 웜업 스텝:", value=default_lr_warmup_steps, precision=0, minimum=0)
		with gr.Row():
			lr_scheduler_num_cycles_number = gr.Number(label="LR 주기(cosine):", value=1, precision=0, minimum=1)
			lr_scheduler_power_number = gr.Number(label="LR Power(poly):", value=1.0, minimum=0.0)

		gr.Markdown("### 고급 설정")
		with gr.Row():
			use_fp16_checkbox = gr.Checkbox(label="FP16/BF16", value=True)
			gradient_checkpointing_checkbox = gr.Checkbox(label="Grad Ckpt", value=False)
			# use_xformers_checkbox = gr.Checkbox(label="XFormers", value=default_use_xformers)
		with gr.Row():
			# UNet 공유 체크박스 (LoRA 타겟 변경 시 상호작용 상태 변경)
			use_shared_unet_checkbox = gr.Checkbox(label="UNet 공유(UNet 학습 시)", value=default_use_shared_unet, interactive=(default_lora_target == "UNet"))
			max_token_length_number = gr.Number(label="최대 토큰:", value=default_max_token_length, precision=0, minimum=75)
			noise_offset_number = gr.Number(label="노이즈 오프셋:", value=default_noise_offset, minimum=0.0)


	with gr.Tab("미세조정 실행"):
		gr.Markdown("### 미세조정 시작")
		# ... (안내 문구) ...
		gr.Markdown("<p style='color: red;'><b>학습 진행 로그는 터미널 또는 Docker 로그에 출력됩니다.</b></p>")
		status_textbox = gr.Textbox(label="상태", value="대기 중...", interactive=False)
		run_training_button = gr.Button("미세조정 시작", variant="primary")


	# --- 이벤트 핸들러 연결 ---
	non_path_input_components = [
		lora_target_dropdown, base_model_id_textbox, # <<< lora_target 추가
		num_epochs_number, batch_size_number, learning_rate_textbox, optimizer_type_dropdown,
		dpo_beta_number, seed_number, gradient_accumulation_steps_number,
		resolution_width_number, resolution_height_number, checkpointing_steps_number, max_grad_norm_number,
		use_fp16_checkbox, gradient_checkpointing_checkbox, use_shared_unet_checkbox,
		lr_scheduler_dropdown, lr_warmup_steps_number, lr_scheduler_num_cycles_number, lr_scheduler_power_number,
		# use_xformers_checkbox,
		max_token_length_number, noise_offset_number,
	]
	load_outputs_list = [
		lora_target_dropdown, base_model_id_textbox, # <<< lora_target 추가
		num_epochs_number, batch_size_number, learning_rate_textbox, optimizer_type_dropdown,
		dpo_beta_number, seed_number, gradient_accumulation_steps_number,
		resolution_width_number, resolution_height_number, checkpointing_steps_number, max_grad_norm_number,
		use_fp16_checkbox, gradient_checkpointing_checkbox, use_shared_unet_checkbox,
		lr_scheduler_dropdown, lr_warmup_steps_number, lr_scheduler_num_cycles_number, lr_scheduler_power_number,
		# use_xformers_checkbox,
		max_token_length_number, noise_offset_number,
	]

	# LoRA 타겟 변경 시 학습률 기본값 및 UNet 공유 체크박스 활성화 상태 변경
	def update_ui_on_lora_target_change(lora_target):
		new_lr = default_unet_learning_rate if lora_target == "UNet" else default_te_learning_rate
		unet_share_interactive = (lora_target == "UNet")
		# UNet 타겟이 아니면 UNet 공유는 비활성화 (값은 False로 설정되도록 get_ui_settings에서 처리)
		return gr.update(value=new_lr), gr.update(interactive=unet_share_interactive)

	lora_target_dropdown.change(
		fn=update_ui_on_lora_target_change,
		inputs=[lora_target_dropdown],
		outputs=[learning_rate_textbox, use_shared_unet_checkbox],
		queue=False
	)

	# 설정 불러오기
	load_settings_button.click(
		fn=load_settings_gradio_for_list_outputs, # 이 함수는 apply_settings...를 호출하여 UI 업데이트
		inputs=None,
		outputs=load_outputs_list,
		queue=False
	).then(
		fn=lambda: gr.update(value=None), # 다운로드 링크 초기화
		inputs=None, outputs=download_config_file, queue=False
	)

	# 설정 저장
	save_settings_button.click(
		fn=save_settings_direct,
		inputs=non_path_input_components,
		outputs=[download_config_file],
		queue=False
	)

	# 미세조정 시작
	run_training_button.click(
		fn=lambda: (gr.update(value="설정 수집 중...", interactive=False), gr.update(interactive=False)),
		inputs=None, outputs=[status_textbox, run_training_button], queue=False
	).then(
		fn=get_ui_settings, inputs=non_path_input_components, outputs=settings_state, queue=True
	).then(
		fn=lambda: gr.update(value="학습 스크립트 실행 중... (터미널/로그 확인)"),
		inputs=None, outputs=status_textbox, queue=False
	).then(
		fn=run_training_gradio, inputs=settings_state, outputs=[status_textbox, run_training_button], queue=True
	)


# --- Gradio 앱 실행 ---
if __name__ == "__main__":
	print("Starting Gradio server...")
	# ... (기존 시작 로그 메시지) ...
	print(f"Script Dir: {SCRIPT_DIR}")
	# print(f"Training script: {TRAINING_SCRIPT_PATH}") # 이제 동적으로 결정됨
	print(f"Expecting Models in: {CONTAINER_MODELS_DIR} (Found: {fixed_pretrained_checkpoint_path})")
	print(f"Expecting LoRA in: {CONTAINER_LORA_DIR} (Found: {fixed_pretrained_lora_path})")
	print(f"Expecting Dataset in: {CONTAINER_DATASETS_DIR} (Found: {fixed_preference_dataset_path})")
	print(f"Using Config file: {fixed_config_path}")
	print(f"Saving output to: {CONTAINER_OUTPUT_DIR}")
	print(f"Ensure Docker volumes are correctly mapped...")
	print("Server will run at http://127.0.0.1:7866")
	sys.stdout.flush()

	demo.launch(server_name="0.0.0.0", server_port=7866, share=False, inbrowser=True, show_api=False)
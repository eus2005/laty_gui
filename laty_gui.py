import os
import sys
import subprocess
import platform
import json
import threading
import shutil
import time
import re

# --- 기본 설정) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # /app
TRAINING_SCRIPT_PATH = os.path.join('Main', 'train_dpo_lora.py')

# Gradio 임포트
try:
	import gradio as gr
except ImportError as e:
	print(f"오류: Gradio 라이브러리를 찾을 수 없습니다 - {e}")
	print("pip install gradio 를 실행하여 설치하거나 Dockerfile에 추가하세요.")
	sys.exit(1)

# --- 기본값 설정 (Flask 코드에서 가져옴 + Kohya 참고 추가) ---
default_config_path = "dpo_lora_finetune_settings.json"
default_pretrained_checkpoint_path = os.path.join(SCRIPT_DIR, "models", "your_model.safetensors")
default_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
default_pretrained_lora_path = os.path.join(SCRIPT_DIR, "Lora", "your_existing_lora.safetensors")
default_preference_dataset_path = os.path.join(SCRIPT_DIR, "datasets", "dpo_preferences.jsonl")
default_model_save_path = os.path.join(SCRIPT_DIR, "Lora", "finetuned_lora")
default_resolution_width = 1024
default_resolution_height = 1024
# <<< ADDED Kohya-inspired defaults >>>
default_lr_scheduler = "cosine_with_restarts" # Kohya 기본값 중 하나
default_lr_warmup_steps = 100 # Kohya 예시 값
default_optimizer_type = "AdamW8bit" # Kohya에서 자주 사용됨
default_max_token_length = 75 # CLIP 기본값
default_noise_offset = 0.0 # Kohya 기본값
default_use_xformers = True # 가능하면 사용하는 것이 좋음

# ANSI 이스케이프 코드 제거 함수 (Gradio 출력에 불필요한 색상 코드 제거 - 이제 터미널 출력용)
def strip_ansi_codes(s):
	if isinstance(s, str):
		ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
		return ansi_escape.sub('', s)
	return s

# --- Gradio 앱 로직 함수 ---

def get_ui_settings(
	config_path, pretrained_checkpoint_path, base_model_id, pretrained_lora_path,
	preference_dataset_path, model_save_path,
	num_epochs, batch_size, learning_rate, optimizer_type,
	dpo_beta, seed, gradient_accumulation_steps,
	resolution_width, resolution_height, checkpointing_steps, max_grad_norm,
	use_fp16, gradient_checkpointing,
    # <<< ADDED Kohya-inspired settings >>>
    lr_scheduler, lr_warmup_steps, lr_scheduler_num_cycles, lr_scheduler_power,
    use_xformers, max_token_length, noise_offset
):
	"""Gradio UI에서 현재 설정 값을 수집하여 딕셔너리로 반환합니다."""
	settings = {
		'configPath': config_path, # Keep configPath in the dict for saving/loading logic
		'pretrainedCheckpointPath': pretrained_checkpoint_path,
		'baseModelId': base_model_id,
		'pretrainedLoraPath': pretrained_lora_path,
		'preferenceDatasetPath': preference_dataset_path,
		'modelSavePath': model_save_path,
		'numEpochs': int(num_epochs), # Ensure integer type
		'batchSize': int(batch_size), # Ensure integer type
		'learningRate': learning_rate, # Keep as string for flexible input
		'optimizerType': optimizer_type,
		'dpoBeta': float(dpo_beta), # Ensure float type
		'seed': int(seed), # Ensure integer type
		'gradientAccumulationSteps': int(gradient_accumulation_steps), # Ensure integer type
		'resolutionWidth': int(resolution_width), # Ensure integer type
		'resolutionHeight': int(resolution_height), # Ensure integer type
		'checkpointingSteps': int(checkpointing_steps), # Ensure integer type
		'maxGradNorm': float(max_grad_norm), # Ensure float type
		'useFp16': use_fp16,
		'gradientCheckpointing': gradient_checkpointing,
		'use_8bit_adam': (optimizer_type == 'AdamW8bit'), # Derived flag
        # <<< ADDED Kohya-inspired settings >>>
        'lrScheduler': lr_scheduler,
        'lrWarmupSteps': int(lr_warmup_steps),
        'lrSchedulerNumCycles': int(lr_scheduler_num_cycles), # Assuming integer cycles
        'lrSchedulerPower': float(lr_scheduler_power), # Assuming float power
        'useXformers': use_xformers,
        'maxTokenLength': int(max_token_length),
        'noiseOffset': float(noise_offset),
	}
	return settings

# Helper to apply settings to UI components for loading
def apply_settings_to_ui_updates_dict(settings):
	"""설정 딕셔너리에서 값을 읽어 Gradio UI 컴포넌트 업데이트 객체를 반환합니다."""
	updates = {}
	# Use gr.update to set component values. Keys here match the keys in the settings dict.

	updates['pretrainedCheckpointPath'] = gr.update(value=settings.get('pretrainedCheckpointPath', default_pretrained_checkpoint_path))
	updates['baseModelId'] = gr.update(value=settings.get('baseModelId', default_base_model_id))
	updates['pretrainedLoraPath'] = gr.update(value=settings.get('pretrainedLoraPath', default_pretrained_lora_path))
	updates['preferenceDatasetPath'] = gr.update(value=settings.get('preferenceDatasetPath', default_preference_dataset_path))
	updates['modelSavePath'] = gr.update(value=settings.get('modelSavePath', default_model_save_path))

	# Get values, using .get() for safety with older config files
	num_epochs_val = settings.get('numEpochs', 3)
	batch_size_val = settings.get('batchSize', 1)
	learning_rate_val = settings.get('learningRate', '5e-7') # Keep learning rate as string
	optimizer_type_val = settings.get('optimizerType', default_optimizer_type) # Use default
	dpo_beta_val = settings.get('dpoBeta', 0.1)
	seed_val = settings.get('seed', 1)
	gradient_accumulation_steps_val = settings.get('gradientAccumulationSteps', 4)
	resolution_width_val = settings.get('resolutionWidth', default_resolution_width)
	resolution_height_val = settings.get('resolutionHeight', default_resolution_height)
	checkpointing_steps_val = settings.get('checkpointingSteps', 200)
	max_grad_norm_val = settings.get('maxGradNorm', 1.0)
	use_fp16_val = settings.get('useFp16', True)
	gradient_checkpointing_val = settings.get('gradientCheckpointing', False)

    # <<< ADDED Kohya-inspired settings with defaults >>>
	lr_scheduler_val = settings.get('lrScheduler', default_lr_scheduler)
	lr_warmup_steps_val = settings.get('lrWarmupSteps', default_lr_warmup_steps)
	lr_scheduler_num_cycles_val = settings.get('lrSchedulerNumCycles', 1) # Default for cosine_with_restarts
	lr_scheduler_power_val = settings.get('lrSchedulerPower', 1.0) # Default for polynomial
	use_xformers_val = settings.get('useXformers', default_use_xformers)
	max_token_length_val = settings.get('maxTokenLength', default_max_token_length)
	noise_offset_val = settings.get('noiseOffset', default_noise_offset)


	# Handle optimizerType and use_8bit_adam dependency for UI display
	if optimizer_type_val is None: # Should not happen with default set, but for safety
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

    # <<< ADDED updates for Kohya-inspired settings >>>
	updates['lrScheduler'] = gr.update(value=lr_scheduler_val)
	updates['lrWarmupSteps'] = gr.update(value=lr_warmup_steps_val)
	updates['lrSchedulerNumCycles'] = gr.update(value=lr_scheduler_num_cycles_val)
	updates['lrSchedulerPower'] = gr.update(value=lr_scheduler_power_val)
	updates['useXformers'] = gr.update(value=use_xformers_val)
	updates['maxTokenLength'] = gr.update(value=max_token_length_val)
	updates['noiseOffset'] = gr.update(value=noise_offset_val)

	return updates


# --- 설정 저장 함수 (터미널 출력용) ---
def save_settings_direct(
    config_path, pretrained_checkpoint_path, base_model_id, pretrained_lora_path,
    preference_dataset_path, model_save_path,
    num_epochs, batch_size, learning_rate, optimizer_type,
    dpo_beta, seed, gradient_accumulation_steps,
    resolution_width, resolution_height, checkpointing_steps, max_grad_norm,
    use_fp16, gradient_checkpointing,
    # <<< ADDED Kohya-inspired settings >>>
    lr_scheduler, lr_warmup_steps, lr_scheduler_num_cycles, lr_scheduler_power,
    use_xformers, max_token_length, noise_offset
):
	"""Gradio에서 받은 설정을 JSON 파일로 저장하고 터미널에 상태 메시지를 출력합니다."""
	if not config_path:
		print("오류: 저장할 설정 파일 경로를 입력해주세요.")
		return # Return None, no output component

	file_path = os.path.abspath(config_path)
	try:
		os.makedirs(os.path.dirname(file_path), exist_ok=True)

		# Get the settings dict from UI inputs
		settings_dict = get_ui_settings(
            config_path, pretrained_checkpoint_path, base_model_id, pretrained_lora_path,
            preference_dataset_path, model_save_path,
            num_epochs, batch_size, learning_rate, optimizer_type,
            dpo_beta, seed, gradient_accumulation_steps,
            resolution_width, resolution_height, checkpointing_steps, max_grad_norm,
            use_fp16, gradient_checkpointing,
            # <<< ADDED Kohya-inspired settings >>>
            lr_scheduler, lr_warmup_steps, lr_scheduler_num_cycles, lr_scheduler_power,
            use_xformers, max_token_length, noise_offset
        )

		# Remove configPath itself before saving the core settings
		if 'configPath' in settings_dict:
			del settings_dict['configPath'] # configPath is the path *to* the file, not a setting *in* the file
		# Remove derived flag
		if 'use_8bit_adam' in settings_dict:
			del settings_dict['use_8bit_adam']

		with open(file_path, 'w', encoding='utf-8') as f:
			json.dump(settings_dict, f, indent=2, ensure_ascii=False)

		print(f"설정이 '{config_path}'에 성공적으로 저장되었습니다.")
		return # Return None, no output component

	except Exception as e:
		print(f"설정 저장 중 오류 발생: {e}")
		print(f"Error details: {e}")
		return # Return None, no output component


# --- 설정 불러오기 함수 (list/tuple 반환 버전, 터미널 출력용) ---
def load_settings_gradio_for_list_outputs(config_path):
	"""설정 파일을 불러와 Gradio UI 컴포넌트 업데이트 객체의 리스트/튜플을 반환하고 터미널에 상태를 출력합니다."""
	# Define the order of keys for the output list/tuple (excluding status)
	# This order MUST match the order of components in load_outputs_list below
	output_key_order = [
		# configPath is the input, not an output to update with loaded value
		'pretrainedCheckpointPath', # Key for pretrained_checkpoint_path_textbox
		'baseModelId', # Key for base_model_id_textbox
		'pretrainedLoraPath', # Key for pretrained_lora_path_textbox
		'preferenceDatasetPath', # Key for preference_dataset_path_textbox
		'modelSavePath', # Key for model_save_path_textbox
		'numEpochs', # Key for num_epochs_number
		'batchSize', # Key for batch_size_number
		'learningRate', # Key for learning_rate_textbox
		'optimizerType', # Key for optimizer_type_dropdown
		'dpoBeta', # Key for dpo_beta_number
		'seed', # Key for seed_number
		'gradientAccumulationSteps', # Key for gradient_accumulation_steps_number
		'resolutionWidth', # Key for resolution_width_number
		'resolutionHeight', # Key for resolution_height_number
		'checkpointingSteps', # Key for checkpointing_steps_number
		'maxGradNorm', # Key for max_grad_norm_number
		'useFp16', # Key for use_fp16_checkbox
		'gradientCheckpointing', # Key for gradient_checkpointing_checkbox
        # <<< ADDED Kohya-inspired settings keys >>>
        'lrScheduler',           # Key for lr_scheduler_dropdown
        'lrWarmupSteps',         # Key for lr_warmup_steps_number
        'lrSchedulerNumCycles',  # Key for lr_scheduler_num_cycles_number
        'lrSchedulerPower',      # Key for lr_scheduler_power_number
        'useXformers',           # Key for use_xformers_checkbox
        'maxTokenLength',        # Key for max_token_length_number
        'noiseOffset',           # Key for noise_offset_number
	]

	# Initialize a dictionary to hold updates, starting with defaults for all settings inputs
	updates_dict = apply_settings_to_ui_updates_dict({})

	message = "" # To hold the final status message

	if not config_path:
		message = "오류: 불러올 설정 파일 경로를 입력해주세요."
		print(message) # Print to terminal
	else:
		file_path = os.path.abspath(config_path)
		try:
			if not os.path.exists(file_path):
				message = f"오류: 설정 파일 '{config_path}'을(를) 찾을 수 없습니다."
				print(message) # Print to terminal
			else:
				with open(file_path, 'r', encoding='utf-8') as f:
					settings = json.load(f)

				# Get updates based on loaded settings, overwriting defaults
				loaded_settings_updates = apply_settings_to_ui_updates_dict(settings)
				updates_dict.update(loaded_settings_updates) # Overwrite defaults with loaded values

				message = f"설정을 '{config_path}'에서 성공적으로 불러왔습니다."
				print(message) # Print to terminal

		except json.JSONDecodeError:
			message = f"오류: 설정 파일 '{config_path}'의 형식이 잘못되었습니다 (JSON 오류)."
			print(message) # Print to terminal
		except Exception as e:
			print(f"Error loading settings: {e}")
			message = f'설정 불러오기 중 오류 발생: {e}'
			print(message) # Print to terminal

	# Construct the final output list/tuple in the defined order
	output_list = []
	for key in output_key_order:
		output_list.append(updates_dict.get(key, gr.update())) # Add a default empty update if key not found

	return tuple(output_list) # Return the tuple of gr.update objects


# --- 학습 실행 함수 (터미널 출력용) ---
def run_training_gradio(settings_dict):
	"""Gradio에서 받은 설정으로 accelerate launch 명령어를 실행하고 출력을 터미널에 스트리밍합니다. (DPO용 수정)"""
	print("\n" + "="*20 + " LoRA DPO Fine-tuning 시작 " + "="*20)

	try:
		accelerate_cmd = shutil.which("accelerate")
		if not accelerate_cmd:
			print("오류: 'accelerate' 명령어를 찾을 수 없습니다. 가상 환경 또는 시스템 PATH를 확인하세요.")
			return

		training_script_abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, TRAINING_SCRIPT_PATH))
		training_script_normalized_path = os.path.normpath(training_script_abs_path)

		if not os.path.exists(training_script_normalized_path):
			print(f"오류: 학습 스크립트 경로를 찾을 수 없습니다: {training_script_normalized_path}")
			return

		# Mapping from UI ID/Key (from settings_dict) to Script Arg Name
		gui_to_script_args = {
			'pretrainedCheckpointPath': 'pretrained_checkpoint_path',
			'baseModelId': 'base_model_id',
			'pretrainedLoraPath': 'pretrained_lora_path',
			'preferenceDatasetPath': 'dataset_path',
			'modelSavePath': 'output_dir',
			'numEpochs': 'num_train_epochs',
			'batchSize': 'train_batch_size',
			'learningRate': 'learning_rate',
			'dpoBeta': 'dpo_beta',
			'seed': 'seed',
			'gradientAccumulationSteps': 'gradient_accumulation_steps',
			'resolutionWidth': 'resolution_width',
			'resolutionHeight': 'resolution_height',
			'maxGradNorm': 'max_grad_norm',
			'checkpointingSteps': 'checkpointing_steps',
            # <<< ADDED Kohya-inspired mappings >>>
            'lrScheduler': 'lr_scheduler',
            'lrWarmupSteps': 'lr_warmup_steps',
            'lrSchedulerNumCycles': 'lr_scheduler_num_cycles',
            'lrSchedulerPower': 'lr_scheduler_power',
            'maxTokenLength': 'max_token_length',
            'noiseOffset': 'noise_offset',
			# Note: optimizer settings like adam betas are passed to the script if needed
            # Currently using defaults in train script, but could be exposed
		}

		# Build accelerate launch command
		cmd = [accelerate_cmd, 'launch', training_script_normalized_path]

		# Add arguments from settings_dict
		for gui_key, script_arg_name in gui_to_script_args.items():
			if gui_key in settings_dict and settings_dict[gui_key] is not None and str(settings_dict[gui_key]).strip() != "":
				value = settings_dict[gui_key]
				val_str = str(value)

				# Resolve relative paths
				abs_path_keys = ['pretrainedCheckpointPath', 'pretrainedLoraPath', 'preferenceDatasetPath', 'modelSavePath']
				if gui_key in abs_path_keys and not os.path.isabs(val_str):
					abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, val_str))
					val_str = abs_path
					# print(f"Resolved relative path for '{gui_key}': {val_str}") # Optional: Log resolved path

				cmd.extend([f'--{script_arg_name}', val_str])

		# Handle boolean/flag arguments
		if settings_dict.get('useFp16') is True:
			cmd.extend(['--mixed_precision', 'fp16']) # Assuming fp16 if checked
		else:
			# Determine mixed_precision based on fp16 checkbox (bf16 not directly supported in UI)
			# If not fp16, could default to 'no' or keep it unset for script default
			cmd.extend(['--mixed_precision', 'no'])

		if settings_dict.get('gradientCheckpointing') is True:
			cmd.append('--gradient_checkpointing')

		if settings_dict.get('optimizerType') == 'AdamW8bit':
			cmd.append('--use_8bit_adam')

        # <<< ADDED Kohya-inspired flags >>>
		if settings_dict.get('useXformers') is True:
			cmd.append('--use_xformers')

		# Log the final command
		print(f"Executing command: {' '.join(cmd)}")
		print("-" * (40 + len(" LoRA DPO Fine-tuning 시작 ")))

		# Execute the command and stream output to terminal
		process = subprocess.Popen(
			cmd,
			cwd=SCRIPT_DIR, # Execute in the script directory
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			bufsize=1,
			universal_newlines=True
		)

		# Stream output
		while True:
			stdout_line = process.stdout.readline()
			stderr_line = process.stderr.readline()
			if not stdout_line and not stderr_line and process.poll() is not None:
				break
			if stdout_line:
				print(strip_ansi_codes(stdout_line.strip()))
			if stderr_line:
				print(f"[STDERR] {strip_ansi_codes(stderr_line.strip())}")
			time.sleep(0.01)

		# Process remaining output
		remaining_stdout = process.stdout.read()
		if remaining_stdout:
			print(strip_ansi_codes(remaining_stdout.strip()))
		remaining_stderr = process.stderr.read()
		if remaining_stderr:
			print(f"[STDERR] {strip_ansi_codes(remaining_stderr.strip())}")

		returncode = process.wait()

		if returncode == 0:
			print("="*20 + " DPO Fine-tuning 완료 " + "="*20 + "\n")
		else:
			print(f"오류: DPO 미세조정 스크립트 실행 중 오류 발생 (Exit code: {returncode})")

	except FileNotFoundError as e:
		print(f"오류: 명령어 또는 스크립트 파일 실행 오류 - {e}. PATH 설정을 확인하거나 파일이 존재하는지 확인하세요.")
	except Exception as e:
		print(f"오류: DPO 미세조정 스크립트 실행 중 문제 발생: {e}")
		import traceback
		traceback.print_exc()


# --- Gradio UI 정의 ---

with gr.Blocks(title="DPO LoRA 미세조정 GUI") as demo:
	gr.Markdown("# DPO 기반 LoRA 미세조정 GUI")

	settings_state = gr.State()

	with gr.Tab("설정"):
		gr.Markdown("### 설정 관리")
		config_path_textbox = gr.Textbox(
			label="설정 파일 경로 (.json):",
			value=default_config_path,
			placeholder="예: finetune_settings.json"
		)
		gr.Markdown("<small>미세조정 설정을 저장하거나 불러올 JSON 파일의 경로입니다.</small>")

		with gr.Row():
			load_settings_button = gr.Button("설정 불러오기")
			save_settings_button = gr.Button("설정 저장")

		gr.Markdown("### 경로 설정") # Section Renamed

		pretrained_checkpoint_path_textbox = gr.Textbox(
			label="베이스 모델 체크포인트 경로:",
			value=default_pretrained_checkpoint_path,
			placeholder="예: /path/to/your_model.safetensors"
		)
		gr.Markdown("<small>사용할 베이스 모델의 <b>단일 체크포인트 파일</b> 경로 (<code>.safetensors</code> 또는 <code>.ckpt</code>).</small>")

		base_model_id_textbox = gr.Textbox(
			label="호환 베이스 ID/경로 (Config/Tokenizer용):",
			value=default_base_model_id,
			placeholder="예: stabilityai/stable-diffusion-xl-base-1.0"
		)
		gr.Markdown("<small>위 체크포인트 파일과 호환되는 <b>Hugging Face 모델 ID</b> 또는 <b>로컬 모델 디렉토리 경로</b>.<br>(모델 구조 및 토크나이저 로드에 사용. 예: SDXL 체크포인트에는 <code>stabilityai/stable-diffusion-xl-base-1.0</code>)</small>")

		pretrained_lora_path_textbox = gr.Textbox(
			label="기존 LoRA 경로:",
			value=default_pretrained_lora_path,
			placeholder="예: /path/to/your_existing_lora.safetensors"
		)
		gr.Markdown("<small>미세조정할 기존 LoRA 파일 경로.</small>")

		preference_dataset_path_textbox = gr.Textbox(
			label="선호도 데이터셋 경로 (.jsonl):",
			value=default_preference_dataset_path,
			placeholder="예: /path/to/dpo_preferences.jsonl"
		)
		gr.Markdown("<small>미리 생성된 (프롬프트, 선호/비선호 비교 점수) 데이터셋 파일 경로 (<code>.jsonl</code>).</small>")

		model_save_path_textbox = gr.Textbox(
			label="미세조정된 LoRA 저장 폴더:",
			value=default_model_save_path,
			placeholder="예: /path/to/save/finetuned_lora"
		)
		gr.Markdown("<small>미세조정된 LoRA 모델이 저장될 **폴더** 경로.</small>")


		gr.Markdown("### 기본 학습 파라미터")

		with gr.Row():
			num_epochs_number = gr.Number(label="에포크 수:", value=3, precision=0, minimum=1)
			batch_size_number = gr.Number(label="배치 크기 (장치당):", value=1, precision=0, minimum=1)
			gradient_accumulation_steps_number = gr.Number(label="그래디언트 축적 스텝:", value=4, precision=0, minimum=1)

		with gr.Row():
			dpo_beta_number = gr.Number(label="DPO Beta:", value=0.1, minimum=0.0)
			seed_number = gr.Number(label="시드:", value=1, precision=0)
			max_grad_norm_number = gr.Number(label="최대 그래디언트 Norm:", value=1.0, minimum=0.0)

		with gr.Row():
			resolution_width_number = gr.Number(label="해상도 너비:", value=default_resolution_width, precision=0, minimum=64, step=8)
			resolution_height_number = gr.Number(label="해상도 높이:", value=default_resolution_height, precision=0, minimum=64, step=8)
			checkpointing_steps_number = gr.Number(label="체크포인트 저장 주기(스텝):", value=200, precision=0, minimum=0, info="0이면 저장 안함")

		gr.Markdown("### 옵티마이저 및 학습률")

		with gr.Row():
			learning_rate_textbox = gr.Textbox(label="기본 학습률:", value="5e-7", placeholder="예: 1e-6")
			optimizer_type_dropdown = gr.Dropdown(
				label="옵티마이저:",
				# Kohya에서 지원하는 다른 옵티마이저 추가 가능 (예: Prodigy) - 단, train 스크립트 수정 필요
				choices=["AdamW", "AdamW8bit"], # "Prodigy", "SGD" 등 추가 가능
				value=default_optimizer_type
			)

		with gr.Row():
			lr_scheduler_dropdown = gr.Dropdown(
				label="LR 스케줄러:",
				choices=["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"],
				value=default_lr_scheduler
			)
			lr_warmup_steps_number = gr.Number(label="LR 웜업 스텝:", value=default_lr_warmup_steps, precision=0, minimum=0)

		with gr.Row():
			lr_scheduler_num_cycles_number = gr.Number(label="LR 주기 (cosine_restarts):", value=1, precision=0, minimum=1)
			lr_scheduler_power_number = gr.Number(label="LR Power (polynomial):", value=1.0, minimum=0.0)


		gr.Markdown("### 고급 설정")

		with gr.Row():
            # <<< RENAMED for clarity (mixed precision) >>>
			use_fp16_checkbox = gr.Checkbox(label="FP16/BF16 사용 (mixed_precision)", value=True)
			gradient_checkpointing_checkbox = gr.Checkbox(label="그래디언트 체크포인팅", value=False)
            # <<< ADDED use_xformers checkbox >>>
			use_xformers_checkbox = gr.Checkbox(label="XFormers 사용", value=default_use_xformers, info="설치된 경우 메모리 절약")

		with gr.Row():
			max_token_length_number = gr.Number(label="최대 토큰 길이:", value=default_max_token_length, precision=0, minimum=75) # SDXL은 보통 77 이상
			noise_offset_number = gr.Number(label="노이즈 오프셋:", value=default_noise_offset, minimum=0.0, info="0 초과 시 적용")


	# --- 미세조정 시작 탭 ---
	with gr.Tab("미세조정 실행"):
		gr.Markdown("### 미세조정 시작")
		gr.Markdown("모든 설정이 올바른지 확인 후 '미세조정 시작' 버튼을 누르세요.")
		gr.Markdown("<p style='color: red;'><b>학습 진행 로그는 이 GUI를 실행한 터미널에 출력됩니다.</b></p>") # 강조
		run_training_button = gr.Button("미세조정 시작", variant="primary")

		gr.Markdown("---")
		# 서버 종료 버튼은 유지하지 않음 (Ctrl+C로 종료)


	# --- 이벤트 핸들러 연결 ---

	# UI 컴포넌트 목록 (get_ui_settings 함수의 입력 순서와 일치하도록)
	all_input_components = [
		config_path_textbox,
		pretrained_checkpoint_path_textbox, base_model_id_textbox, pretrained_lora_path_textbox,
		preference_dataset_path_textbox, model_save_path_textbox,
		num_epochs_number, batch_size_number, learning_rate_textbox, optimizer_type_dropdown,
		dpo_beta_number, seed_number, gradient_accumulation_steps_number,
		resolution_width_number, resolution_height_number, checkpointing_steps_number, max_grad_norm_number,
		use_fp16_checkbox, gradient_checkpointing_checkbox,
        # <<< ADDED Kohya-inspired components >>>
        lr_scheduler_dropdown, lr_warmup_steps_number, lr_scheduler_num_cycles_number, lr_scheduler_power_number,
        use_xformers_checkbox, max_token_length_number, noise_offset_number,
	]

	# Define the list of output components for loading settings
	# This list MUST match the order of keys in output_key_order
	load_outputs_list = [
        pretrained_checkpoint_path_textbox,
		base_model_id_textbox,
		pretrained_lora_path_textbox,
		preference_dataset_path_textbox,
		model_save_path_textbox,
		num_epochs_number,
		batch_size_number,
		learning_rate_textbox,
		optimizer_type_dropdown,
		dpo_beta_number,
		seed_number,
		gradient_accumulation_steps_number,
		resolution_width_number,
		resolution_height_number,
		checkpointing_steps_number,
		max_grad_norm_number,
		use_fp16_checkbox,
		gradient_checkpointing_checkbox,
        # <<< ADDED Kohya-inspired components >>>
        lr_scheduler_dropdown,
        lr_warmup_steps_number,
        lr_scheduler_num_cycles_number,
        lr_scheduler_power_number,
        use_xformers_checkbox,
        max_token_length_number,
        noise_offset_number,
    ]


	# 설정 불러오기 버튼 클릭 시
	load_settings_button.click(
		fn=load_settings_gradio_for_list_outputs,
		inputs=config_path_textbox,
		outputs=load_outputs_list,
		queue=False
	)


	# 설정 저장 버튼 클릭 시
	save_settings_button.click(
		fn=save_settings_direct,
		inputs=all_input_components,
		outputs=None,
		queue=False
	)


	# 미세조정 시작 버튼 클릭 시
	run_training_button.click(
		fn=get_ui_settings, # Step 1: Get settings
		inputs=all_input_components,
		outputs=settings_state, # Output to state
		queue=True # Fast
	).then(
		fn=run_training_gradio, # Step 2: Run training using state
		inputs=settings_state,
		outputs=None, # Logs to terminal
		queue=True # Long running
	)


# --- Gradio 앱 실행 ---
if __name__ == "__main__":
	print("Starting Gradio server for LoRA DPO Fine-tuning GUI...")
	print(f"Script Dir: {SCRIPT_DIR}")
	print(f"Training script: {TRAINING_SCRIPT_PATH}")
	print("Server will run at http://0.0.0.0:7866 (Access via http://localhost:7866 or host IP)")
	print("Gradio 서버를 종료하려면 터미널에서 Ctrl+C를 누르세요.")

	demo.launch(
		server_name="0.0.0.0",
		server_port=7866,
		share=False,
		inbrowser=True,
		show_api=False,
		# debug=True # 필요시 활성화
	)
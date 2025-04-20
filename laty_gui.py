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

# --- 기본값 설정 (Flask 코드에서 가져옴) ---
default_config_path = "dpo_lora_finetune_settings.json"
default_pretrained_checkpoint_path = os.path.join(SCRIPT_DIR, "models", "your_model.safetensors")
default_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
default_pretrained_lora_path = os.path.join(SCRIPT_DIR, "Lora", "your_existing_lora.safetensors")
# <<< CHANGED key and filename
default_preference_dataset_path = os.path.join(SCRIPT_DIR, "datasets", "dpo_preferences.jsonl")
# <<< CHANGED output folder name
default_model_save_path = os.path.join(SCRIPT_DIR, "Lora", "finetuned_lora")
# <<< ADDED default resolution width and height
default_resolution_width = 1024
default_resolution_height = 1024

# ANSI 이스케이프 코드 제거 함수 (Gradio 출력에 불필요한 색상 코드 제거)
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
	use_fp16, gradient_checkpointing
):
	"""Gradio UI에서 현재 설정 값을 수집하여 딕셔너리로 반환합니다."""
	settings = {
		'configPath': config_path,
		'pretrainedCheckpointPath': pretrained_checkpoint_path,
		'baseModelId': base_model_id,
		'pretrainedLoraPath': pretrained_lora_path,
		'preferenceDatasetPath': preference_dataset_path,
		'modelSavePath': model_save_path,
		'numEpochs': num_epochs,
		'batchSize': batch_size,
		'learningRate': learning_rate,
		'optimizerType': optimizer_type, # Keep optimizerType for saving/loading
		'dpoBeta': dpo_beta,
		'seed': seed,
		'gradientAccumulationSteps': gradient_accumulation_steps,
		'resolutionWidth': resolution_width,
		'resolutionHeight': resolution_height,
		'checkpointingSteps': checkpointing_steps,
		'maxGradNorm': max_grad_norm,
		'useFp16': use_fp16,
		'gradientCheckpointing': gradient_checkpointing,
		# Add the specific flag expected by the script based on optimizerType
		'use_8bit_adam': (optimizer_type == 'AdamW8bit')
	}
	return settings

def apply_settings_to_ui_updates_dict(settings):
	"""설정 딕셔너리에서 값을 읽어 Gradio UI 컴포넌트 업데이트 객체를 반환합니다.
	반환 딕셔너리의 키는 설정 딕셔너리의 키와 동일합니다.
	"""
	updates = {}
	# Use gr.update to set component values. Keys here match the keys in the settings dict.

	updates['configPath'] = gr.update(value=settings.get('configPath', default_config_path))
	updates['pretrainedCheckpointPath'] = gr.update(value=settings.get('pretrainedCheckpointPath', default_pretrained_checkpoint_path))
	updates['baseModelId'] = gr.update(value=settings.get('baseModelId', default_base_model_id))
	updates['pretrainedLoraPath'] = gr.update(value=settings.get('pretrainedLoraPath', default_pretrained_lora_path))

	updates['preferenceDatasetPath'] = gr.update(value=settings.get('preferenceDatasetPath', default_preference_dataset_path))

	updates['modelSavePath'] = gr.update(value=settings.get('modelSavePath', default_model_save_path))

	updates['numEpochs'] = gr.update(value=settings.get('numEpochs', 3))
	updates['batchSize'] = gr.update(value=settings.get('batchSize', 1))
	updates['learningRate'] = gr.update(value=settings.get('learningRate', 0.0000001))

	# Handle optimizerType and use_8bit_adam dependency for UI display
	optimizer_type_value = settings.get('optimizerType')
	# If optimizerType was not saved (older config), derive from use_8bit_adam
	if optimizer_type_value is None:
		optimizer_type_value = 'AdamW8bit' if settings.get('use_8bit_adam', False) else 'AdamW'
	updates['optimizerType'] = gr.update(value=optimizer_type_value) # Key matches settings dict


	updates['dpoBeta'] = gr.update(value=settings.get('dpoBeta', 0.1))
	updates['seed'] = gr.update(value=settings.get('seed', 1))
	updates['gradientAccumulationSteps'] = gr.update(value=settings.get('gradientAccumulationSteps', 4))

	updates['resolutionWidth'] = gr.update(value=settings.get('resolutionWidth', default_resolution_width))
	updates['resolutionHeight'] = gr.update(value=settings.get('resolutionHeight', default_resolution_height))
	updates['checkpointingSteps'] = gr.update(value=settings.get('checkpointingSteps', 200))
	updates['maxGradNorm'] = gr.update(value=settings.get('maxGradNorm', 1.0))

	updates['useFp16'] = gr.update(value=settings.get('useFp16', True))
	updates['gradientCheckpointing'] = gr.update(value=settings.get('gradientCheckpointing', False))

	return updates

def save_settings_gradio(config_path, settings_dict):
	"""Gradio에서 받은 설정을 JSON 파일로 저장하고 상태 메시지를 반환합니다."""
	if not config_path:
		return "오류: 저장할 설정 파일 경로를 입력해주세요.", None # Return message and None for updates

	file_path = os.path.abspath(config_path)
	try:
		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		with open(file_path, 'w', encoding='utf-8') as f:
			# Remove configPath itself before saving the core settings
			core_settings = settings_dict.copy()
			if 'configPath' in core_settings:
				del core_settings['configPath']
			json.dump(core_settings, f, indent=2, ensure_ascii=False)
		return f"설정이 '{config_path}'에 성공적으로 저장되었습니다.", None # Return message and None for updates
	except Exception as e:
		print(f"Error saving settings: {e}")
		return f'설정 저장 중 오류 발생: {e}', None # Return message and None for updates

# --- 설정 불러오기 함수 (list/tuple 반환 버전) ---
# 이 함수는 load_settings_button의 outputs 리스트/튜플과 순서를 맞춰야 합니다.
def load_settings_gradio_for_list_outputs(config_path):
    # Define the order of keys for the output list/tuple
    # This order MUST match the order of components in load_outputs_list below
    output_key_order = [
        'status_output', # Key for the status textbox
        'configPath', # Key for config_path_textbox
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
    ]

    # Initialize a dictionary to hold all possible updates, with defaults or blank values
    # Use the dictionary-based update function
    updates_dict = apply_settings_to_ui_updates_dict({}) # Start with defaults for all settings inputs
    updates_dict['status_output'] = gr.update(value="") # Initialize status update

    message = "" # To hold the final status message

    if not config_path:
        message = "오류: 불러올 설정 파일 경로를 입력해주세요."
        updates_dict['status_output'] = gr.update(value=message)
    else:
        file_path = os.path.abspath(config_path)
        try:
            if not os.path.exists(file_path):
                message = f"오류: 설정 파일 '{config_path}'을(를) 찾을 수 없습니다."
                updates_dict['status_output'] = gr.update(value=message)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                # Pass configPath back to apply_settings_to_ui_updates_dict if it uses it (it does for default value logic)
                settings['configPath'] = config_path

                # Get updates based on loaded settings, overwriting defaults
                loaded_settings_updates = apply_settings_to_ui_updates_dict(settings)
                updates_dict.update(loaded_settings_updates) # Overwrite defaults with loaded values

                message = f"설정을 '{config_path}'에서 성공적으로 불러왔습니다."
                updates_dict['status_output'] = gr.update(value=message) # Update status message

        except json.JSONDecodeError:
            message = f"오류: 설정 파일 '{config_path}'의 형식이 잘못되었습니다 (JSON 오류)."
            updates_dict['status_output'] = gr.update(value=message)
            # On error, keep the default values loaded initially
        except Exception as e:
            print(f"Error loading settings: {e}")
            message = f'설정 불러오기 중 오류 발생: {e}'
            updates_dict['status_output'] = gr.update(value=message)
            # On error, keep the default values loaded initially


    # Construct the final output list/tuple in the defined order
    # The values are the gr.update objects from updates_dict
    output_list = []
    for key in output_key_order:
        # Get the gr.update object for this key from the updates_dict
        # Use .get() with a default in case a key was missed somewhere
        # This maps the internal update dictionary keys to the ordered list of outputs
        output_list.append(updates_dict.get(key, gr.update())) # Add a default empty update if key not found

    return tuple(output_list) # Return the tuple of gr.update objects


def run_training_gradio(settings_dict):
	"""Gradio에서 받은 설정으로 accelerate launch 명령어를 실행하고 출력을 스트리밍합니다. (DPO용 수정)"""
	output_textbox = "" # Accumulate output

	def update_output(message, is_error=False):
		nonlocal output_textbox
		clean_message = strip_ansi_codes(message)
		output_textbox += clean_message + "\n"
		# Yield the accumulated output
		yield output_textbox.strip()

	yield from update_output("\n" + "="*20 + " LoRA DPO Fine-tuning 시작 " + "="*20) # <<< UPDATED message

	try:
		accelerate_cmd = shutil.which("accelerate")
		if not accelerate_cmd:
			yield from update_output("오류: 'accelerate' 명령어를 찾을 수 없습니다. 가상 환경 또는 시스템 PATH를 확인하세요.", is_error=True)
			return

		training_script_abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, TRAINING_SCRIPT_PATH))
		training_script_normalized_path = os.path.normpath(training_script_abs_path)

		if not os.path.exists(training_script_normalized_path):
			yield from update_output(f"오류: 학습 스크립트 경로를 찾을 수 없습니다: {training_script_normalized_path}", is_error=True)
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
			# Note: adam parameters are not in the GUI currently, but could be added
			# 'adamBeta1': 'adam_beta1',
			# 'adamBeta2': 'adam_beta2',
			# 'adamWeightDecay': 'adam_weight_decay',
			# 'adamEpsilon': 'adam_epsilon',
			'maxGradNorm': 'max_grad_norm',
			# Note: dataloader_num_workers is not in the GUI currently, but could be added
			# 'dataloaderNumWorkers': 'dataloader_num_workers',
			'checkpointingSteps': 'checkpointing_steps',
		}

		# Build accelerate launch command
		cmd = [accelerate_cmd, 'launch', training_script_normalized_path]

		# Add arguments from settings_dict
		for gui_key, script_arg_name in gui_to_script_args.items():
			# Check if the key exists in settings_dict and the value is not None/empty string
			if gui_key in settings_dict and settings_dict[gui_key] is not None and str(settings_dict[gui_key]).strip() != "":
				value = settings_dict[gui_key]
				val_str = str(value)

				# Ensure paths are absolute using SCRIPT_DIR for relative paths
				abs_path_keys = ['pretrainedCheckpointPath', 'pretrainedLoraPath', 'preferenceDatasetPath', 'modelSavePath']
				if gui_key in abs_path_keys and not os.path.isabs(val_str):
					abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, val_str))
					val_str = abs_path
					yield from update_output(f"Resolved relative path for '{gui_key}': {val_str}") # Log resolved path

				# Add the argument to the command
				cmd.extend([f'--{script_arg_name}', val_str])

		# Handle boolean/flag arguments and special cases like mixed_precision
		if settings_dict.get('useFp16') is True:
			# Assuming script expects '--mixed_precision fp16'
			cmd.extend(['--mixed_precision', 'fp16'])
		elif settings_dict.get('useFp16') is False:
			# If FP16 is unchecked, use 'no' for mixed_precision (assuming fp32)
			cmd.extend(['--mixed_precision', 'no'])
		# Note: bf16 is not currently exposed in the GUI

		if settings_dict.get('gradientCheckpointing') is True:
			cmd.append('--gradient_checkpointing')

		# Handle optimizer type via the --use_8bit_adam flag if AdamW8bit was selected
		if settings_dict.get('optimizerType') == 'AdamW8bit':
			cmd.append('--use_8bit_adam') # Assuming the script supports this flag
		# Other optimizer types are handled by the script's default AdamW behavior
		# If script requires specific flags for AdamW, Prodigy, SGD, they need to be added here


		# Log the final command
		yield from update_output(f"Executing command: {' '.join(cmd)}")
		yield from update_output("-" * (40 + len(" LoRA DPO Fine-tuning 시작 ")))

		# Execute the command and stream output
		# Use Popen with PIPE to capture stdout/stderr
		process = subprocess.Popen(
			cmd,
			cwd=SCRIPT_DIR, # Execute in the script directory
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True, # Decode output as text
			bufsize=1, # Line buffering
			universal_newlines=True # Ensure consistent newline handling
		)

		# Stream output line by line
		# Combine stdout and stderr for simpler display in Gradio
		while True:
			stdout_line = process.stdout.readline()
			stderr_line = process.stderr.readline()

			if not stdout_line and not stderr_line and process.poll() is not None:
				break # Process finished and pipes are empty

			if stdout_line:
				yield from update_output(stdout_line.strip())
			if stderr_line:
				# Indicate stderr lines, possibly with a different color if CSS is applied to the Gradio app
				yield from update_output(f"[STDERR] {stderr_line.strip()}", is_error=True)

			# Avoid busy waiting, give the CPU a break
			time.sleep(0.01)

		# After the loop, there might still be some remaining output
		# Read anything left in the pipes
		remaining_stdout = process.stdout.read()
		if remaining_stdout:
			yield from update_output(remaining_stdout.strip())

		remaining_stderr = process.stderr.read()
		if remaining_stderr:
			yield from update_output(f"[STDERR] {remaining_stderr.strip()}", is_error=True)


		# Wait for the process to truly finish (should be immediate after pipes are empty)
		returncode = process.wait()

		if returncode == 0:
			yield from update_output("="*20 + " DPO Fine-tuning 완료 " + "="*20 + "\n") # <<< UPDATED message
		else:
			yield from update_output(f"오류: DPO 미세조정 스크립트 실행 중 오류 발생 (Exit code: {returncode})", is_error=True)

	except FileNotFoundError as e:
		# More specific error if accelerate or the script isn't found
		yield from update_output(f"오류: 명령어 또는 스크립트 파일 실행 오류 - {e}. PATH 설정을 확인하거나 파일이 존재하는지 확인하세요.", is_error=True)
	except Exception as e:
		yield from update_output(f"오류: DPO 미세조정 스크립트 실행 중 문제 발생: {e}", is_error=True) # <<< UPDATED message


# --- 서버 종료 함수 (제거) ---
# def shutdown_server_gradio():
# 	"""Gradio 서버 종료를 시도합니다."""
# 	print("Gradio 서버 종료 요청됨.")
# 	try:
# 		yield "서버 종료 요청을 받았습니다. 곧 종료됩니다."
# 		time.sleep(1) # Give the message time to appear
# 		os._exit(0) # More forceful exit than sys.exit()
# 	except Exception as e:
# 		yield f"오류: 서버 종료 중 문제 발생: {e}"
# 		print(f"Error during server shutdown: {e}")


# --- Gradio UI 정의 ---

with gr.Blocks(title="DPO LoRA 미세조정 GUI") as demo: # <<< UPDATED title
	gr.Markdown("# DPO 기반 LoRA 미세조정 GUI") # <<< UPDATED heading

	# 상태 메시지 출력 영역
	status_output = gr.Textbox(label="상태 / 로그", interactive=False, lines=10)

	# 설정 값을 임시로 저장할 State 변수 정의 (학습 실행 버튼용)
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

		gr.Markdown("### 미세조정 설정")

		pretrained_checkpoint_path_textbox = gr.Textbox(
			label="베이스 모델 체크포인트 경로:",
			value=default_pretrained_checkpoint_path,
			placeholder="예: /path/to/your_model.safetensors"
		)
		gr.Markdown("<small>사용할 베이스 모델의 <b>단일 체크포인트 파일</b> 경로 (<code>.safetensors</code> 또는 <code>.ckpt</code>)를 입력하세요.</small>")

		base_model_id_textbox = gr.Textbox(
			label="호환 베이스 ID/경로 (Config/Tokenizer용):",
			value=default_base_model_id,
			placeholder="예: stabilityai/stable-diffusion-xl-base-1.0"
		)
		gr.Markdown("<small>위 체크포인트 파일과 호환되는 <b>Hugging Face 모델 ID</b> 또는 <b>로컬 모델 디렉토리 경로</b>를 입력하세요.<br>(모델 구조 및 토크나이저 로드에 사용됩니다. 예: SDXL 체크포인트에는 <code>stabilityai/stable-diffusion-xl-base-1.0</code>)</small>")

		pretrained_lora_path_textbox = gr.Textbox(
			label="기존 LoRA 경로:",
			value=default_pretrained_lora_path,
			placeholder="예: /path/to/your_existing_lora.safetensors"
		)
		gr.Markdown("<small>미세조정할 기존 LoRA 파일 경로입니다.</small>")

		# <<< CHANGED ID and Label
		preference_dataset_path_textbox = gr.Textbox(
			label="선호도 데이터셋 경로:",
			value=default_preference_dataset_path,
			placeholder="예: /path/to/dpo_preferences.jsonl"
		)
		gr.Markdown("<small>미리 생성된 (프롬프트, 선호/비선호 응답 점수) 데이터셋 파일 경로입니다 (예: <code>.jsonl</code>).</small>")

		# <<< CHANGED output folder name
		model_save_path_textbox = gr.Textbox(
			label="미세조정된 LoRA 저장 경로:",
			value=default_model_save_path,
			placeholder="예: /path/to/save/finetuned_lora"
		)
		gr.Markdown("<small>미세조정된 LoRA 모델이 저장될 **폴더** 경로입니다.</small>")


		gr.Markdown("### 학습 파라미터")

		with gr.Row():
			num_epochs_number = gr.Number(label="에포크 수:", value=3, precision=0, minimum=1)
			batch_size_number = gr.Number(label="배치 크기:", value=1, precision=0, minimum=1)

		with gr.Row():
			learning_rate_textbox = gr.Textbox(label="학습률:", value="5e-7", placeholder="예: 0.0000001")
			optimizer_type_dropdown = gr.Dropdown(
				label="옵티마이저:",
				choices=["AdamW", "AdamW8bit", "Prodigy", "SGD"],
				value="AdamW8bit" # Assuming AdamW8bit is preferred and bitsandbytes is available
			)
			# Note: AdamW8bit selection implicitly sets use_8bit_adam flag later

		with gr.Row():
			# <<< CHANGED ID and Label from klBeta
			dpo_beta_number = gr.Number(label="DPO Beta:", value=0.1, minimum=0.0)
			seed_number = gr.Number(label="시드:", value=1, precision=0)
			gradient_accumulation_steps_number = gr.Number(label="그래디언트 축적 스텝:", value=4, precision=0, minimum=1)

		with gr.Row():
			# <<< ADDED resolution inputs
			resolution_width_number = gr.Number(label="해상도 너비:", value=default_resolution_width, precision=0, minimum=64, step=8)
			resolution_height_number = gr.Number(label="해상도 높이:", value=default_resolution_height, precision=0, minimum=64, step=8)

		with gr.Row():
			checkpointing_steps_number = gr.Number(label="체크포인트 저장 주기(스텝):", value=200, precision=0, minimum=1)
			max_grad_norm_number = gr.Number(label="최대 그래디언트 Norm:", value=1.0, minimum=0.0)

		with gr.Row():
			use_fp16_checkbox = gr.Checkbox(label="FP16/BF16", value=True)
			gradient_checkpointing_checkbox = gr.Checkbox(label="그래디언트 체크포인팅", value=False)

	# --- 미세조정 시작 탭 ---
	with gr.Tab("미세조정 실행"):
		gr.Markdown("### 미세조정 시작")
		gr.Markdown("모든 설정이 올바른지 확인 후 '미세조정 시작' 버튼을 누르세요.")
		run_training_button = gr.Button("미세조정 시작", variant="primary")

		gr.Markdown("---")
		# --- 서버 관리 섹션 제거 ---
		# gr.Markdown("### 서버 관리")
		# gr.Markdown("경고: 아래 버튼은 Gradio 서버를 종료합니다. 터미널에서 다시 실행해야 합니다.")
		# shutdown_button = gr.Button("Gradio 서버 정지", variant="secondary")


	# --- 이벤트 핸들러 연결 ---

	# UI 컴포넌트 목록 (get_ui_settings 함수의 입력 순서와 일치하도록)
	all_input_components = [
		config_path_textbox,
		pretrained_checkpoint_path_textbox, base_model_id_textbox, pretrained_lora_path_textbox,
		preference_dataset_path_textbox, model_save_path_textbox,
		num_epochs_number, batch_size_number, learning_rate_textbox, optimizer_type_dropdown,
		dpo_beta_number, seed_number, gradient_accumulation_steps_number,
		resolution_width_number, resolution_height_number, checkpointing_steps_number, max_grad_norm_number,
		use_fp16_checkbox, gradient_checkpointing_checkbox
	]

	# Define the list of output components for loading settings
	# The order of components in this list MUST match the order of keys in output_key_order
	# used within the load_settings_gradio_for_list_outputs function.
	load_outputs_list = [
        status_output, # First output should be the status message
        config_path_textbox,
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
    ]


	# 설정 불러오기 버튼 클릭 시
	# Use the function that returns a list/tuple of updates
	# The outputs parameter is the list of components to update
	load_settings_button.click(
		fn=load_settings_gradio_for_list_outputs, # Call the correct function
		inputs=config_path_textbox,
		# outputs is a list of components, matching the function's return list order
		outputs=load_outputs_list, # Use the defined list
		queue=False
	)


	# 설정 저장 버튼 클릭 시
	# save_settings_gradio 함수는 config_path와 settings_dict를 받음
	def save_settings_direct(
		config_path, pretrained_checkpoint_path, base_model_id, pretrained_lora_path,
		preference_dataset_path, model_save_path,
		num_epochs, batch_size, learning_rate, optimizer_type,
		dpo_beta, seed, gradient_accumulation_steps,
		resolution_width, resolution_height, checkpointing_steps, max_grad_norm,
		use_fp16, gradient_checkpointing
	):

		# Recreate settings_dict inside this function
		settings_dict = get_ui_settings(
			config_path, pretrained_checkpoint_path, base_model_id, pretrained_lora_path,
			preference_dataset_path, model_save_path,
			num_epochs, batch_size, learning_rate, optimizer_type,
			dpo_beta, seed, gradient_accumulation_steps,
			resolution_width, resolution_height, checkpointing_steps, max_grad_norm,
			use_fp16, gradient_checkpointing
		)
		# Now call the core save logic. It returns (message, None). We only need the message for status_output.
		message, _ = save_settings_gradio(config_path, settings_dict)
		return message # Return only message

	# Save button only needs to return a message to the status output
	save_settings_button.click(
		fn=save_settings_direct,
		inputs=all_input_components, # All components are inputs
		outputs=status_output,
		queue=False
	)


	# 미세조정 시작 버튼 클릭 시
	# Corrected logic for run:
	# 1. Capture all setting values from UI inputs.
	# 2. Call get_ui_settings to create the dict, outputting it to settings_state.
	# 3. Call run_training_gradio taking settings_state as input, streaming output to status_output.
	run_training_button.click(
		fn=get_ui_settings, # Step 1 & 2: Get settings dict from UI inputs
		inputs=all_input_components,
		outputs=settings_state, # Output the settings_dict into the defined state variable
		queue=True # This step is fast
	).then(
		fn=run_training_gradio, # Step 3: Use the settings_dict from the state as input
		inputs=settings_state, # Use the state variable from the previous step as input
		outputs=status_output, # Stream output to textbox
		queue=True # This step is long-running and uses yielding
	)


# --- Gradio 앱 실행 ---
if __name__ == "__main__":
	print("Starting Gradio server for LoRA DPO Fine-tuning GUI...") # <<< UPDATED message
	print(f"Script Dir: {SCRIPT_DIR}")
	print(f"Training script: {TRAINING_SCRIPT_PATH}")
	print("Server will run at http://0.0.0.0:7866 (Access via http://localhost:7866 or host IP)")
	# --- Ctrl+C 종료 안내 추가 ---
	print("Gradio 서버를 종료하려면 터미널에서 Ctrl+C를 누르세요.")
	# --- 안내 끝 ---

	# `--share`, `--server-port`, `--inbrowser` 등의 인자는 launch() 함수에서 직접 설정하거나
	# 명령줄 인자로 --share, --server-port 7866 등을 넘겨 실행할 수 있습니다.
	# 여기서는 launch 함수에서 기본 설정을 합니다.
	demo.launch(
		server_name="0.0.0.0",
		server_port=7866,
		share=False, # 개발 시에는 False, 공유 시 True로 변경하거나 명령줄 인자로 제어
		inbrowser=True, # 브라우저 자동 열기
		show_api=False, # API 탭 숨김
		# enable_queue=True # Blocks에서는 queue=True를 .click()에서 개별 설정
	)
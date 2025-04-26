import logging
from transformers import AutoTokenizer
from accelerate import Accelerator

logger = logging.getLogger(__name__)

def load_tokenizers(
	base_model_id_for_tokenizer: str,
	revision: str = None,
	cache_dir: str = None,
	accelerator: Accelerator = None, # 로깅용
) -> tuple:
	"""
	Tokenizer 두 개를 로드합니다.

	Args:
		base_model_id_for_tokenizer (str): 토크나이저 로드 기준 모델 ID 또는 경로.
		revision (str, optional): 모델 리비전.
		cache_dir (str, optional): 캐시 디렉토리.
		accelerator (Accelerator, optional): Accelerate 로거 사용 위한 객체.

	Returns:
		tuple: (tokenizer_one, tokenizer_two)
			   오류 발생 시 (None, None) 반환.
	"""
	print_fn = accelerator.print if accelerator else print
	tokenizer_one, tokenizer_two = None, None

	print_fn(f"Loading tokenizers from: {base_model_id_for_tokenizer}")
	try:
		tokenizer_one = AutoTokenizer.from_pretrained(
			base_model_id_for_tokenizer,
			revision=revision,
			use_fast=False,
			cache_dir=cache_dir,
			subfolder="tokenizer"
		)
		tokenizer_two = AutoTokenizer.from_pretrained(
			base_model_id_for_tokenizer,
			revision=revision,
			use_fast=False,
			cache_dir=cache_dir,
			subfolder="tokenizer_2"
		)
		print_fn("Tokenizers loaded successfully.")
		return tokenizer_one, tokenizer_two
	except Exception as e:
		print_fn(f"[ERROR] Failed to load tokenizers from {base_model_id_for_tokenizer}: {e}")
		logger.error("Failed to load tokenizers", exc_info=True)
		return None, None
import os
import logging
import random
from functools import partial
from typing import Optional, Dict, Any, Tuple
import argparse
import numpy as np
import time 
import math 
import json 

import torch
from PIL import Image
from torchvision import transforms
from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm # tqdm 임포트
from accelerate import Accelerator # Accelerator 임포트 (로깅용)


# 로거 설정 확인
logger = logging.getLogger(__name__)

# --- 토크나이징 함수 ---
def tokenize_captions(tokenizer: AutoTokenizer, prompts: list[str]) -> torch.Tensor:
	"""주어진 프롬프트 리스트를 토크나이징합니다."""
	try:
		return tokenizer(
			prompts,
			truncation=True,
			padding="max_length",
			max_length=tokenizer.model_max_length,
			return_tensors="pt",
		).input_ids
	except Exception as e:
		logger.error(f"Error in tokenize_captions: {e}")
		raise

# --- 데이터 전처리 함수 ---
def preprocess_data(examples: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
	"""JSONL 샘플을 처리하여 Latent 및 메타데이터를 로드합니다."""
	processed = {
		"latents_w": [], "latents_l": [],
		"prompt": [],
		"original_sizes_w": [], "crop_top_lefts_w": [],
	}
	skipped_count = 0
	prompt_col = "prompt_text_reference"
	score_col = "comparison_score"
	img0_path_col = "image_0_relative_path"
	img1_path_col = "image_1_relative_path"

	# 입력 examples 구조 확인 (batched=True 시 리스트 형태)
	if not isinstance(examples[prompt_col], list):
		# 단일 샘플 처리 (batched=False 또는 map 호출 방식에 따라)
		num_examples = 1
		# 입력 딕셔너리를 리스트로 감싸서 아래 로직 통일 (비효율적일 수 있음)
		# examples = {k: [v] for k, v in examples.items()}
		# 또는 아래 루프를 단일 처리로 변경
		logger.warning("Processing single example in preprocess_data. Batching is recommended.")
		# 단일 처리 로직 필요 (아래 코드는 배치 기준) - 현재 map(batched=True) 호출하므로 이 경우는 드묾
	else:
		num_examples = len(examples[prompt_col])

	for i in range(num_examples):
		try:
			prompt = examples[prompt_col][i]
			score = examples[score_col][i]
			img0_rel_path = examples[img0_path_col][i]
			img1_rel_path = examples[img1_path_col][i]

			if score == 1.0: rel_path_w, rel_path_l = img0_rel_path, img1_rel_path
			elif score == -1.0: rel_path_w, rel_path_l = img1_rel_path, img0_rel_path
			else: logger.warning(f"Sample {i} has invalid score ({score}). Skipping."); skipped_count += 1; continue

			base_path_w = os.path.join(args.image_data_root, rel_path_w)
			base_path_l = os.path.join(args.image_data_root, rel_path_l)
			cache_path_w = os.path.splitext(base_path_w)[0] + ".pt"
			cache_path_l = os.path.splitext(base_path_l)[0] + ".pt"

			try:
				data_w = torch.load(cache_path_w, map_location="cpu")
				data_l = torch.load(cache_path_l, map_location="cpu")
				latent_w = data_w['latent']
				original_size_w = tuple(data_w['original_size'].tolist())
				crop_top_left_w = tuple(data_w['crop_top_left'].tolist())
				latent_l = data_l['latent']
			except FileNotFoundError: logger.error(f"Latent cache missing: {cache_path_w} or {cache_path_l}. Skipping."); skipped_count += 1; continue
			except KeyError as e: logger.error(f"Missing key in cache file ({e}). Skipping."); skipped_count += 1; continue
			except Exception as e: logger.error(f"Error loading cache for sample {i}: {e}. Skipping."); skipped_count += 1; continue

			apply_flip = not args.no_hflip and random.random() < 0.5
			if apply_flip:
				latent_w = torch.flip(latent_w, dims=[-1])
				latent_l = torch.flip(latent_l, dims=[-1])

			processed["latents_w"].append(latent_w)
			processed["latents_l"].append(latent_l)
			processed["prompt"].append(prompt)
			processed["original_sizes_w"].append(original_size_w)
			processed["crop_top_lefts_w"].append(crop_top_left_w)

		except Exception as outer_e:
			logger.error(f"Outer error processing sample {i}: {outer_e}", exc_info=True)
			skipped_count += 1
			continue

	if skipped_count > 0: logger.warning(f"Skipped {skipped_count} samples during preprocessing batch.")
	return processed


# --- 데이터 로더 생성 함수 (기존 collate_fn 포함) ---
def collate_fn(examples: list[Dict[str, Any]], tokenizer_one: AutoTokenizer, tokenizer_two: AutoTokenizer) -> Dict[str, Any]:
	 if not examples: return {}

	 # latents 데이터 변환 및 스태킹
	 try:
		 latents_w_batch = []
		 latents_l_batch = []
		 for ex in examples:
			 latent_w_data = ex["latents_w"]
			 latent_l_data = ex["latents_l"]

			 # <<< 데이터 타입을 확인하고 Tensor로 변환 >>>
			 if isinstance(latent_w_data, list):
				 # print("[CollateFn Debug] Converting list to tensor for latents_w") # 필요시 디버그 출력
				 latent_w_tensor = torch.tensor(latent_w_data, dtype=torch.float16) # dtype 명시 (FP16 가정)
			 elif isinstance(latent_w_data, torch.Tensor):
				 latent_w_tensor = latent_w_data # 이미 텐서면 그대로 사용
			 else:
				 raise TypeError(f"Unexpected type for latents_w: {type(latent_w_data)}")

			 if isinstance(latent_l_data, list):
				 # print("[CollateFn Debug] Converting list to tensor for latents_l")
				 latent_l_tensor = torch.tensor(latent_l_data, dtype=torch.float16) # dtype 명시
			 elif isinstance(latent_l_data, torch.Tensor):
				 latent_l_tensor = latent_l_data
			 else:
				 raise TypeError(f"Unexpected type for latents_l: {type(latent_l_data)}")

			 latents_w_batch.append(latent_w_tensor)
			 latents_l_batch.append(latent_l_tensor)

		 # <<< 변환된 텐서 리스트를 stack >>>
		 latents_w = torch.stack(latents_w_batch)
		 latents_l = torch.stack(latents_l_batch)

	 except TypeError as e:
		 print(f"[CollateFn Error] Error during tensor conversion or stacking: {e}")
		 try: print(f"[CollateFn Error] Data structure causing error (first example latents_w): {examples[0]['latents_w']}")
		 except: print("[CollateFn Error] Could not log error data structure.")
		 return {}
	 except Exception as e:
		 print(f"[CollateFn Error] Unexpected error processing latents: {e}")
		 return {}

	 # 프롬프트 추출 및 토크나이징
	 prompts = [ex.get("prompt", "") for ex in examples]
	 if not prompts: print("[CollateFn Error] No prompts found."); return {}
	 try:
		 tokens_one = tokenize_captions(tokenizer_one, prompts)
		 tokens_two = tokenize_captions(tokenizer_two, prompts)
	 except Exception as e: print(f"[CollateFn Error] Error tokenizing captions: {e}"); return {}

	 # 배치 데이터 구성
	 batch_data = {
		 "latents_w": latents_w,
		 "latents_l": latents_l,
		 "input_ids_one": tokens_one,
		 "input_ids_two": tokens_two,
		 "original_sizes_w": [ex["original_sizes_w"] for ex in examples],
		 "crop_top_lefts_w": [ex["crop_top_lefts_w"] for ex in examples],
	 }
	 return batch_data

# --- 데이터셋 준비 메인 함수 (train_text_encoder_dpo.py 에서 이동) ---
def prepare_dataset(args, accelerator, tokenizer_one, tokenizer_two):
	"""데이터셋 로드, 전처리 및 DataLoader 생성을 담당합니다."""
	accelerator.print("Loading and preprocessing dataset...")
	try:
		train_dataset = load_dataset("json", data_files=args.dataset_path, cache_dir=args.cache_dir, split=args.dataset_split_name, streaming=args.use_streaming)
		logger.info(f"Dataset loaded {'in streaming mode' if args.use_streaming else 'in map-style mode'}.")
		if not args.use_streaming: logger.info(f"Dataset size (map-style): {len(train_dataset)} samples.")

		# 샘플 수 제한
		if args.max_train_samples is not None:
			if args.use_streaming:
				train_dataset = train_dataset.take(args.max_train_samples)
				logger.info(f"Limited streaming dataset to {args.max_train_samples} samples.")
			elif len(train_dataset) > args.max_train_samples:
				original_length = len(train_dataset)
				train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
				logger.info(f"Truncated map-style dataset from {original_length} to {args.max_train_samples} samples.")
			else:
				logger.info(f"Dataset size ({len(train_dataset)}) <= max_train_samples ({args.max_train_samples}). Using full dataset.")

		# 전처리 함수 정의 및 컬럼 확인
		preprocess_fn = partial(preprocess_data, args=args)
		required_cols = ["prompt_text_reference", "comparison_score", "image_0_relative_path", "image_1_relative_path"]
		if args.use_streaming:
			first_sample = next(iter(train_dataset)) # 첫 샘플로 컬럼 확인
			original_columns = list(first_sample.keys())
			if not all(col in original_columns for col in required_cols): raise ValueError("Streaming dataset missing required columns.")
			# 스트리밍 데이터셋은 map 적용 후 다시 생성 필요 -> 아래에서 다시 로드/take/map 적용
			train_dataset_stream_reset = load_dataset("json", data_files=args.dataset_path, cache_dir=args.cache_dir, split=args.dataset_split_name, streaming=True)
			if args.max_train_samples is not None: train_dataset_stream_reset = train_dataset_stream_reset.take(args.max_train_samples)
			train_dataset = train_dataset_stream_reset # 원본 iterator로 리셋
		else:
			original_columns = train_dataset.column_names
			if not all(col in original_columns for col in required_cols): raise ValueError("Map-style dataset missing required columns.")
		logger.info(f"Original dataset columns checked.")

		# .map() 적용
		num_proc = args.dataloader_num_workers if args.dataloader_num_workers > 0 and not args.use_streaming else None
		logger.info(f"Preprocessing dataset {'sequentially' if num_proc is None else f'with {num_proc} processes'}...")
		start_map_time = time.time()
		train_dataset_processed = train_dataset.map(preprocess_fn, batched=True, num_proc=num_proc) # remove_columns 자동
		end_map_time = time.time()
		logger.info(f"Dataset preprocessing finished in {end_map_time - start_map_time:.2f} seconds.")

		# DataLoader 생성
		accelerator.print("Creating DataLoader...")
		collate_fn_with_tokenizers = partial(collate_fn, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two)
		if args.use_streaming:
			train_dataset_processed = train_dataset_processed.shuffle(seed=args.seed, buffer_size=args.train_batch_size * 10)
			shuffle_dataloader = False
		else:
			shuffle_dataloader = True

		train_dataloader = DataLoader(
			train_dataset_processed, # <<< 전처리된 데이터셋 사용 >>>
			batch_size=args.train_batch_size, shuffle=shuffle_dataloader,
			collate_fn=collate_fn_with_tokenizers, num_workers=args.dataloader_num_workers, pin_memory=True
		)
		accelerator.print("DataLoader created.")

		# --- 데이터 로더 직접 테스트 ---
		accelerator.print("Attempting to fetch the first batch...")
		start_fetch_time = time.time()
		first_batch = next(iter(train_dataloader))
		end_fetch_time = time.time()
		accelerator.print(f"Successfully fetched the first batch in {end_fetch_time - start_fetch_time:.2f} seconds.")
		if first_batch and "latents_w" in first_batch: accelerator.print(f"First batch latents_w shape: {first_batch['latents_w'].shape}")
		else: accelerator.print("[WARN] First batch seems invalid or empty.")
		# --- 테스트 끝 ---

		# <<< train_dataset_processed 반환 (map 적용된 결과) >>>
		return train_dataloader, train_dataset_processed

	except Exception as e:
		logger.error(f"Failed to prepare dataset or dataloader: {e}", exc_info=True)
		return None, None

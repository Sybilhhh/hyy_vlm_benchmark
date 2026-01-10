#!/usr/bin/env python3
"""
Fill missing/invalid short_editing_instruction using Qwen3-32B model.

Usage:
# Single GPU
python fill_short_instruction.py --output_csv data_info_filled.csv

# Multi-GPU with torchrun
torchrun --nproc_per_node=8 fill_short_instruction.py --input_csv /home/dyvm6xra/dyvm6xrauser04/yuyang/SE_Vidgen/video_edit_1211_t2v/0101_dynamic_video/data_info2.csv --output_csv data_info_filled.csv --resume
"""

import csv
import os
import re
import argparse
import fcntl
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model path
DEFAULT_LLM_PATH = "/scratch/dyvm6xra/shared_cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"

# CSV columns
CSV_COLUMNS = [
    'video1_path', 'video2_path', 'video1_caption', 'video2_caption', 
    'editing_instruction', 'short_editing_instruction',
    'reverse_editing_instruction', 'short_reverse_editing_instruction',
]

# Target path prefix
TARGET_PREFIX = "home/dyvm6xra/dyvm6xrauser04/yuyang/SE_Vidgen/video_edit_1211_t2v/0101_dynamic_video/static_video/"


def setup_distributed():
    """Initialize distributed environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_chinese(text):
    """Check if text contains Chinese characters"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_invalid_short_instruction(text):
    """
    Check if short_editing_instruction is invalid (needs to be fixed).
    Returns: (is_invalid: bool, reason: str)
    """
    if not text or text.strip() == '':
        return True, 'empty'
    if is_chinese(text):
        return True, 'chinese'
    # Not a proper sentence (less than 3 words)
    words = text.split()
    if len(words) < 3:
        return True, 'too_short'
    return False, 'valid'


def load_csv_rows(csv_path, filter_prefix=None):
    """Load all rows from CSV file"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filter_prefix is None or row.get('video1_path', '').startswith(filter_prefix):
                rows.append(row)
    return rows


def load_existing_video1_paths(output_csv):
    """Load all video1_path values from output CSV"""
    existing_paths = set()
    if os.path.exists(output_csv):
        try:
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video1_path = row.get('video1_path', '')
                    if video1_path:
                        existing_paths.add(video1_path)
        except Exception as e:
            print(f"Warning: Could not read output CSV: {e}")
    return existing_paths


class ShortInstructionGenerator:
    def __init__(self, model_path, device):
        print(f"Loading Qwen3-32B model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Use SDPA instead of flash_attention_2
        )
        self.model = self.model.to(device).eval()
        self.device = device
        
        self.system_prompt = """You are an expert at summarizing video editing instructions. Your task is to create a SHORT editing instruction (~20-30 words) from a longer editing instruction.

Requirements:
- The short instruction should be ~20-30 words (concise but complete, no more than 30 words)
- Capture the core editing actions only
- Use imperative verbs: "Replace...", "Transform...", "Add...", "Remove...", "Change...", "Apply..."
- Start immediately with action verbs
- Focus on what TO CREATE/DO, not what to preserve
- FORBIDDEN: Never use "To transform", "In order to", "Begin by", "Start by", "Keep the same", "Maintain the", "Leave unchanged"
- Output ONLY the short instruction, nothing else"""

    def generate(self, editing_instruction):
        """Generate short_editing_instruction from editing_instruction"""
        if not editing_instruction or editing_instruction.strip() == '':
            return ""
        
        user_prompt = f"""Create a short editing instruction (~20-30 words) from this longer instruction:

"{editing_instruction}"

Output ONLY the short instruction: /no_think"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        answer = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clean up the answer
        answer = answer.strip()
        # Remove any thinking tags if present
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        
        return answer


def append_row_to_csv(row, output_csv, existing_paths=None):
    """
    Append a row to CSV with file locking for concurrent safety.
    Only append if video1_path doesn't already exist in output CSV.
    """
    video1_path = row.get('video1_path', '')
    
    # Check if this video1_path already exists
    if existing_paths is not None and video1_path in existing_paths:
        return False  # Skip, already exists
    
    file_exists = os.path.exists(output_csv)
    
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists or os.path.getsize(output_csv) == 0:
                writer.writeheader()
            
            # Ensure only CSV columns are written
            filtered_row = {col: row.get(col, '') for col in CSV_COLUMNS}
            writer.writerow(filtered_row)
            return True
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def parse_args():
    parser = argparse.ArgumentParser(description="Fill missing short_editing_instruction using Qwen3-32B")
    
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/dyvm6xra/dyvm6xrauser04/yuyang/SE_Vidgen/video_edit_1211_t2v/0101_dynamic_video/data_info.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/dyvm6xra/dyvm6xrauser04/yuyang/SE_Vidgen/video_edit_1211_t2v/0101_dynamic_video/data_info_filled.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        default=DEFAULT_LLM_PATH,
        help="Path to Qwen3-32B model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output CSV, skip already processed items",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of items to process",
    )
    parser.add_argument(
        "--batch_log_interval",
        type=int,
        default=100,
        help="Log progress every N items",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize distributed environment
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_main_process:
        print("\n" + "="*80)
        print("🚀 SHORT INSTRUCTION FILLER (Qwen3-32B)")
        print("="*80)
        print(f"Input CSV: {args.input_csv}")
        print(f"Output CSV: {args.output_csv}")
        print(f"LLM Model: {args.llm_path}")
        print(f"Distributed: {world_size} processes")
        print("="*80 + "\n")
    
    # Load existing video1_paths from output CSV to avoid duplicates
    existing_paths = load_existing_video1_paths(args.output_csv)
    if is_main_process and existing_paths:
        print(f"📋 Found {len(existing_paths)} existing video1_paths in output CSV")
    
    # Load data on main process
    if is_main_process:
        print(f"📖 Loading CSV data from: {args.input_csv}")
        all_rows = load_csv_rows(args.input_csv, filter_prefix=TARGET_PREFIX)
        print(f"✅ Loaded {len(all_rows)} rows matching target prefix")
        
        # Filter rows that need fixing
        rows_to_fix = []
        rows_valid = []
        for row in all_rows:
            video1_path = row.get('video1_path', '')
            
            # Skip if already in output CSV (regardless of resume flag)
            if video1_path in existing_paths:
                continue
                
            invalid, reason = is_invalid_short_instruction(row.get('short_editing_instruction', ''))
            if invalid:
                rows_to_fix.append(row)
            else:
                rows_valid.append(row)
        
        print(f"📊 Valid rows to copy: {len(rows_valid)}")
        print(f"📊 Rows needing fix: {len(rows_to_fix)}")
        
        # Resume mode additional filtering (if needed)
        if args.resume:
            # Already filtered by existing_paths, just log
            print(f"🔄 Resume mode: Skipping {len(existing_paths)} already processed items")
        
        # Limit samples if specified
        if args.max_samples:
            rows_to_fix = rows_to_fix[:args.max_samples]
            print(f"✂️ Limited to {args.max_samples} samples")
        
        # Copy valid rows to output (only if not already in output)
        if rows_valid:
            print(f"📝 Copying {len(rows_valid)} valid rows to output...")
            for row in tqdm(rows_valid, desc="Copying valid rows"):
                append_row_to_csv(row, args.output_csv, existing_paths)
                # Update existing_paths to avoid duplicates in the same run
                existing_paths.add(row.get('video1_path', ''))
    else:
        rows_to_fix = None
        rows_valid = None
    
    # Broadcast to all processes
    if world_size > 1:
        data_list = [rows_to_fix]
        dist.broadcast_object_list(data_list, src=0)
        rows_to_fix = data_list[0]
    
    if rows_to_fix is None or len(rows_to_fix) == 0:
        if is_main_process:
            print("✅ No items need processing. Done!")
        cleanup_distributed()
        return
    
    # Split work across processes
    items_per_process = len(rows_to_fix) // world_size
    start_idx = rank * items_per_process
    end_idx = len(rows_to_fix) if rank == world_size - 1 else start_idx + items_per_process
    my_items = rows_to_fix[start_idx:end_idx]
    
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"🤖 Loading LLM Model...")
        print(f"{'='*80}")
    
    # Initialize generator
    print(f"⏳ Rank {rank}: Loading model...")
    generator = ShortInstructionGenerator(args.llm_path, device)
    print(f"✅ Rank {rank}: Model loaded")
    
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"🎬 Processing {len(rows_to_fix)} rows across {world_size} GPUs")
        print(f"📊 Each GPU processes ~{len(my_items)} rows")
        print(f"{'='*80}")
    
    # Process items
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.time()
    
    for idx, row in enumerate(tqdm(my_items, desc=f"Rank {rank}", disable=(rank != 0))):
        try:
            video1_path = row.get('video1_path', '')
            
            # Double-check if this video1_path already exists (race condition protection)
            if video1_path in existing_paths:
                skipped_count += 1
                continue
            
            editing_instruction = row.get('editing_instruction', '')
            
            # Generate short instruction
            short_instruction = generator.generate(editing_instruction)
            
            if short_instruction:
                row['short_editing_instruction'] = short_instruction
                processed_count += 1
            else:
                failed_count += 1
            
            # Save result (with duplicate check)
            if append_row_to_csv(row, args.output_csv, existing_paths):
                # Update existing_paths to avoid duplicates
                existing_paths.add(video1_path)
            else:
                skipped_count += 1
            
        except Exception as e:
            print(f"⚠️ Rank {rank}: Error processing row {idx}: {e}")
            failed_count += 1
            # Still try to save the row with original (possibly invalid) short_editing_instruction
            append_row_to_csv(row, args.output_csv, existing_paths)
            existing_paths.add(video1_path)
        
        # Log progress
        if (idx + 1) % args.batch_log_interval == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"📊 Rank {rank}: {idx + 1}/{len(my_items)} | Success: {processed_count} | Skipped: {skipped_count} | Failed: {failed_count} | Speed: {speed:.2f}/s")
    
    print(f"✅ Rank {rank}: Done! {processed_count} success, {skipped_count} skipped, {failed_count} failed")
    
    # Synchronize
    if world_size > 1:
        try:
            dist.barrier()
        except Exception:
            pass
    
    if is_main_process:
        print(f"\n{'='*80}")
        print("🎉 PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        # Count final output
        if os.path.exists(args.output_csv):
            with open(args.output_csv, 'r', encoding='utf-8') as f:
                csv_count = sum(1 for _ in f) - 1
            print(f"📊 Total rows in output CSV: {csv_count}")
        
        print(f"✅ Results saved to: {args.output_csv}")
        print(f"{'='*80}\n")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
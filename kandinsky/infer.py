#!/usr/bin/env python3
"""
Unified inference script for Kandinsky-5 fine-tuned checkpoints.

Loads a DiT checkpoint into the Kandinsky5UnifiedPipeline and runs inference
over a benchmark CSV. Supports t2v, t2i, ti2i, tv2v, prop, i2v, tv2c task types.

Supports multi-GPU data parallelism via torchrun: each rank loads a full
pipeline on its own GPU and processes a disjoint shard of CSV rows.

The task type is determined per-row from a 'task' column in the CSV.
If the CSV has no 'task' column, a --task fallback must be supplied.

CSV formats:
  - t2v: 'caption' column
  - tv2v: 'video_path' + 'instruction'
  - ti2i: 'image_path' (or 'video_path') + 'instruction'
  - t2i: 'caption' column

Usage (single GPU):
    python infer_unified.py \
        --conf_path configs/k5_unified.yaml \
        --dit_checkpoint outputs/unified_run/checkpoint-10/dit.safetensors \
        --csv_path /path/to/benchmark.csv \
        --output_dir outputs/infer/checkpoint-10/bench

Usage (multi-GPU):
    torchrun --nproc_per_node=4 infer_unified.py \
        --conf_path configs/k5_unified.yaml \
        --dit_checkpoint outputs/unified_run/checkpoint-10/dit.safetensors \
        --csv_path /path/to/benchmark.csv \
        --output_dir outputs/infer/checkpoint-10/bench
"""

import argparse
import csv
import logging
import os

import torch
from omegaconf import OmegaConf

from kandinsky.utils import get_unified_pipeline

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _get_dist_info():
    """Return (rank, local_rank, world_size) from torchrun env vars, or (0, 0, 1) for single-GPU."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


def read_csv(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _resolve_task(row, fallback_task):
    """Get the task from the row's 'task' or 'task_type' column, falling back to fallback_task."""
    task = row.get("task_type", "").strip() or row.get("task", "").strip()
    if task:
        return task
    if fallback_task:
        return fallback_task
    raise ValueError(
        f"Row has no 'task' column and no --task fallback was provided. "
        f"Row keys: {list(row.keys())}"
    )


def _mask_distributed_env():
    """Temporarily remove distributed env vars so get_unified_pipeline sees world_size=1.

    For data-parallel inference each rank loads an independent pipeline on its
    own GPU.  We hide the torchrun env so the pipeline constructor doesn't try
    to set up tensor parallelism or collective ops.
    """
    saved = {}
    for key in ("LOCAL_RANK", "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE"):
        saved[key] = os.environ.pop(key, None)
    return saved


def _restore_distributed_env(saved):
    for key, val in saved.items():
        if val is not None:
            os.environ[key] = val


# ---------------------------------------------------------------------------
# Core loop: run an already-constructed pipeline over CSV rows
# ---------------------------------------------------------------------------

def run_csv_rows_with_pipeline(
    pipeline,
    csv_path,
    output_dir,
    data_root=None,
    fallback_task=None,
    num_steps=None,
    guidance_weight=None,
    seed=42,
    time_length=5,
    height=None,
    width=None,
    expand_prompts=False,
    max_samples=None,
    rank=0,
    world_size=1,
):
    """Iterate over a benchmark CSV and call *pipeline* for each sample.

    This is the shared core used both by the standalone script and by the
    in-training benchmark path.  The caller is responsible for constructing
    (and later cleaning up) the pipeline object.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_rows = read_csv(csv_path)
    if max_samples is not None:
        all_rows = all_rows[:max_samples]

    original_indices = list(range(len(all_rows)))
    my_indices = original_indices[rank::world_size]
    my_rows = [all_rows[i] for i in my_indices]

    total = len(all_rows)
    logger.info(
        f"[rank {rank}/{world_size}] Running inference on "
        f"{len(my_rows)}/{total} samples from {csv_path}"
    )

    for local_idx, (orig_idx, row) in enumerate(zip(my_indices, my_rows)):
        task = _resolve_task(row, fallback_task)

        # Video-style outputs for t2v / tv2v / prop / i2v / tv2c
        ext = ".mp4" if task in ("t2v", "tv2v", "prop", "i2v", "tv2c") else ".png"
        save_path = os.path.join(output_dir, f"{orig_idx:04d}{ext}")

        if os.path.exists(save_path):
            logger.info(f"[{local_idx+1}/{len(my_rows)}] Skipping (exists): {save_path}")
            continue

        if task in ("t2v", "t2i"):
            text = row.get("caption", row.get("instruction", ""))
        else:
            text = row.get("instruction", row.get("caption", ""))

        video_path = None
        if task in ("tv2v", "prop", "i2v", "tv2c"):
            # Prefer explicit video_path, fall back to video1_path for edit-style CSVs
            video_path = row.get("video_path", "") or row.get("video1_path", "")
            if data_root and video_path and not os.path.isabs(video_path):
                video_path = os.path.join(data_root, video_path)

        image_path = None
        if task == "ti2i":
            # ti2i: source image, or a single-frame video path
            image_path = row.get("image_path", row.get("video_path", ""))
            if data_root and image_path and not os.path.isabs(image_path):
                image_path = os.path.join(data_root, image_path)

        logger.info(
            f"[rank {rank}][{local_idx+1}/{len(my_rows)}] task={task} "
            f"text={text[:80]}{'...' if len(text) > 80 else ''}"
        )

        try:
            pipeline(
                text=text,
                task_type=task,
                video=video_path,
                image=image_path,
                time_length=time_length,
                height=height,
                width=width,
                seed=seed,
                num_steps=num_steps,
                guidance_weight=guidance_weight,
                expand_prompts=expand_prompts,
                save_path=save_path,
                progress=True,
            )
            logger.info(f"  Saved: {save_path}")
        except Exception as e:
            logger.error(f"  Failed on sample {orig_idx}: {e}")
            continue

    logger.info(f"[rank {rank}/{world_size}] Done. Outputs in {output_dir}")


# ---------------------------------------------------------------------------
# Standalone entry point: loads pipeline from disk, runs CSV, cleans up
# ---------------------------------------------------------------------------

def run_inference_on_csv(
    conf_path,
    dit_checkpoint,
    csv_path,
    output_dir,
    data_root=None,
    fallback_task=None,
    num_steps=None,
    guidance_weight=None,
    seed=42,
    time_length=5,
    height=None,
    width=None,
    quantized_qwen=False,
    text_token_padding=False,
    expand_prompts=False,
    max_samples=None,
    rank=0,
    local_rank=0,
    world_size=1,
):
    """Load a fresh pipeline from a checkpoint file, run CSV inference, clean up.

    This is the standalone path used by the CLI and run_inference.sh.
    For in-training benchmarks, use run_csv_rows_with_pipeline directly
    with the already-loaded models.

    For multi-GPU, each rank loads an independent pipeline on cuda:<local_rank>
    and processes a disjoint shard of CSV rows. No collectives are used.
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    offload = world_size == 1

    logger.info(
        f"[rank {rank}/{world_size}] Loading pipeline on {device} from "
        f"config={conf_path}, dit={dit_checkpoint}"
    )

    conf = OmegaConf.load(conf_path)
    conf.model.checkpoint_path = dit_checkpoint

    tmp_conf_path = os.path.join(output_dir, f"_tmp_conf_rank{rank}.yaml")
    OmegaConf.save(conf, tmp_conf_path)

    saved_env = _mask_distributed_env()
    try:
        pipeline = get_unified_pipeline(
            device_map=device,
            conf_path=tmp_conf_path,
            offload=offload,
            quantized_qwen=quantized_qwen,
            text_token_padding=text_token_padding,
        )
    finally:
        _restore_distributed_env(saved_env)

    if os.path.exists(tmp_conf_path):
        os.remove(tmp_conf_path)
    logger.info(f"[rank {rank}/{world_size}] Pipeline loaded on {device}")

    run_csv_rows_with_pipeline(
        pipeline=pipeline,
        csv_path=csv_path,
        output_dir=output_dir,
        data_root=data_root,
        fallback_task=fallback_task,
        num_steps=num_steps,
        guidance_weight=guidance_weight,
        seed=seed,
        time_length=time_length,
        height=height,
        width=width,
        expand_prompts=expand_prompts,
        max_samples=max_samples,
        rank=rank,
        world_size=world_size,
    )

    del pipeline
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Kandinsky-5 unified inference")

    parser.add_argument("--conf_path", type=str, required=True,
                        help="Path to unified YAML config")
    parser.add_argument("--dit_checkpoint", type=str, required=True,
                        help="Path to fine-tuned dit.safetensors")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to benchmark CSV")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory prepended to relative media paths")
    parser.add_argument("--task", type=str, default=None,
                        choices=["t2v", "t2i", "ti2i", "tv2v", "prop", "i2v", "tv2c"],
                        help="Fallback task type (used when CSV has no 'task' column)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated outputs")

    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--guidance_weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_length", type=int, default=5)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)

    parser.add_argument("--quantized_qwen", action="store_true")
    parser.add_argument("--text_token_padding", action="store_true")
    parser.add_argument("--expand_prompts", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = _get_dist_info()

    run_inference_on_csv(
        conf_path=args.conf_path,
        dit_checkpoint=args.dit_checkpoint,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        data_root=args.data_root,
        fallback_task=args.task,
        num_steps=args.num_steps,
        guidance_weight=args.guidance_weight,
        seed=args.seed,
        time_length=args.time_length,
        height=args.height,
        width=args.width,
        quantized_qwen=args.quantized_qwen,
        text_token_padding=args.text_token_padding,
        expand_prompts=args.expand_prompts,
        max_samples=args.max_samples,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified inference script for Kandinsky-5 fine-tuned checkpoints.

Loads a DiT checkpoint into the Kandinsky5UnifiedPipeline and runs inference
over a benchmark CSV. Supports t2v, t2i, ti2i, tv2v, prop, i2vtask types.

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
import traceback
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from kandinsky.utils import get_unified_pipeline
from kandinsky.unified_pipeline import SUPPORTED_TASK_TYPES

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BUCKET_GROUPS = [
    [(480, 848), (544, 720), (640, 640), (720, 544), (848, 480)],
    [(720, 1280), (832, 1104), (960, 960), (1104, 832), (1280, 720)],
    [(768, 1360), (880, 1184), (1024, 1024), (1184, 880), (1360, 768)],
    [(1088, 1920), (1248, 1664), (1440, 1440), (1664, 1248), (1920, 1088)],
]
STANDARD_FRAME_COUNTS = [49, 81, 121]


def get_video_info(video_path):
    """Return (height, width, num_frames, fps) using decord."""
    import decord

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    h, w = vr[0].shape[0], vr[0].shape[1]
    n = len(vr)
    fps = vr.get_avg_fps()
    return h, w, n, fps


def get_bucket(h, w, max_group=None):
    """Lucy/Hunyuan-style: pick resolution bucket by pixels + aspect ratio."""
    if h == 0 or w == 0:
        return (480, 848)
    groups = BUCKET_GROUPS[:max_group] if max_group is not None else BUCKET_GROUPS
    input_pixels = h * w
    input_ratio = w / h
    best_group = groups[0]
    for group in groups:
        rep_h, rep_w = group[2]
        if rep_h * rep_w <= input_pixels:
            best_group = group
        else:
            break
    best_bucket = None
    min_ratio_diff = float("inf")
    for bucket_h, bucket_w in best_group:
        diff = abs((bucket_w / bucket_h) - input_ratio)
        if diff < min_ratio_diff:
            min_ratio_diff = diff
            best_bucket = (bucket_h, bucket_w)
    return best_bucket or (480, 848)


def get_target_frames(num_frames, frame_mode="auto"):
    """Lucy-style target frame count (49/81/121 or nearest 4k+1)."""
    if frame_mode == "49":
        return 49
    if frame_mode == "81":
        return 81
    if frame_mode == "nearest":
        k = max(0, round((num_frames - 1) / 4))
        return max(1, min(4 * k + 1, 121))
    if abs(num_frames - 49) <= abs(num_frames - 81):
        return 49
    if num_frames < 121:
        return 81
    return 81


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


def _extract_frame_from_video(video_path, frame_idx=0):
    """Extract a single frame from video as PIL Image. For keyframe when CSV has no image_path."""
    import decord
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frame_idx = min(frame_idx, len(vr) - 1)
    frame = vr[frame_idx].asnumpy()
    return Image.fromarray(frame).convert("RGB")


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
    guide_image_base_path=None,
    use_prop_bucket=False,
    frame_mode="auto",
    group_pixels=None,
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
    task_counter = Counter()
    logger.info(
        f"[rank {rank}/{world_size}] Running inference on "
        f"{len(my_rows)}/{total} samples from {csv_path}"
    )

    for local_idx, (orig_idx, row) in enumerate(zip(my_indices, my_rows)):
        task = _resolve_task(row, fallback_task)
        task_counter[task] += 1

        # Video-style outputs for t2v / tv2v / prop / i2v / keyframe
        ext = ".mp4" if task in ("t2v", "tv2v", "prop", "i2v", "keyframe") else ".png"
        save_path = os.path.join(output_dir, f"{orig_idx:04d}{ext}")

        if os.path.exists(save_path):
            logger.info(f"[{local_idx+1}/{len(my_rows)}] Skipping (exists): {save_path}")
            continue

        if task in ("t2v", "t2i"):
            # 生成类任务优先使用 caption，其次 instruction
            text = row.get("caption", "") or row.get("instruction", "")
        else:
            # 编辑 / 传播类任务：instruction 与 editing_instruction 并行使用
            instr = row.get("instruction", "") or ""
            edit_instr = row.get("editing_instruction", "") or ""
            if instr and edit_instr:
                text = f"{instr}\n{edit_instr}"
            else:
                text = instr or edit_instr or row.get("caption", "")

        video_path = None
        if task in ("tv2v", "prop", "keyframe"):
            # Prefer explicit video_path, fall back to video1_path for edit-style CSVs
            video_path = row.get("video_path", "") or row.get("video1_path", "")
            if data_root and video_path and not os.path.isabs(video_path):
                video_path = os.path.join(data_root, video_path)
            if not video_path or not os.path.isfile(video_path):
                logger.error(
                    f"[rank {rank}] prop requires an existing video_path/video1_path. "
                    f"Skip sample {orig_idx}: missing source video {video_path}"
                )
                continue

        image_path = None
        if task in ("ti2i", "i2v"):
            # ti2i: source image, or a single-frame video path
            image_path = row.get("image_path", "") or row.get("guided_image_path", "") or row.get("guide_image_path", "")
            if data_root and image_path and not os.path.isabs(image_path):
                image_path = os.path.join(data_root, image_path)

        guide_image_path = None
        if task in ("prop", "keyframe"):  
            guide_image_path = row.get("guide_image_path", "") or row.get("guided_image_path", "")
            if guide_image_path and not os.path.isabs(guide_image_path) and data_root:
                guide_image_path = os.path.join(data_root, guide_image_path)
            if not guide_image_path or not os.path.isfile(guide_image_path):
                logger.error(
                    f"[rank {rank}] {task} requires guide_image_path. "
                    f"Skip sample {orig_idx}: path={guide_image_path!r}"
                )
                continue
                

        # Lucy/Hunyuan-style prop bucket (resolution) from source video
        prop_height, prop_width = height, width
        if task in ("prop", "keyframe") and use_prop_bucket and video_path and (height is None or width is None):
            try:
                vh, vw, nf, _ = get_video_info(video_path)
                prop_height, prop_width = get_bucket(vh, vw, max_group=group_pixels)
                nf_use = nf
                if row.get("frames") not in (None, ""):
                    try:
                        nf_use = min(nf, int(row["frames"]))
                    except (ValueError, TypeError):
                        pass
                prop_num_frames = get_target_frames(nf_use, frame_mode)
                logger.info(
                    f"[rank {rank}] Lucy-style prop bucket: video {vh}x{vw} {nf}f -> "
                    f"bucket {prop_height}x{prop_width} {prop_num_frames}f (target)"
                )
            except Exception as e:
                logger.warning(f"[rank {rank}] use_prop_bucket failed: {e}, using defaults")
                prop_height, prop_width = height, width
        effective_height = prop_height if task in ("prop", "keyframe") else height
        effective_width = prop_width if task in ("prop", "keyframe") else width

        logger.info(
            f"[rank {rank}][{local_idx+1}/{len(my_rows)}] task={task} "
            f"text={text[:80]}{'...' if len(text) > 80 else ''}"
        )

        if task in ("prop", "keyframe"):
            logger.info(
                f"[rank {rank}] Paths for sample {orig_idx}: "
                f"video_path={video_path}, image_path={image_path}, "
                f"guide_image_path={guide_image_path}, save_path={save_path}"
            )

        try:
            extra_kwargs = {}
            if task in ("prop", "keyframe"):
                extra_kwargs["guide_image"] = guide_image_path

            pipeline(
                text=text,
                task_type=task,
                video=video_path,
                image=image_path ,
                time_length=time_length,
                height=effective_height,
                width=effective_width,
                seed=seed,
                num_steps=num_steps,
                guidance_weight=guidance_weight,
                expand_prompts=expand_prompts,
                save_path=save_path,
                progress=True,
                **extra_kwargs,
            )
            logger.info(f"  Saved: {save_path}")
        except Exception as e:
            logger.error(f"  Failed on sample {orig_idx}: {e}")
            if orig_idx == 0:  # Log full traceback for first failure
                for line in traceback.format_exc().splitlines():
                    logger.error(f"    {line}")
            continue

    logger.info(f"[rank {rank}/{world_size}] Done. Outputs in {output_dir}")
    logger.info(
        f"[rank {rank}/{world_size}] Task stats (rows per task): {dict(task_counter)}"
    )


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
    guide_image_base_path=None,
    use_prop_bucket=False,
    frame_mode="auto",
    group_pixels=None,
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
    try:
        logger.info(f"Pipeline SUPPORTED_TASK_TYPES: {SUPPORTED_TASK_TYPES}")
    except Exception:
        pass

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
        guide_image_base_path=guide_image_base_path,
        use_prop_bucket=use_prop_bucket,
        frame_mode=frame_mode,
        group_pixels=group_pixels,
        rank=rank,
        world_size=world_size,
    )

    del pipeline
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass  # Ignore CUDA errors during cleanup (e.g. after earlier inference failures)


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
                        choices=["t2v", "t2i", "ti2i", "tv2v", "prop", "i2v", "keyframe"],
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

    parser.add_argument("--guide_image_base_path", type=str, default=None,
                        help="For prop: base dir for guide images; path = base_path / save_name (e.g. xxx.png)")
    parser.add_argument("--use_prop_bucket", action="store_true",
                        help="For prop: Lucy-style resolution from source video (bucket)")
    parser.add_argument("--frame_mode", type=str, default="auto",
                        choices=["49", "81", "auto", "nearest"],
                        help="When use_prop_bucket: target frame count mode (logging only in this script)")
    parser.add_argument("--group_pixels", type=int, default=None,
                        help="Hunyuan-style: max bucket group (1-4) to use; limits resolution")

    return parser.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = _get_dist_info()

    # Suppress heavy HuggingFace/transformers loading progress (e.g. "Loading weights")
    # so that logs remain compact during multi-GPU runs.
    try:
        import transformers

        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        # Also disable HF Hub download progress bars just in case.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    except Exception:
        pass

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
        guide_image_base_path=args.guide_image_base_path,
        use_prop_bucket=args.use_prop_bucket,
        frame_mode=args.frame_mode,
        group_pixels=args.group_pixels,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


if __name__ == "__main__":
    main()

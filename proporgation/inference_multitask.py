"""
Pure TV2V/Propagation inference (NO QwenVL / NO projector/connector).

This script uses:
- `WanVideoPipeline` and `ModelConfig` from `diffsynth.pipelines.wan_video_tv2v`
- Local model directory (no downloading) via ModelConfig(path=...)
- Optional checkpoint loading for `pipe.dit` from a .safetensors file

Example (TV2V):
python examples/wanvideo/model_training/inference_multitask.py \
  --model_dir /scratch/dyvm6xra/dyvm6xrauser13/models/Wan-AI/Wan2.2-TI2V-5B \
  --tokenizer_path /scratch/dyvm6xra/dyvm6xrauser13/models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl/ \
  --video_path /path/to/input.mp4 \
  --instruction "Add snow to the scene" \
  --output_path outputs/tv2v_output.mp4

Example (Propagation, single):
python examples/wanvideo/model_training/inference_multitask.py \
  --model_dir /scratch/dyvm6xra/dyvm6xrauser13/models/Wan-AI/Wan2.2-TI2V-5B \
  --tokenizer_path /scratch/dyvm6xra/dyvm6xrauser13/models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl/ \
  --task_type prop \
  --video_path /path/to/video1.mp4 \
  --img2_path /path/to/cond_frame.png \
  --instruction "" \
  --output_path outputs/prop_output.mp4
"""

from __future__ import annotations

import argparse
import glob
import os
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import load_file

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video_tv2v import WanVideoPipeline, ModelConfig  # type: ignore


DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, low quality, jpeg artifacts, blurry, watermark, subtitles, static, "
    "bad anatomy, deformed, extra fingers, bad hands, bad face"
)


# -----------------------------
# Distributed helpers (optional)
# -----------------------------

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def init_distributed() -> bool:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print(f"[dist] initialized: world_size={world_size}")
        return True
    return False


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


# -----------------------------
# Model loading / checkpoint
# -----------------------------

def build_model_configs_from_dir(model_dir: str) -> list[ModelConfig]:
    model_dir = os.path.abspath(model_dir)
    dit_files = sorted(glob.glob(os.path.join(model_dir, "diffusion_pytorch_model-*.safetensors")))
    if not dit_files:
        single = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
        if os.path.exists(single):
            dit_files = [single]
    if not dit_files:
        raise FileNotFoundError(f"No DiT safetensors found in: {model_dir}")

    t5_file = os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    if not os.path.exists(t5_file):
        t5_file = os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.safetensors")
    if not os.path.exists(t5_file):
        raise FileNotFoundError(f"No T5 file found in: {model_dir}")

    vae_file = os.path.join(model_dir, "Wan2.2_VAE.pth")
    if not os.path.exists(vae_file):
        vae_file = os.path.join(model_dir, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_file):
        raise FileNotFoundError(f"No VAE file found in: {model_dir}")

    return [
        ModelConfig(path=dit_files),
        ModelConfig(path=t5_file),
        ModelConfig(path=vae_file),
    ]


def load_checkpoint_dit(pipe: WanVideoPipeline, checkpoint_path: str):
    """Load only DiT weights from a .safetensors checkpoint into pipe.dit."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = load_file(checkpoint_path)
    # Common patterns:
    # - prefixed keys: "pipe.dit.*" or "dit.*"
    # - raw DiT keys (from this repo's multitask training): e.g. "blocks.0.self_attn.q.weight"
    dit_keys = {k: v for k, v in state_dict.items() if k.startswith("pipe.dit.") or k.startswith("dit.")}
    if dit_keys:
        clean = {k.replace("pipe.dit.", "").replace("dit.", ""): v for k, v in dit_keys.items()}
        ckpt_kind = "prefixed"
    else:
        # Fallback: treat the entire checkpoint as DiT weights.
        # This is the format produced by `train_multitask.py` epoch checkpoints (raw module keys).
        clean = state_dict
        ckpt_kind = "raw"

    if not clean:
        raise ValueError(f"Empty checkpoint (no tensors): {checkpoint_path}")

    # Check if ref_conv weights are in checkpoint
    ref_conv_keys = [k for k in clean.keys() if k.startswith("ref_conv")]
    if ref_conv_keys:
        print(f"[ckpt] Found ref_conv weights in checkpoint: {ref_conv_keys}")
    else:
        print(f"[ckpt] WARNING: No ref_conv weights found in checkpoint! Token concat may not work properly.")

    res = pipe.dit.load_state_dict(clean, strict=False)
    missing_keys, unexpected_keys = res
    print(f"[ckpt] loaded dit ({ckpt_kind}): {checkpoint_path} ({len(clean)} tensors)")
    
    # Check ref_conv loading status
    ref_conv_missing = [k for k in missing_keys if k.startswith("ref_conv")]
    if ref_conv_missing:
        print(f"[ckpt] WARNING: ref_conv keys missing after load: {ref_conv_missing}")
    elif ref_conv_keys:
        print(f"[ckpt] ref_conv weights loaded successfully!")
    
    if len(missing_keys) > 0:
        non_ref_missing = [k for k in missing_keys if not k.startswith("ref_conv")]
        if non_ref_missing:
            print(f"[ckpt] other missing keys (up to 5): {non_ref_missing[:5]}")


def ensure_ref_conv(pipe: WanVideoPipeline, reference_concat_method: str):
    """Ensure ref_conv exists on DiT if using token or hybrid concat method."""
    # In our propagation setup, `channel_real` can also use token-concat style conditioning via `conditional_image`.
    if reference_concat_method not in ["token", "hybrid", "channel_real"]:
        return
    
    import torch.nn as nn
    
    def _add_ref_conv(dit, vae):
        if dit is None:
            return
        if hasattr(dit, "ref_conv") and getattr(dit, "ref_conv") is not None:
            print(f"[ref_conv] DiT already has ref_conv")
            return
        # Get VAE z_dim dynamically (WanVideoVAE: z_dim=16, WanVideoVAE38: z_dim=48)
        in_ch = getattr(vae, 'z_dim', 16) if vae is not None else 16
        k = 2
        s = 2
        dit.ref_conv = nn.Conv2d(in_ch, dit.dim, kernel_size=(k, k), stride=(s, s))
        dit.has_ref_conv = True
        # Move to same device and dtype as dit
        device = next(dit.parameters()).device
        dtype = next(dit.parameters()).dtype
        dit.ref_conv = dit.ref_conv.to(device=device, dtype=dtype)
        print(f"[ref_conv] Created ref_conv for DiT: in_ch={in_ch}, out_ch={dit.dim}, kernel={k}, stride={s}")
    
    vae = getattr(pipe, "vae", None)
    _add_ref_conv(getattr(pipe, "dit", None), vae)
    _add_ref_conv(getattr(pipe, "dit2", None), vae)


def ensure_channel_real_patch_embedding(pipe: WanVideoPipeline, reference_concat_method: str):
    """Modify DiT's patch_embedding to accept 2x channels for channel_real mode."""
    if reference_concat_method != "channel_real":
        return
    
    import torch.nn as nn
    
    def _modify_patch_embedding(dit, vae):
        if dit is None:
            return
        
        # Get VAE z_dim dynamically
        z_dim = getattr(vae, 'z_dim', 48) if vae is not None else 48
        target_in_channels = z_dim * 2  # 48 -> 96
        
        # Check if already modified
        if dit.patch_embedding.in_channels == target_in_channels:
            print(f"[channel_real] patch_embedding already has {target_in_channels} in_channels")
            return
        
        original_pe = dit.patch_embedding
        device = original_pe.weight.device
        dtype = original_pe.weight.dtype
        
        # Create new Conv3d with 2x input channels
        new_pe = nn.Conv3d(
            target_in_channels, dit.dim,
            kernel_size=original_pe.kernel_size,
            stride=original_pe.stride,
            padding=original_pe.padding,
            dilation=original_pe.dilation,
            groups=original_pe.groups,
            bias=original_pe.bias is not None
        ).to(device=device, dtype=dtype)
        
        # Initialize weights: first half (reference) to zeros, second half from original
        with torch.no_grad():
            new_pe.weight[:, :z_dim] = 0.0  # Reference channels (first half)
            new_pe.weight[:, z_dim:] = original_pe.weight.clone()  # Input channels (second half)
            if original_pe.bias is not None:
                new_pe.bias.copy_(original_pe.bias)
        
        dit.patch_embedding = new_pe
        dit.in_dim = target_in_channels
        print(f"[channel_real] Modified patch_embedding: in_channels {z_dim} -> {target_in_channels}")
    
    vae = getattr(pipe, "vae", None)
    _modify_patch_embedding(getattr(pipe, "dit", None), vae)
    _modify_patch_embedding(getattr(pipe, "dit2", None), vae)


# -----------------------------
# Video IO + inference
# -----------------------------

def read_video_frames(video_path: str, max_frames: int, width: int | None, height: int | None) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[Image.Image] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        if width is not None and height is not None:
            img = img.resize((width, height), Image.BICUBIC)
        frames.append(img)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    if max_frames is not None and len(frames) > max_frames:
        # IMPORTANT:
        # Do NOT uniformly resample long videos down to max_frames, as that changes temporal speed
        # (appears "accelerated") and increases per-step motion, often breaking faces.
        # Instead, keep a contiguous window and truncate.
        frames = frames[:max_frames]
    return frames


def read_image(path: str, width: int | None, height: int | None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if width is not None and height is not None:
        img = img.resize((width, height), Image.BICUBIC)
    return img


def extract_first_frame_to_png(video_path: str, out_png: str) -> str:
    """
    Extract the first frame of a video as a PNG (RGB) using OpenCV.
    Returns the absolute path to the written PNG.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for first-frame extraction: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    # cv2.imwrite expects BGR; `frame` is BGR already
    ok = cv2.imwrite(out_png, frame)
    if not ok:
        raise RuntimeError(f"Failed to write png: {out_png}")
    return os.path.abspath(out_png)


@torch.no_grad()
def tv2v_infer(
    pipe: WanVideoPipeline,
    video_path: str,
    instruction: str,
    output_path: str,
    height: int,
    width: int,
    max_frames: int,
    denoising_strength: float,
    cfg_scale: float,
    num_inference_steps: int,
    seed: int | None,
    reference_video_path: str | None = None,
    longcat_image_path: str | None = None,
    reference_concat_method: str = "channel_real",
):
    """
    TV2V / Propagation inference.
    
    For propagation (CORRECTED logic matching training):
    INPUTS:
      - video_path (video1): SOURCE video (provides structure/motion)
      - longcat_image_path (video2's first frame): TARGET style (the edited appearance to propagate)
      - instruction: editing prompt
    OUTPUT:
      - video2: generated edited video with propagated style
    
    reference_concat_method controls how the condition is fused:
      - "channel": prepend video2's first frame along time dimension
      - "token": patchify video2's first frame via dit.ref_conv
      - "hybrid": conditional_image (video2's 1st frame) → token, reference_video (video1's 1st frame) → channel
      - "channel_real": channel_real uses video1 (full) as temporal structure (channel concat) + video2[0] as token style (conditional_image)
    """
    # video1 = input_video (provides structure/motion for generation)
    input_frames = read_video_frames(video_path, max_frames=max_frames, width=width, height=height)
    
    # Build reference_video and conditional_image based on concat method:
    # - video2's first frame (longcat_image_path) = target style to propagate
    # - video1 (input_frames) = structure/motion reference
    conditional_image = None
    reference_frames = None
    longcat_video = None
    
    if longcat_image_path is not None:
        # Propagation: video2's first frame is the target style
        style_img = read_image(longcat_image_path, width=width, height=height)
        longcat_video = [style_img]
        
        if reference_concat_method == "hybrid":
            # Hybrid mode (matching training):
            # - conditional_image (video2's first frame) → token concat (target style)
            # - reference_video (video1's first frame) → channel concat (source structure)
            conditional_image = [style_img]
            reference_frames = [input_frames[0]] if input_frames else None  # video1's first frame
        elif reference_concat_method == "channel_real":
            # Channel-real mode (matching training request):
            # - reference_video = video1 (FULL) -> temporal structure via channel_real concat
            # - conditional_image = video2's first frame -> token-concat global style via ref_conv
            reference_frames = input_frames  # full video1
            conditional_image = [style_img]
        else:
            # "channel" or "token" mode:
            # - use video2's first frame as the condition
            conditional_image = [style_img]
            reference_frames = [style_img]
    elif reference_video_path is not None:
        reference_frames = read_video_frames(reference_video_path, max_frames=max_frames, width=width, height=height)
        conditional_image = [reference_frames[0]] if reference_frames else None
    else:
        # No explicit condition - use video1's first frame
        reference_frames = [input_frames[0]] if input_frames else None
        conditional_image = [input_frames[0]] if input_frames else None
    
    out_frames = pipe(
        prompt=instruction,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        input_video=input_frames,  # video1: provides structure/motion
        reference_video=reference_frames,  # condition for channel concat
        conditional_image=conditional_image,  # condition for token concat (if applicable)
        reference_concat_method=reference_concat_method,
        longcat_video=longcat_video,
        denoising_strength=denoising_strength,
        height=height,
        width=width,
        num_frames=len(input_frames),
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        tiled=True,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_video(out_frames, output_path, fps=15, quality=5)
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Pure TV2V/Propagation inference (no Qwen/connector)")
    p.add_argument("--model_dir", type=str, required=True, help="Wan model directory (local).")
    p.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer folder path (optional).")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional .safetensors checkpoint to load (dit only).")
    # Single-item mode
    p.add_argument("--task_type", type=str, default="tv2v", choices=["tv2v", "prop"], help="Task type for single mode.")
    p.add_argument("--video_path", type=str, default=None, help="Input video path (single TV2V inference).")
    p.add_argument("--instruction", type=str, default=None, help="Editing instruction/prompt (single TV2V inference).")
    p.add_argument("--img2_path", type=str, default=None, help="Propagation conditional image path (single mode, task_type=prop).")
    p.add_argument("--output_path", type=str, default="outputs/tv2v_output.mp4", help="Output mp4 path (single mode).")
    # CSV batch mode
    p.add_argument("--csv_path", type=str, default=None, help="TV2V CSV path (batch inference).")
    p.add_argument("--data_root", type=str, default=None, help="Data root to join with relative paths in CSV.")
    p.add_argument("--output_dir", type=str, default="inference_outputs", help="Output dir (batch mode).")
    p.add_argument(
        "--cond_frames_dir",
        type=str,
        default=None,
        help="Where to write extracted conditional frames for prop CSVs that provide video2_path instead of img2_path. "
             "Default: <output_dir>/cond_frames",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional limit number of rows (batch mode).")
    # Reference concat method
    p.add_argument("--reference_concat_method", type=str, default="channel_real", 
                   choices=["channel", "token", "hybrid", "channel_real"],
                   help="How to fuse reference condition: 'channel' (time concat), 'token' (via ref_conv), 'hybrid' (both), 'channel_real' (true 48→96 channel concat)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--max_frames", type=int, default=81)
    p.add_argument("--denoising_strength", type=float, default=1.0)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    init_distributed()
    local_rank = get_local_rank()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    args = parse_args()
    if get_rank() == 0:
        print(f"[tv2v] device={device}")
        print(f"[tv2v] model_dir={args.model_dir}")
        print(f"[tv2v] reference_concat_method={args.reference_concat_method}")
        if args.csv_path:
            print(f"[tv2v] csv_path={args.csv_path}")
        else:
            print(f"[tv2v] video_path={args.video_path}")

    model_configs = build_model_configs_from_dir(args.model_dir)
    tokenizer_config = None if args.tokenizer_path is None else ModelConfig(path=args.tokenizer_path)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )

    # IMPORTANT: Create ref_conv / modify patch_embedding BEFORE loading checkpoint!
    # Otherwise the trained weights won't be loaded properly.
    ensure_ref_conv(pipe, args.reference_concat_method)
    ensure_channel_real_patch_embedding(pipe, args.reference_concat_method)

    if args.checkpoint:
        load_checkpoint_dit(pipe, args.checkpoint)

    # -----------------------------
    # Batch mode: CSV
    # -----------------------------
    if args.csv_path:
        os.makedirs(args.output_dir, exist_ok=True)
        cond_frames_dir = args.cond_frames_dir or os.path.join(args.output_dir, "cond_frames")
        os.makedirs(cond_frames_dir, exist_ok=True)

        with open(args.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if args.limit is not None:
            rows = rows[: args.limit]

        world_size = dist.get_world_size() if is_distributed() else 1
        rank = get_rank()

        # Each rank processes a strided subset
        my_indices = list(range(rank, len(rows), world_size))
        if rank == 0:
            print(f"[tv2v] total rows={len(rows)}, world_size={world_size}")

        for idx in my_indices:
            row = rows[idx]
            # Task type:
            # - Prefer explicit `task_type` / `task`
            # - If missing but a conditional frame is provided, infer propagation ("prop")
            task_type = (row.get("task_type") or row.get("task") or "").strip()

            video1 = row.get("video1_path") or row.get("video_path")
            # Prefer short_editing_instruction when available (ditto csv), then fall back.
            instruction = (
                row.get("short_editing_instruction")
                or row.get("editing_instruction")
                or row.get("instruction")
                or row.get("caption")
                or ""
            )
            video_name = row.get("video_name") or Path(video1 or f"row_{idx}").stem
            if not video1:
                if rank == 0:
                    print(f"[tv2v] skip row {idx}: missing video1_path")
                continue

            in_path = video1
            if not os.path.isabs(in_path):
                if not args.data_root:
                    raise ValueError(f"--data_root is required for relative paths (row {idx} video1_path={video1})")
                in_path = os.path.join(args.data_root, in_path)

            # Control effective generation length (like hunyuan_edit):
            # if CSV provides a `frames` column, use min(args.max_frames, frames) as max_frames.
            video_length = args.max_frames
            row_frames = row.get("frames")
            if row_frames is not None and str(row_frames).strip() != "":
                try:
                    video_length = int(float(row_frames))
                except Exception:
                    video_length = args.max_frames
                video_length = max(1, min(args.max_frames, video_length))

            # Propagation conditional frame: prefer img2_path if present (dataset prop format).
            img2_path = row.get("img2_path") or row.get("conditional_img_path") or row.get("cond_img_path")
            if img2_path and not os.path.isabs(img2_path):
                if not args.data_root:
                    raise ValueError(f"--data_root is required for relative paths (row {idx} img2_path={img2_path})")
                img2_path = os.path.join(args.data_root, img2_path)

            # If we are doing propagation but only video2_path is available (e.g. data_info_prop.csv),
            # auto-extract the first frame of video2 as the conditional image.
            if (task_type.lower() == "prop" or (not task_type and row.get("video2_path"))) and not img2_path:
                video2 = row.get("video2_path") or row.get("ground_truth_video")
                if not video2:
                    if rank == 0:
                        print(f"[prop] skip row {idx}: missing video2_path (and no img2_path)")
                    continue
                v2_path = video2
                if not os.path.isabs(v2_path):
                    if not args.data_root:
                        raise ValueError(f"--data_root is required for relative paths (row {idx} video2_path={video2})")
                    v2_path = os.path.join(args.data_root, v2_path)
                # Stable file name per row to avoid collisions under DDP.
                out_png = os.path.join(cond_frames_dir, f"{video_name}_row{idx:06d}_img2.png")
                if not os.path.exists(out_png):
                    try:
                        extract_first_frame_to_png(v2_path, out_png)
                    except Exception as e:
                        print(f"[prop][rank{rank}] failed to extract first frame row {idx} ({v2_path}): {e}")
                        continue
                img2_path = os.path.abspath(out_png)

            if not task_type:
                task_type = "prop" if img2_path else "tv2v"

            suffix = "prop" if task_type == "prop" else "tv2v"
            out_path = os.path.join(args.output_dir, f"{video_name}_{suffix}.mp4")

            try:
                tv2v_infer(
                    pipe=pipe,
                    video_path=in_path,
                    instruction=instruction,
                    output_path=out_path,
                    height=args.height,
                    width=args.width,
                    max_frames=video_length,
                    denoising_strength=args.denoising_strength,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    reference_video_path=in_path,
                    longcat_image_path=img2_path if task_type == "prop" else None,
                    reference_concat_method=args.reference_concat_method,
                )
                print(f"[{suffix}][rank{rank}] saved: {out_path}")
            except Exception as e:
                print(f"[{suffix}][rank{rank}] failed row {idx} ({in_path}): {e}")

        if is_distributed():
            dist.barrier()
    else:
        # -----------------------------
        # Single mode
        # -----------------------------
        if not args.video_path:
            raise ValueError("Single inference requires --video_path (or use --csv_path batch mode).")
        if args.task_type == "tv2v" and not args.instruction:
            raise ValueError("Single TV2V requires --instruction.")
        if args.task_type == "prop" and not args.img2_path:
            raise ValueError("Single propagation requires --img2_path (conditional frame).")
        out_path = tv2v_infer(
            pipe=pipe,
            video_path=args.video_path,
            instruction=args.instruction or "",
            output_path=args.output_path,
            height=args.height,
            width=args.width,
            max_frames=args.max_frames,
            denoising_strength=args.denoising_strength,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            reference_video_path=args.video_path,
            longcat_image_path=args.img2_path if args.task_type == "prop" else None,
            reference_concat_method=args.reference_concat_method,
        )
        if get_rank() == 0:
            print(f"[{args.task_type}] saved: {out_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()



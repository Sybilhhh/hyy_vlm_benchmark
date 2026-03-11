#!/usr/bin/env python3
"""
Unified training script for Kandinsky-5 (t2v / t2i / ti2i / tv2v / recon_v / recon_i).

Uses Accelerate for distributed training with online feature extraction:
- DiT (trainable) with instruct_type='channel' (33-ch input)
- Qwen2.5-VL + CLIP text encoder (frozen)
- HunyuanVideo 3D VAE for video, FLUX 2D VAE for images (frozen)
- Flow matching loss

Reconstruction tasks (recon_v, recon_i) are derived from t2v/t2i samples at
dataset load time (controlled by --reconstruction_ratio). They use
source=target with an identity instruction, training the model to preserve
visual content when given a conditioning signal matching the target.

Usage:
    accelerate launch train_unified.py \
        --conf_path configs/k5_unified.yaml \
        --csv_path data/train.csv \
        --output_dir outputs/unified_run
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from kandinsky.data import MultiResDataset, MultiResMultiTaskSamplerDistributed
from kandinsky.generation_utils import encode_video, get_task_mask
from kandinsky.models.dit import get_dit
from kandinsky.models.text_embedders import get_text_embedder
from kandinsky.models.vae import build_vae

logger = get_logger(__name__)

# Env-controlled debug for prop/keyframe training flow
K5_DEBUG_TRAIN_PROP_KEYFRAME = os.environ.get("K5_DEBUG_TRAIN_PROP_KEYFRAME", "0") == "1"
_K5_DEBUG_TRAIN_PROP_KEYFRAME_DONE = False


def _log_shape(name, t, out, prefix="[DEBUG-FLOW]"):
    """Log tensor/array shape for flow diagram."""
    if t is None:
        out(f"{prefix}   {name}: None")
    elif hasattr(t, "shape"):
        out(f"{prefix}   {name}: shape={tuple(t.shape)} dtype={getattr(t, 'dtype', '?')}")
    elif isinstance(t, (int, float)):
        out(f"{prefix}   {name}: {t} (scalar)")
    else:
        out(f"{prefix}   {name}: {type(t).__name__}")


def _log_debug_flow_step0(batch, task, source_video, target_video, target_latent, source_latent,
                           text_embeds, text_cu_seqlens, task_mask_vec, keyframe_idx,
                           noisy_input, target_velocity, timestep, dit_input, dit_timestep,
                           pred_velocity, conf, visual_rope_pos=None, text_rope_pos=None,
                           print_fn=None):
    """一次性输出完整数据流中各阶段 tensor size，供绘制流程图。仅 main 进程调用。"""
    out = print_fn if print_fn is not None else (lambda s: logger.info("%s", s))
    pt = conf.model.dit_params.patch_size
    B, T, H, W, C = target_latent.shape
    out("[DEBUG-FLOW] ========== 数据流 Tensor Size 流程图 ==========")
    out(f"[DEBUG-FLOW] task={task} keyframe_idx={keyframe_idx}")
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [1] Dataset __getitem__")
    instruction = batch.get("instruction", "")
    out(f"[DEBUG-FLOW]   instruction: str len={len(instruction)}")
    for k in ("video1", "video2", "img1", "img2", "video", "image"):
        if k in batch and batch[k] is not None:
            _log_shape(f"batch[{k}]", batch[k], out)
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [2] 训练循环 unsqueeze(0)")
    _log_shape("source_video", source_video, out)
    _log_shape("target_video", target_video, out)
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [3] VAE encode (HunyuanVideo 3D)")
    _log_shape("target_latent (B,T,H,W,C)", target_latent, out)
    _log_shape("source_latent", source_latent, out)
    out("[DEBUG-FLOW]       H,W 为 latent 空间 = pixel/8")
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [4] Text embedder (Qwen+CLIP)")
    for k, v in text_embeds.items():
        _log_shape(f"text_embeds['{k}']", v, out)
    out(f"[DEBUG-FLOW]       text_cu_seqlens={text_cu_seqlens} (有效token数)")
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [5] Task mask")
    _log_shape("task_mask_vec (T,)", task_mask_vec, out)
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [6] build_cond_input_for_training")
    _log_shape("noisy_input [noise(16)|cond(16)|mask(1)]=33ch", noisy_input, out)
    _log_shape("target_velocity", target_velocity, out)
    _log_shape("timestep (B,)", timestep, out)
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [7] 送入 DiT 前")
    _log_shape("dit_input (squeeze batch)", dit_input, out)
    _log_shape("dit_timestep", dit_timestep, out)
    if visual_rope_pos is not None:
        for i, v in enumerate(visual_rope_pos):
            _log_shape(f"visual_rope_pos[{i}] (T/H_p/W_p)", v, out)
    if text_rope_pos is not None:
        _log_shape("text_rope_pos (seq_len)", text_rope_pos, out)
    out(f"[DEBUG-FLOW]       patch_size={pt} -> 视觉token: T/{pt[0]} x H/{pt[1]} x W/{pt[2]}")
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [8] DiT 输出")
    _log_shape("pred_velocity", pred_velocity, out)
    out("[DEBUG-FLOW]")
    out("[DEBUG-FLOW] [9] Loss: MSE(pred_velocity, target_velocity)")
    out("[DEBUG-FLOW] =============================================")


def parse_args():
    parser = argparse.ArgumentParser(description="Kandinsky-5 unified training")

    parser.add_argument("--conf_path", type=str, required=True, help="Path to unified YAML config")
    parser.add_argument("--dit_checkpoint", type=str, default=None,
                        help="Override DiT checkpoint path from config")

    parser.add_argument("--csv_path", type=str, required=True, nargs="+",
                        help="Path(s) to training CSV files")
    parser.add_argument("--data_root", type=str, default=None, nargs="+",
                        help="Root directory for data paths in CSV")
    parser.add_argument("--max_frames", type=int, default=31,
                        help="Max video frames (default: 31 = 5s at 24fps/4)")

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--batch_config", type=str, default=None,
                        help="Path to JSON file with per-task batch sizes "
                             "(keys: tv2v, ti2i, t2v, t2i, recon_v, recon_i). "
                             "Overrides --train_batch_size when provided.")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Fallback batch size when --batch_config is not provided")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                        choices=["linear", "cosine", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    parser.add_argument("--reconstruction_ratio", type=float, default=0.05,
                        help="Fraction of t2v/t2i samples duplicated as recon_v/recon_i "
                             "in the dataset (source=target, identity instruction). "
                             "These are batched and trained as first-class tasks. Default: 0.05")
    parser.add_argument("--prop_prob", type=float, default=0.8,
                        help="When prop and guide_latents_num>=1: probability to pin first frame (noise) and zero cond at frame 0.")
    parser.add_argument("--empty_v_prob", type=float, default=0.2,
                        help="Hunyuan-style: when prop, probability to zero condition from frame guide_latents_num onward (or entire cond if guide_latents_num=0).")
    parser.add_argument("--guide_latents_num_1_prob", type=float, default=0.8,
                        help="Hunyuan-style: for prop, P(guide_latents_num=1). Default 0.4.")
    parser.add_argument("--quantized_qwen", action="store_true",
                        help="Use NF4 quantized Qwen text encoder")
    parser.add_argument("--text_token_padding", action="store_true")
    parser.add_argument("--allow_channel_expansion", action="store_true",
                        help="Allow loading a non-channel-expanded ckpt (e.g. T2V visual_cond=True "
                             "or T2I 16ch) into the 33ch unified DiT by zero-padding "
                             "the visual_embeddings.in_layer weight")
    parser.add_argument("--keep_models_on_gpu", action="store_true",
                        help="Keep VAE and text encoder on GPU instead of offloading to CPU "
                             "each step. Faster but uses ~20GB more VRAM. "
                             "Recommended for 80GB GPUs.")

    parser.add_argument("--debug_data", action="store_true",
                        help="Enable debug mode: log tensor shapes at each stage of the data pipeline "
                             "(dataset -> VAE -> cond input -> DiT). Default: off.")

    parser.add_argument("--benchmark_every_n_steps", type=int, default=None,
                        help="Run benchmark inference every N steps. None = no benchmarking.")
    parser.add_argument("--benchmark_csv", type=str, nargs="+", default=None,
                        help="Benchmark CSV path(s) for periodic inference")
    parser.add_argument("--benchmark_data_root", type=str, nargs="+", default=None,
                        help="Data root per benchmark CSV (use '' for none)")
    parser.add_argument("--benchmark_max_samples", type=int, default=None,
                        help="Max samples per benchmark (for quick testing)")
    parser.add_argument("--benchmark_num_steps", type=int, default=50,
                        help="Denoising steps for benchmark inference")

    args = parser.parse_args()
    if args.logging_dir is None:
        args.logging_dir = "logs"
    return args


def load_dit_with_channel_expansion(dit, state_dict, logger):
    """Load a DiT state_dict, handling mismatched visual_embeddings.in_layer shapes.

    The unified DiT uses instruct_type='channel' which gives:
        visual_embed_dim = 2 * 16 + 1 = 33
        VisualEmbeddings.in_layer: Linear(33 * prod(patch_size), model_dim)
                                   = Linear(132, model_dim)

    Possible source checkpoints:
    1. visual_cond=True (T2V/I2V):  visual_embed_dim=33  -> in_layer(132, model_dim)  [EXACT MATCH]
    2. instruct_type='channel' (I2I): visual_embed_dim=33  -> in_layer(132, model_dim)  [EXACT MATCH]
    3. No conditioning (T2I):          visual_embed_dim=16  -> in_layer(64, model_dim)   [NEEDS EXPANSION]

    For case 3, we zero-pad the weight from (model_dim, 64) to (model_dim, 132).
    The first 64 cols map to the noise channels; the extra 68 cols (cond + mask) start at zero.
    """
    target_key = "visual_embeddings.in_layer.weight"
    target_bias_key = "visual_embeddings.in_layer.bias"

    target_w = dit.state_dict()[target_key]
    source_w = state_dict.get(target_key)

    if source_w is None:
        logger.warning(
            f"Checkpoint missing key '{target_key}'; visual_embeddings.in_layer "
            "will remain randomly initialized."
        )
    elif source_w.shape != target_w.shape:
        model_dim, target_in = target_w.shape
        _, source_in = source_w.shape

        logger.info(
            f"Channel expansion: visual_embeddings.in_layer.weight "
            f"{source_w.shape} -> {target_w.shape}"
        )

        expanded_w = torch.zeros(model_dim, target_in, dtype=source_w.dtype)
        expanded_w[:, :source_in] = source_w
        state_dict[target_key] = expanded_w

        if target_bias_key in state_dict and target_bias_key in dit.state_dict():
            src_b = state_dict[target_bias_key]
            tgt_b = dit.state_dict()[target_bias_key]
            if src_b.shape != tgt_b.shape:
                expanded_b = torch.zeros_like(tgt_b)
                expanded_b[:src_b.shape[0]] = src_b
                state_dict[target_bias_key] = expanded_b

    missing, unexpected = dit.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        logger.warning(f"Missing keys after load: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys after load: {unexpected}")

    return dit


def encode_batch_video(video, vae):
    """Encode a (B, C, T, H, W) video/image tensor to latents via HunyuanVideo 3D VAE.
    For images, T=1."""
    with torch.no_grad():
        latent = encode_video(video.to(dtype=vae.dtype), vae)  # (B, T, H, W, C)
    return latent


def build_cond_input_for_training(
    task_type,
    source_latent,
    target_latent,
    noise,
    task_mask_vec,
    guide_latents_num=0,
    empty_v_prob=0.2,
    keyframe_idx=None,
):
    """Build the 33-channel noisy input for training.

    For editing tasks (ti2i, tv2v, prop): source_latent is the VAE-encoded source.
    For prop with guide_latents_num > 0 (Hunyuan-style): condition's first N frames
    are overwritten by target's first N frames; mask is 1 for first N, 0 for rest;
    with empty_v_prob the condition from frame N onward is zeroed.
    For reconstruction tasks (recon_v, recon_i): source_latent == target_latent, mask=1.
    For generation tasks (t2v, t2i): source_latent is zeros, mask=0.

    Returns:
        noisy_input: (B, T, H, W, 33) = [noisy_target(16) | cond(16) | mask(1)]
        target_velocity: (B, T, H, W, 16) = noise - target_latent
    """
    B, T, H, W, C = target_latent.shape

    if source_latent is None:
        source_latent = torch.zeros_like(target_latent)

    cond_latent = source_latent
    if task_type == "keyframe" and keyframe_idx is not None:
        cond_latent = source_latent.clone()
        idx = max(0, min(int(keyframe_idx), T - 1))
        if empty_v_prob > 0 and random.random() < empty_v_prob:
            cond_latent = torch.zeros_like(cond_latent)#done
        cond_latent[:, idx : idx + 1, :, :, :] = target_latent[:, idx : idx + 1, :, :, :].to(cond_latent.dtype)
    elif task_type == "prop" and guide_latents_num > 0:
        cond_latent = source_latent.clone()
        n_guide = min(guide_latents_num, T)
        if empty_v_prob > 0 and random.random() < empty_v_prob:
            cond_latent[:, n_guide:, :, :, :] = 0.0  #done: v2 caption
        cond_latent[:, :n_guide, :, :, :] = target_latent[:, :n_guide, :, :, :].to(cond_latent.dtype)
    elif task_type == "prop" and guide_latents_num == 0 and empty_v_prob > 0 and random.random() < empty_v_prob:
        cond_latent = torch.zeros_like(source_latent)

    device = target_latent.device
    dtype = target_latent.dtype
    mask_ones = torch.ones(B, T, H, W, 1, device=device, dtype=dtype)
    mask_zeros = torch.zeros(B, T, H, W, 1, device=device, dtype=dtype)
    # Vectorized: expand task_mask (T,) to (B, T, H, W, 1) and select 0/1 per frame.
    task_mask_vec = task_mask_vec.to(device=device)
    task_mask_expanded = task_mask_vec.view(1, T, 1, 1, 1).expand(B, T, H, W, 1)
    mask_channel = torch.where(task_mask_expanded > 0.5, mask_ones, mask_zeros)

    timestep = torch.rand(B, device=target_latent.device)
    sigma = timestep.view(B, 1, 1, 1, 1)
    noisy_target = (1 - sigma) * target_latent + sigma * noise

    noisy_input = torch.cat([noisy_target, cond_latent, mask_channel], dim=-1)

    target_velocity = noise - target_latent

    return noisy_input, target_velocity, timestep


def run_benchmarks(args, dit, vae, text_embedder, conf, accelerator,
                   global_step, output_dir):
    """Run benchmark inference reusing the already-loaded training models.

    Constructs a Kandinsky5UnifiedPipeline from the in-memory DiT, VAE, and
    text_embedder -- no checkpoint reload needed. Each rank processes a
    disjoint shard of CSV rows for data-parallel inference.

    The DiT is temporarily switched to eval mode and back to train mode after.
    """
    if not args.benchmark_csv:
        return

    from infer_unified import run_csv_rows_with_pipeline
    from kandinsky.unified_pipeline import Kandinsky5UnifiedPipeline

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device

    dit_unwrapped = accelerator.unwrap_model(dit)
    dit_unwrapped.eval()

    vae.to(device)
    text_embedder.to(device)

    device_map = {"dit": device, "vae": device, "text_embedder": device}
    pipeline = Kandinsky5UnifiedPipeline(
        device_map=device_map,
        dit=dit_unwrapped,
        text_embedder=text_embedder,
        vae=vae,
        local_dit_rank=0,
        world_size=1,
        conf=conf,
        offload=False,
    )

    data_roots = args.benchmark_data_root or [""] * len(args.benchmark_csv)

    for csv_path, data_root in zip(args.benchmark_csv, data_roots):
        bench_name = os.path.basename(os.path.dirname(csv_path))
        bench_output = os.path.join(
            output_dir, "benchmarks", f"step-{global_step}", bench_name,
        )

        logger.info(
            f"[Benchmark] step={global_step} bench={bench_name} "
            f"rank={rank}/{world_size}"
        )

        try:
            run_csv_rows_with_pipeline(
                pipeline=pipeline,
                csv_path=csv_path,
                output_dir=bench_output,
                data_root=data_root or None,
                num_steps=args.benchmark_num_steps,
                seed=args.seed,
                max_samples=args.benchmark_max_samples,
                rank=rank,
                world_size=world_size,
            )
            logger.info(f"[Benchmark] Done: {bench_output}")
        except Exception as e:
            logger.warning(f"[Benchmark] Failed: {bench_name}: {e}")

    del pipeline

    if not args.keep_models_on_gpu:
        vae.to("cpu")
        text_embedder.to("cpu")
    torch.cuda.empty_cache()

    dit_unwrapped.train()


def main():
    global logger
    ## ----------------- load args and accelerator -----------------
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_config,
    )
    accelerator.even_batches = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        ))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    else:
        # Non-main processes: suppress all logging
        logging.basicConfig(
            level=logging.ERROR,  # Only show errors from non-main processes
            handlers=[logging.NullHandler()],
            force=True
        )
        logger = get_logger(__name__, log_level="ERROR")

    # ---- Load config ----
    conf = OmegaConf.load(args.conf_path)
    if args.dit_checkpoint is not None:
        conf.model.checkpoint_path = args.dit_checkpoint

    # Debug data pipeline: config yaml or --debug_data flag (flag overrides)
    debug_data = getattr(conf, "debug_data", False) or args.debug_data
    if debug_data:
        logger.info("[DEBUG] Data pipeline debug enabled — will log shapes at each stage")

    # ---- Load models ----
    # Stagger model loading across local ranks to avoid filesystem stampede.
    # Only one GPU per node loads at a time; others wait their turn.
    local_rank = accelerator.local_process_index
    num_local = torch.cuda.device_count()
    for loading_rank in range(num_local):
        if local_rank == loading_rank:
            logger.info(f"Loading models (local_rank={local_rank})...")

            vae = build_vae(conf.model.vae)
            vae.requires_grad_(False)
            vae.eval()
            vae.to(dtype=torch.float16)
            # ------------ load keyframe model ------------

            text_embedder = get_text_embedder( #todo
                conf.model.text_embedder, device="cpu",
                quantized_qwen=args.quantized_qwen,
                text_token_padding=args.text_token_padding,
            )

            dit = get_dit(conf.model.dit_params, text_token_padding=args.text_token_padding)
            if os.path.exists(conf.model.checkpoint_path):
                state_dict = load_file(conf.model.checkpoint_path, device="cpu")
                if args.allow_channel_expansion:
                    dit = load_dit_with_channel_expansion(dit, state_dict, logger)
                else:
                    dit.load_state_dict(state_dict, assign=True)
                logger.info(f"Loaded DiT from {conf.model.checkpoint_path}")
            else:
                logger.warning(f"No checkpoint at {conf.model.checkpoint_path}, training from scratch")

        torch.distributed.barrier()
    logger.info(f"All local ranks finished loading models.")

    device = accelerator.device
    offload = not args.keep_models_on_gpu
    if not offload:
        vae.to(device)
        text_embedder.to(device)
        logger.info("Keeping VAE and text encoder on GPU (--keep_models_on_gpu)")

    dit.train()
    dit.to(dtype=torch.bfloat16)

    if args.gradient_checkpointing:
        dit.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # ---- Dataset ----
    logger.info(f"Loading dataset from {args.csv_path}")
    dataset = MultiResDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        max_frames=args.max_frames,
        reconstruction_ratio=args.reconstruction_ratio,
    )

    if args.batch_config is not None:
        if not os.path.isfile(args.batch_config):
            raise FileNotFoundError(
                f"Batch config file not found: {args.batch_config}"
            )
        with open(args.batch_config) as f:
            batch_sizes = json.load(f)
        logger.info(f"Loaded per-task batch sizes from {args.batch_config}: {batch_sizes}")
    else:
        batch_sizes = {
            "tv2v": args.train_batch_size,
            "ti2i": args.train_batch_size,
            "t2v": args.train_batch_size,
            "t2i": args.train_batch_size,
            "recon_v": args.train_batch_size,
            "recon_i": args.train_batch_size,
            "prop": args.train_batch_size,
            "keyframe": args.train_batch_size,
        }

    sampler = MultiResMultiTaskSamplerDistributed(
        dataset,
        video_batch_size=batch_sizes.get("tv2v", args.train_batch_size),
        image_batch_size=batch_sizes.get("ti2i", args.train_batch_size),
        gen_video_batch_size=batch_sizes.get("t2v", args.train_batch_size),
        gen_image_batch_size=batch_sizes.get("t2i", args.train_batch_size),
        recon_video_batch_size=batch_sizes.get("recon_v", batch_sizes.get("t2v", args.train_batch_size)),
        recon_image_batch_size=batch_sizes.get("recon_i", batch_sizes.get("t2i", args.train_batch_size)),
        prop_batch_size=batch_sizes.get("prop", args.train_batch_size),
        keyframe_batch_size=batch_sizes.get("keyframe", args.train_batch_size),  # TODO: add prop,keyframe batch size
        shuffle=True,
        seed=args.seed,
    )

    def collate_fn(examples):
        # Training loop currently expects a single-sample batch (no stacking).
        # When sampler returns multiple indices, we use only the first to avoid
        # shape mismatch; consider implementing proper batching for efficiency.
        return examples[0]

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps * accelerator.num_processes
            if args.max_train_steps else None
        ),
    )

    accelerator.wait_for_everyone()
    logger.info("All ranks ready, preparing model / optimizer / scheduler.")
    dit, optimizer, lr_scheduler = accelerator.prepare(
        dit, optimizer, lr_scheduler,
    )

    # ---- Training setup ----
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tb_config = {
            k: ",".join(v) if isinstance(v, list) else v
            for k, v in vars(args).items() if v is not None
        }
        accelerator.init_trackers("kandinsky5-unified", config=tb_config)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # Checkpoint dir is under output_dir only (data_root can be a list).
            output_dir = args.output_dir
            dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(output_dir, dirs[-1]) if dirs else None
        else:
            path = args.resume_from_checkpoint

        if path is not None:
            accelerator.print(f"Resuming from {path}")
            accelerator.load_state(path)
            global_step = int(path.split("-")[-1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dit):
                task = batch["task"]
                instruction = batch.get("instruction", "")

                if task in ("tv2v", "recon_v", "prop", "i2v", "keyframe"):
                    source_video = batch["video1"].unsqueeze(0).to(accelerator.device)
                    target_video = batch["video2"].unsqueeze(0).to(accelerator.device)
                elif task in ("ti2i", "recon_i"):
                    source_video = batch["img1"].unsqueeze(0).to(accelerator.device)
                    target_video = batch["img2"].unsqueeze(0).to(accelerator.device)
                elif task == "t2v":
                    source_video = None
                    target_video = batch["video"].unsqueeze(0).to(accelerator.device)
                elif task == "t2i":
                    source_video = None
                    target_video = batch["image"].unsqueeze(0).to(accelerator.device)
                else:
                    raise ValueError(
                        f"Unknown task '{task}' at step {step}. "
                        f"Cannot skip inside accumulate() — would hang other ranks."
                    )

                is_reconstruction = task in ("recon_v", "recon_i")



                if offload:
                    vae.to(accelerator.device)
                with torch.no_grad():
                    target_latent = encode_batch_video(target_video, vae)
                    if is_reconstruction:
                        source_latent = target_latent
                    elif source_video is not None:
                        source_latent = encode_batch_video(source_video, vae)
                    else:
                        source_latent = None
                if offload:
                    vae.to("cpu")
                    torch.cuda.empty_cache()

                B, T, H, W, C = target_latent.shape


                if offload:
                    text_embedder.to(accelerator.device)
                if task in ("ti2i", "recon_i"):
                    type_of_content = "image_edit"
                elif task in ("tv2v", "recon_v", "i2v"):
                    type_of_content = "video"
                elif task in ["keyframe","prop"]:
                    type_of_content = "keyframe"
                elif T == 1:
                    type_of_content = "image"
                else:
                    type_of_content = "video"

                                # Keyframe propagation: randomly pick a single guide frame; mask=1 only at that frame
                keyframe_idx = None
                if task == "keyframe":
                    keyframe_idx = random.randint(0, max(T - 1, 0))
                elif task == "prop":
                    keyframe_idx = 0
                source_image = source_video[:, keyframe_idx:keyframe_idx+1, :, :, :]
                target_image = target_video[:, keyframe_idx:keyframe_idx+1, :, :, :]
                images_for_qwen = [
                    source_image,
                    target_image,
                ]
                with torch.no_grad():
                    text_embeds, text_cu_seqlens, attention_mask = text_embedder.encode( #TODO
                        [instruction], type_of_content=type_of_content,
                        images=images_for_qwen,
                    )

                if offload:
                    text_embedder.to("cpu")
                    torch.cuda.empty_cache()

                for key in text_embeds:
                    text_embeds[key] = text_embeds[key].to(
                        device=accelerator.device, dtype=torch.bfloat16,
                    )
                text_cu_seqlens = text_cu_seqlens.to(device=accelerator.device)[-1].item()

                # Prop (Hunyuan-style): sample guide_latents_num in {0, 1, 4}; condition first N frames = target, mask 1 for first N
                guide_latents_num = 0
                if task in ["prop", "keyframe"]:
                    p1 = getattr(args, "guide_latents_num_1_prob", 0.8)
                    guide_p = random.random()
                    if guide_p < p1:
                        guide_latents_num = 1
                    else:
                        guide_latents_num = 0

                task_mask_vec = get_task_mask(
                    task,
                    T,
                    guide_latents_num=guide_latents_num,
                    keyframe_idx=keyframe_idx
                )

                noise = torch.randn_like(target_latent)
                noisy_input, target_velocity, timestep = build_cond_input_for_training(
                    task,
                    source_latent,
                    target_latent,
                    noise,
                    task_mask_vec,
                    guide_latents_num=guide_latents_num if task in ["prop", "keyframe"] else 0,
                    empty_v_prob=args.empty_v_prob if task in ["prop", "keyframe"] else 0.0,
                    keyframe_idx=keyframe_idx if task == "keyframe" else None
                )

                visual_rope_pos = [
                    torch.arange(T),
                    torch.arange(H // conf.model.dit_params.patch_size[1]),
                    torch.arange(W // conf.model.dit_params.patch_size[2]),
                ]
                text_rope_pos = torch.arange(text_cu_seqlens)

                scale_factor = tuple(conf.metrics.scale_factor)

                dit_input = noisy_input.squeeze(0).to(dtype=torch.bfloat16)
                dit_timestep = (timestep * 1000).to(dtype=torch.bfloat16)

                pred_velocity = dit(
                    dit_input,
                    text_embeds["text_embeds"],
                    text_embeds["pooled_embed"],
                    dit_timestep,
                    visual_rope_pos,
                    text_rope_pos,
                    scale_factor=scale_factor,
                )

                # [DEBUG-FLOW] 首 step 一次性输出完整数据流 size（仅 main 进程）
                if debug_data and global_step == 0 and step == 0 and accelerator.is_main_process:
                    lines = []
                    def _collect(s):
                        lines.append(s)
                        accelerator.print(s)
                    _log_debug_flow_step0(
                        batch, task,
                        source_video, target_video,
                        target_latent, source_latent,
                        text_embeds, text_cu_seqlens,
                        task_mask_vec, keyframe_idx,
                        noisy_input, target_velocity, timestep,
                        dit_input, dit_timestep,
                        pred_velocity, conf,
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=text_rope_pos,
                        print_fn=_collect,
                    )
                    debug_flow_path = os.path.join(args.output_dir, "debug_data_flow.txt")
                    with open(debug_flow_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    accelerator.print(f"[DEBUG-FLOW] 已写入 {debug_flow_path}")

                # Env-controlled debug trace for first prop/keyframe sample in training
                global _K5_DEBUG_TRAIN_PROP_KEYFRAME_DONE
                if (
                    K5_DEBUG_TRAIN_PROP_KEYFRAME
                    and not _K5_DEBUG_TRAIN_PROP_KEYFRAME_DONE
                    and task in ("prop", "keyframe")
                    and accelerator.is_main_process
                ):
                    dbg_lines = []

                    def _dbg(s: str):
                        dbg_lines.append(s)
                        accelerator.print(s)

                    _dbg("[K5-DEBUG-TRAIN-PROP-KEYFRAME] ====== 数据流 Tensor Size (训练) ======")
                    _dbg(f"[K5-DEBUG-TRAIN-PROP-KEYFRAME] step={global_step} task={task} keyframe_idx={keyframe_idx}")
                    _dbg("[K5-DEBUG-TRAIN-PROP-KEYFRAME] batch keys: " + ", ".join(sorted(batch.keys())))

                    _log_debug_flow_step0(
                        batch, task,
                        source_video, target_video,
                        target_latent, source_latent,
                        text_embeds, text_cu_seqlens,
                        task_mask_vec, keyframe_idx,
                        noisy_input, target_velocity, timestep,
                        dit_input, dit_timestep,
                        pred_velocity, conf,
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=text_rope_pos,
                        print_fn=_dbg,
                    )

                    debug_path = os.path.join(args.output_dir, "debug_train_prop_keyframe.txt")
                    try:
                        with open(debug_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(dbg_lines))
                        accelerator.print(f"[K5-DEBUG-TRAIN-PROP-KEYFRAME] 已写入 {debug_path}")
                    except Exception as e:
                        accelerator.print(f"[K5-DEBUG-TRAIN-PROP-KEYFRAME] 写入失败: {e}")

                    _K5_DEBUG_TRAIN_PROP_KEYFRAME_DONE = True

                loss = F.mse_loss(
                    pred_velocity.float(),
                    target_velocity.squeeze(0).float(),
                    reduction="mean",
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg_loss = accelerator.gather(loss.detach()).mean().item()

                    tb_logs = {
                        "train/loss": avg_loss,
                        f"train/loss_{task}": avg_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "train/epoch": epoch,
                        "perf/vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                    }
                    accelerator.log(tb_logs, step=global_step)
                    torch.cuda.reset_peak_memory_stats()

                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}", task=task, lr=f"{tb_logs['train/lr']:.2e}",
                    )
                    if accelerator.is_main_process:
                        logger.info(
                            f"step={global_step} loss={avg_loss:.4f} "
                            f"grad_norm={tb_logs['train/grad_norm']:.3f} "
                            f"lr={tb_logs['train/lr']:.2e} task={task} "
                            f"vram={tb_logs['perf/vram_gb']:.1f}GB"
                        )

                if global_step % args.checkpointing_steps == 0:
                    ckpt_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}",
                    )
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        dit_unwrapped = accelerator.unwrap_model(dit)
                        dit_ckpt_path = os.path.join(ckpt_dir, "dit.safetensors")
                        save_file(dit_unwrapped.state_dict(), dit_ckpt_path)
                        logger.info(f"Saved checkpoint to {ckpt_dir}")

                    accelerator.wait_for_everyone()

                    if (
                        args.benchmark_every_n_steps is not None
                        and global_step % args.benchmark_every_n_steps == 0
                    ):
                        run_benchmarks(
                            args, dit, vae, text_embedder, conf,
                            accelerator, global_step, args.output_dir,
                        )
                        accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break

    # Save final
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        dit_unwrapped = accelerator.unwrap_model(dit)
        final_path = os.path.join(args.output_dir, "dit_final.safetensors")
        save_file(dit_unwrapped.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

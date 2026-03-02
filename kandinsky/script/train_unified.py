#!/usr/bin/env python3
"""
Unified training script for Kandinsky-5 (t2v / t2i / ti2i / tv2v / prop).

Uses Accelerate for distributed training with online feature extraction:
- DiT (trainable) with instruct_type='channel' (33-ch input)
- Qwen2.5-VL + CLIP text encoder (frozen)
- HunyuanVideo 3D VAE for video, FLUX 2D VAE for images (frozen)
- Flow matching loss

Usage:
    accelerate launch train_unified.py \
        --conf_path configs/k5_unified.yaml \
        --csv_path data/train.csv \
        --output_dir outputs/unified_run
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from kandinsky.data import MultiResDataset, MultiResMultiTaskSamplerDistributed
from kandinsky.generation_utils import encode_video, get_task_mask, merge_tensor_by_mask
from kandinsky.models.dit import get_dit
from kandinsky.models.text_embedders import get_text_embedder
from kandinsky.models.vae import build_vae

logger = get_logger(__name__)


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
    parser.add_argument("--train_batch_size", type=int, default=1)
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

    parser.add_argument("--quantized_qwen", action="store_true",
                        help="Use NF4 quantized Qwen text encoder")
    parser.add_argument("--text_token_padding", action="store_true")
    parser.add_argument("--allow_channel_expansion", action="store_true",
                        help="Allow loading a non-channel-expanded ckpt (e.g. T2V visual_cond=True "
                             "or T2I 16ch) into the 33ch unified DiT by zero-padding "
                             "the visual_embeddings.in_layer weight")

    parser.add_argument("--prop_prob", type=float, default=0.8,
                        help="When prop and guide_latents_num>=1: probability to pin first frame (noise) and zero cond at frame 0.")
    parser.add_argument("--empty_v_prob", type=float, default=0.4,
                        help="Hunyuan-style: when prop, probability to zero condition from frame guide_latents_num onward (or entire cond if guide_latents_num=0).")
    parser.add_argument("--guide_latents_num_1_prob", type=float, default=0.4,
                        help="Hunyuan-style: for prop, P(guide_latents_num=1). Default 0.4.")
    parser.add_argument("--guide_latents_num_4_prob", type=float, default=0.4,
                        help="Hunyuan-style: for prop, P(guide_latents_num=4). P(0)=1 - 1_prob - 4_prob.")

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
        args.logging_dir = os.path.join(args.output_dir, "logs")
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

    if source_w is not None and source_w.shape != target_w.shape:
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
    empty_v_prob=0.0,
):
    """Build the 33-channel noisy input for training.

    For editing tasks (ti2i, tv2v, prop): source_latent is the VAE-encoded source.
    For prop with guide_latents_num > 0 (Hunyuan-style): condition's first N frames
    are overwritten by target's first N frames; mask is 1 for first N, 0 for rest;
    with empty_v_prob the condition from frame N onward is zeroed.

    Returns:
        noisy_input: (B, T, H, W, 33) = [noisy_target(16) | cond(16) | mask(1)]
        target_velocity: (B, T, H, W, 16) = noise - target_latent
    """
    B, T, H, W, C = target_latent.shape

    if source_latent is None:
        source_latent = torch.zeros_like(target_latent)

    cond_latent = source_latent
    if task_type == "prop" and guide_latents_num > 0:
        cond_latent = source_latent.clone()
        n_guide = min(guide_latents_num, T)
        cond_latent[:, :n_guide, :, :, :] = target_latent[:, :n_guide, :, :, :].to(cond_latent.dtype)
        if empty_v_prob > 0 and random.random() < empty_v_prob:
            cond_latent[:, n_guide:, :, :, :] = 0.0
    elif task_type == "prop" and guide_latents_num == 0 and empty_v_prob > 0 and random.random() < empty_v_prob:
        cond_latent = torch.zeros_like(source_latent)

    mask_ones = torch.ones(B, T, H, W, 1, device=target_latent.device, dtype=target_latent.dtype)
    mask_zeros = torch.zeros(B, T, H, W, 1, device=target_latent.device, dtype=target_latent.dtype)

    mask_channel = torch.zeros(B, T, H, W, 1, device=target_latent.device, dtype=target_latent.dtype)
    for b in range(B):
        single_mask = merge_tensor_by_mask(
            mask_zeros[b], mask_ones[b], mask=task_mask_vec, dim=0,
        )
        mask_channel[b] = single_mask

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

    vae.to("cpu")
    text_embedder.to("cpu")
    torch.cuda.empty_cache()

    dit_unwrapped.train()


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
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

    # ---- Load config ----
    conf = OmegaConf.load(args.conf_path)
    if args.dit_checkpoint is not None:
        conf.model.checkpoint_path = args.dit_checkpoint

    # ---- Load models ----
    logger.info("Loading models...")

    vae = build_vae(conf.model.vae)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(dtype=torch.float16)

    text_embedder = get_text_embedder(
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

    dit.train()
    dit.to(dtype=torch.bfloat16)

    # Skip DiT gradient checkpointing when using bf16: it triggers CheckpointError with
    # torch.compile + non-reentrant checkpoint (recomputed tensor metadata mismatch).
    if args.gradient_checkpointing:
        if args.mixed_precision == "bf16":
            logger.warning(
                "gradient_checkpointing is disabled for bf16 (incompatible with this DiT); "
                "training will use more VRAM."
            )
        else:
            dit.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # ---- Dataset ----
    logger.info(f"Loading dataset from {args.csv_path}")
    dataset = MultiResDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        max_frames=args.max_frames,
    )

    sampler = MultiResMultiTaskSamplerDistributed(
        dataset,
        video_batch_size=args.train_batch_size,
        image_batch_size=args.train_batch_size,
        gen_video_batch_size=args.train_batch_size,
        gen_image_batch_size=args.train_batch_size,
        shuffle=True,
        seed=args.seed,
    )

    def collate_fn(examples):
        if len(examples) == 1:
            return examples[0]
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

    dit, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        dit, optimizer, train_dataloader, lr_scheduler,
    )

    # ---- Training setup ----
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

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
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if dirs else None
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

                # Prepare source/target tensors (all encoded via single 3D VAE)
                if task in ("tv2v", "prop", "style_transfer"):
                    source_video = batch["video1"].unsqueeze(0).to(accelerator.device)
                    target_video = batch["video2"].unsqueeze(0).to(accelerator.device)
                elif task == "ti2i":
                    source_video = batch["img1"].unsqueeze(0).to(accelerator.device)
                    target_video = batch["img2"].unsqueeze(0).to(accelerator.device)
                elif task == "t2v":
                    source_video = None
                    target_video = batch["video"].unsqueeze(0).to(accelerator.device)
                elif task == "t2i":
                    source_video = None
                    target_video = batch["image"].unsqueeze(0).to(accelerator.device)
                else:
                    continue

                # VAE encode: move to GPU, encode, move back to CPU
                vae.to(accelerator.device)
                with torch.no_grad():
                    target_latent = encode_batch_video(target_video, vae)
                    if source_video is not None:
                        source_latent = encode_batch_video(source_video, vae)
                    else:
                        source_latent = None
                vae.to("cpu")
                torch.cuda.empty_cache()

                B, T, H, W, C = target_latent.shape

                # Text encode: move to GPU, encode, move back to CPU
                text_embedder.to(accelerator.device)
                if task == "ti2i":
                    type_of_content = "image_edit"
                    images_for_qwen = None
                elif T == 1:
                    type_of_content = "image"
                    images_for_qwen = None
                else:
                    type_of_content = "video"
                    images_for_qwen = None

                with torch.no_grad():
                    text_embeds, text_cu_seqlens, attention_mask = text_embedder.encode(
                        [instruction], type_of_content=type_of_content,
                        images=images_for_qwen,
                    )
                text_embedder.to("cpu")
                torch.cuda.empty_cache()

                for key in text_embeds:
                    text_embeds[key] = text_embeds[key].to(
                        device=accelerator.device, dtype=torch.bfloat16,
                    )
                text_cu_seqlens = text_cu_seqlens.to(device=accelerator.device)[-1].item()

                # Prop (Hunyuan-style): sample guide_latents_num in {0, 1, 4}; condition first N frames = target, mask 1 for first N
                guide_latents_num = 0
                if task == "prop":
                    p1 = getattr(args, "guide_latents_num_1_prob", 0.4)
                    p4 = getattr(args, "guide_latents_num_4_prob", 0.4)
                    guide_p = random.random()
                    if guide_p < p1:
                        guide_latents_num = 1
                    elif guide_p < p1 + p4:
                        guide_latents_num = 4
                    else:
                        guide_latents_num = 0

                task_mask_vec = get_task_mask(
                    task if task not in ("style_transfer", "prop") else "tv2v",
                    T,
                    guide_latents_num=guide_latents_num if task == "prop" else 0,
                )

                noise = torch.randn_like(target_latent)
                noisy_input, target_velocity, timestep = build_cond_input_for_training(
                    task,
                    source_latent,
                    target_latent,
                    noise,
                    task_mask_vec,
                    guide_latents_num=guide_latents_num if task == "prop" else 0,
                    empty_v_prob=args.empty_v_prob if task == "prop" else 0.0,
                )

                # Prop: pin first frame in noise channel and zero cond at first frame (match inference; Hunyuan-style when guide_latents_num >= 1)
                if task == "prop" and guide_latents_num >= 1 and random.random() < args.prop_prob:
                    noisy_input[:, 0, :, :, 0:16] = target_latent[:, 0, :, :, :].to(noisy_input.dtype)
                    noisy_input[:, 0, :, :, 16:32] = 0.0

                # RoPE positions
                visual_rope_pos = [
                    torch.arange(T),
                    torch.arange(H // conf.model.dit_params.patch_size[1]),
                    torch.arange(W // conf.model.dit_params.patch_size[2]),
                ]
                text_rope_pos = torch.arange(text_cu_seqlens)

                scale_factor = tuple(conf.metrics.scale_factor)

                # Forward through DiT
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_velocity = dit(
                        noisy_input.squeeze(0),
                        text_embeds["text_embeds"],
                        text_embeds["pooled_embed"],
                        timestep * 1000,
                        visual_rope_pos,
                        text_rope_pos,
                        scale_factor=scale_factor,
                    )

                # Flow matching loss: MSE between predicted and target velocity
                loss = F.mse_loss(
                    pred_velocity.float(),
                    target_velocity.squeeze(0).float(),
                    reduction="mean",
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.log_interval == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "task": task,
                        "step": global_step,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(
                        {k: v for k, v in logs.items() if k != "task"},
                        step=global_step,
                    )

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        ckpt_dir = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}",
                        )
                        accelerator.save_state(ckpt_dir)
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

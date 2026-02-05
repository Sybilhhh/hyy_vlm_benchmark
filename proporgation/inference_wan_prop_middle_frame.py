#!/usr/bin/env python
# coding: utf-8
"""
Inference script for WanEditPipeline
Supports bucketing strategy for different resolutions based on pixel count
Multi-node multi-GPU inference using Accelerate
Support prop by huziwen in 26.1
Support middle frame inference: generate frames before and after the middle frame
"""
import argparse
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object

from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.utils import export_to_video, load_video

from pipeline.pipeline_wan_edit import WanEditPipeline
from dataset.dataset_multires_online import load_image, load_video_frames
from train_wan_prop import process_vae_latent, _modify_patch_embedding_for_channel_concat
import decord
from torchvision import transforms

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Resolution bucket groups based on pixel count
BUCKET_GROUPS = [
    # Group 1: ~400k pixels
    [(480, 848), (544, 720), (640, 640), (720, 544), (848, 480)],
    # Group 2: ~920k pixels
    [(720, 1280), (832, 1104), (960, 960), (1104, 832), (1280, 720)],
    # Group 3: ~1M pixels
    [(768, 1360), (880, 1184), (1024, 1024), (1184, 880), (1360, 768)],
    # Group 4: ~2M pixels
    # [(1088, 1920), (1248, 1664), (1440, 1440), (1664, 1248), (1920, 1088)]
]

# Standard frame counts
STANDARD_FRAME_COUNTS = [49, 81, 121]


def get_nearest_4k_plus_1(num_frames: int) -> int:
    """
    Get the nearest frame count that follows the 4k+1 pattern.
    """
    k = round((num_frames - 1) / 4)
    k = max(0, k)
    return 4 * k + 1


def get_bucket(h: int, w: int) -> Tuple[int, int]:
    """
    Find the best resolution bucket based on pixel count and aspect ratio.
    """
    if h == 0 or w == 0:
        logger.warning(f"Invalid dimensions: h={h}, w={w} (zero dimension)")
        return (480, 848)
    
    try:
        input_pixels = h * w
        input_ratio = w / h
        
        best_group = BUCKET_GROUPS[0]
        for group in BUCKET_GROUPS:
            rep_h, rep_w = group[2]
            if rep_h * rep_w <= input_pixels:
                best_group = group
            else:
                break
        
        best_bucket = None
        min_ratio_diff = float('inf')
        
        for bucket_h, bucket_w in best_group:
            bucket_ratio = bucket_w / bucket_h
            diff = abs(input_ratio - bucket_ratio)
            if diff < min_ratio_diff:
                min_ratio_diff = diff
                best_bucket = (bucket_h, bucket_w)
        
        return best_bucket
    
    except Exception as e:
        logger.error(f"Error in get_bucket for h={h}, w={w}: {e}")
        return (480, 848)


def get_target_frames(num_frames: int, frame_mode: str = "auto") -> int:
    """
    Get the target frame count based on the mode.
    """
    if frame_mode == "49":
        return 49
    elif frame_mode == "81":
        return 81
    elif frame_mode == "auto":
        if abs(num_frames - 49) <= abs(num_frames - 81):
            return 49
        elif num_frames < 121:
            return 81
        else:
            return 81
    elif frame_mode == "nearest":
        target = get_nearest_4k_plus_1(num_frames)
        target = max(1, min(target, 121))
        return target
    else:
        if abs(num_frames - 49) <= abs(num_frames - 81):
            return 49
        elif num_frames < 121:
            return 81
        else:
            return 81


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get video dimensions and frame count.
    """
    import decord
    decord.bridge.set_bridge('torch')
    
    vr = decord.VideoReader(video_path)
    height = vr[0].shape[0]
    width = vr[0].shape[1]
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    
    return height, width, num_frames, fps


def load_video_frames_from_middle(
    video_path: str,
    middle_frame_index: int,
    frames_before: int,
    frames_after: int,
    target_fps=16, max_frames=81, target_size=(480, 848), select_last=False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load video frames around middle frame.
    Returns: (before_tensor, middle_tensor, after_tensor)
    Each tensor has shape (1, C, T, H, W) normalized to [-1, 1]
    """
    try:
         # Initialize video reader with error handling
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        if total_frames < max_frames:
            raise ValueError(f"Video {video_path} has only {total_frames} frames, but max_frames is set to {max_frames}")
            
        # Get original frame dimensions
        original_h, original_w = vr[0].shape[:2]
        
        if select_last:
            start = total_frames - max_frames
            frame_indices = list(range(start, total_frames))
        else:
            frame_indices = list(range(min(max_frames, total_frames)))

        
        # Extract frames with memory safety
        try:
            frames = vr.get_batch(frame_indices).asnumpy()  # Shape: (T, H, W, C)
        except Exception as e:
            frames = vr.get_batch(frame_indices).numpy()  # Shape: (T, H, W, C)
        
        # Immediately delete video reader to free memory
        del vr
        
        # Crop and resize frames
        target_h, target_w = target_size
        # Calculate aspect ratios
        original_aspect = original_w / original_h
        target_aspect = target_w / target_h
        
        # Center crop to match target aspect ratio
        if original_aspect > target_aspect:
            # Original is wider, crop width
            crop_w = int(original_h * target_aspect)
            crop_h = original_h
            start_w = (original_w - crop_w) // 2
            start_h = 0
        else:
            # Original is taller, crop height
            crop_h = int(original_w / target_aspect)
            crop_w = original_w
            start_h = (original_h - crop_h) // 2
            start_w = 0
            
        # Apply center crop (before converting to tensor)
        frames = frames[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]  # (T, crop_h, crop_w, C)
        
        # Convert to torch tensor and change format
        frames = torch.from_numpy(frames).float()  # (T, crop_h, crop_w, C)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, crop_h, crop_w)
        
        # Resize to target size
        resize_transform = transforms.Resize((target_h, target_w), antialias=True)
        frames = resize_transform(frames)

        # Calculate frame indices
        start_idx = max(0, middle_frame_index - frames_before)
        end_idx = min(total_frames, middle_frame_index + frames_after + 1)
        middle_idx = middle_frame_index - start_idx
        
        # Get middle frame as numpy array (H, W, 3) for visualization
        # Convert from (T, C, H, W) to (H, W, C) for middle frame
        # Values are still in [0, 255] range at this point
        middle_frame_tensor = frames[middle_idx].permute(1, 2, 0)  # (H, W, C)
        # Convert to numpy and ensure uint8 format
        middle_frame_numpy = middle_frame_tensor.clamp(0, 255).cpu().numpy().astype(np.uint8)

        # Normalize to [-1, 1] range (assuming input is in [0, 255])
        frames = frames / 127.5 - 1.0
        
        # Add batch dimension and rearrange to (B, C, T, H, W)
        frames = frames.unsqueeze(0)  # (1, T, C, H, W)
        frames = frames.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
        
        # Split into before, middle, after
        middle_idx_in_extracted = middle_frame_index - start_idx
        before_tensor = frames[:, :, :middle_idx_in_extracted+1, :, :] if middle_idx_in_extracted > 0 else torch.zeros(1, 3, 0, target_h, target_w, dtype=frames.dtype, device=frames.device)
        after_tensor = frames[:, :, middle_idx_in_extracted:, :, :] if middle_idx_in_extracted+1 < frames.shape[2] else torch.zeros(1, 3, 0, target_h, target_w, dtype=frames.dtype, device=frames.device)
        
        return before_tensor, after_tensor, middle_frame_numpy
    except MemoryError as e:
        print(f"MemoryError loading video {video_path}: {str(e)}")
        import gc
        gc.collect()  # Force garbage collection
        return None, None, None
    except Exception as e:
        import traceback
        print(f"Error loading video {video_path}: {str(e)}, {traceback.format_exc()}")
        return None, None, None

def load_and_resize_video(
    video_path: str,
    target_height: int,
    target_width: int,
    target_frames: int,
) -> List[Image.Image]:
    """
    Load video and resize to target dimensions.
    """
    def convert_video(frames: List[Image.Image]) -> List[Image.Image]:
        total_frames = len(frames)
        
        if total_frames > target_frames:
            # indices = [int(i * total_frames / target_frames) for i in range(target_frames)]
            # indices = [min(idx, total_frames - 1) for idx in indices]
            # frames = [frames[i] for i in indices]
            frames = frames[:target_frames]
        elif total_frames < target_frames:
            frames = frames + [frames[-1]] * (target_frames - total_frames)
        
        frames = [frame.resize((target_width, target_height), Image.LANCZOS) for frame in frames]
        
        return frames
    
    video = load_video(video_path, convert_method=convert_video)
    
    return video


def create_pipeline(
    base_model_path: str,
    transformer_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Create LucyEditPipeline with optional custom transformer or LoRA.
    """
    logger.info(f"Loading pipeline from {base_model_path}")
    
    # Load VAE (always float32 for accuracy)
    logger.info("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        base_model_path, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    
    # Load custom transformer if specified
    if transformer_path is not None:
        logger.info(f"Loading custom transformer from {transformer_path}")
        transformer = WanTransformer3DModel.from_pretrained(
            transformer_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        _modify_patch_embedding_for_channel_concat(transformer)
        
        pipe = WanEditPipeline.from_pretrained(
            base_model_path,
            vae=vae,
            transformer=transformer,
            torch_dtype=dtype,
        )
    else:
        pipe = WanEditPipeline.from_pretrained(
            base_model_path,
            vae=vae,
            torch_dtype=dtype,
        )
    
    # Load LoRA if specified
    if lora_path is not None:
        logger.info(f"Loading LoRA adapter from {lora_path}")
        pipe.load_lora_weights(lora_path)
    
    pipe.to(device)
    
    logger.info("Pipeline loaded successfully")
    logger.info(f"  Transformer in_channels: {pipe.transformer.config.in_channels}")
    logger.info(f"  Transformer out_channels: {pipe.transformer.config.out_channels}")
    
    return pipe


def split_samples_for_distributed(samples: List[dict], accelerator: Accelerator) -> List[dict]:
    """
    Split samples across distributed processes.
    Each process gets a unique subset of samples.
    """
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    
    # Calculate samples per process
    total_samples = len(samples)
    samples_per_process = total_samples // num_processes
    remainder = total_samples % num_processes
    
    # Distribute remainder samples to first few processes
    if process_index < remainder:
        start_idx = process_index * (samples_per_process + 1)
        end_idx = start_idx + samples_per_process + 1
    else:
        start_idx = process_index * samples_per_process + remainder
        end_idx = start_idx + samples_per_process
    
    local_samples = samples[start_idx:end_idx]
    
    logger.info(
        f"Process {process_index}/{num_processes}: "
        f"Processing samples {start_idx} to {end_idx - 1} "
        f"({len(local_samples)} samples)"
    )
    
    return local_samples


def parse_args():
    parser = argparse.ArgumentParser(description="LucyEdit Distributed Inference Script")
    
    # Model paths
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="decart-ai/Lucy-Edit-1.1-Dev",
        help="Path to the base LucyEdit model",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to custom trained transformer model (optional)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (optional)",
    )
    
    # Input/Output
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON file with list of objects containing: video_path, instruction, save_name, middle_frame_index (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output videos",
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        default="",
        help="Base path to prepend to video paths in JSON",
    )
    
    # Frame settings
    parser.add_argument(
        "--frame_mode",
        type=str,
        default="auto",
        choices=["49", "81", "auto", "nearest"],
        help="Frame count mode",
    )
    parser.add_argument(
        "--middle_frame_index",
        type=int,
        default=None,
        help="Index of the middle frame (0-based). If None, will use middle of video or from JSON",
    )
    parser.add_argument(
        "--frames_before",
        type=int,
        default=None,
        help="Number of frames to generate before middle frame. If None, will use half of target_frames",
    )
    parser.add_argument(
        "--frames_after",
        type=int,
        default=None,
        help="Number of frames to generate after middle frame. If None, will use half of target_frames",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=24.0,
        help="Output video FPS",
    )
    
    # Inference settings
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    # Device settings
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for inference",
    )
    
    return parser.parse_args()


def process_single_sample(
    sample: dict,
    idx: int,
    total: int,
    pipe,
    args: argparse.Namespace,
    generator: Optional[torch.Generator],
    process_index: int,
    device=None,
    is_main_process=False
) -> dict:
    """
    Process a single video sample with middle frame inference.
    Generates frames before and after the middle frame.
    """
    video_path = sample['video_path']
    instruction = sample['instruction']
    save_name = sample.get('save_name')
    
    # Prepend base path if specified
    if args.video_base_path:
        full_video_path = os.path.join(args.video_base_path, video_path)
    else:
        full_video_path = video_path
    
    logger.info(f"\n[Process {process_index}] {'=' * 50}")
    logger.info(f"[Process {process_index}] Processing sample {idx + 1}/{total}")
    logger.info(f"[Process {process_index}]   Video: {full_video_path}")
    logger.info(f"[Process {process_index}]   Instruction: {instruction}")
    
    try:
        # Get video info for bucketing
        original_height, original_width, original_frames, original_fps = get_video_info(full_video_path)
        original_pixels = original_height * original_width
        logger.info(
            f"[Process {process_index}]   Original: {original_width}x{original_height} "
            f"({original_pixels // 1000}k pixels), {original_frames} frames @ {original_fps:.2f} fps"
        )
        
        # Find best resolution bucket
        target_height, target_width = get_bucket(original_height, original_width)
        
        # Determine target frames
        target_frames = get_target_frames(original_frames, args.frame_mode)
        
        # Get middle frame index from sample or args
        middle_frame_index = sample.get('middle_frame_index', args.middle_frame_index)
        if middle_frame_index is None:
            # Default to middle of video
            middle_frame_index = original_frames // 2
        
        # Determine frames before and after
        if args.frames_before is not None:
            frames_before = args.frames_before
        else:
            frames_before = target_frames // 2
        
        if args.frames_after is not None:
            frames_after = args.frames_after
        else:
            frames_after = target_frames // 2
        
        logger.info(
            f"[Process {process_index}]   Middle frame index: {middle_frame_index}, "
            f"Frames before: {frames_before}, Frames after: {frames_after}"
        )
        logger.info(
            f"[Process {process_index}]   Target: {target_width}x{target_height}, "
            f"Total target frames: {frames_before + 1 + frames_after} (frame_mode={args.frame_mode})"
        )
        
        # Process guide image (middle frame)
        guide_image_path = sample.get("edited_middle_frame_path") or sample.get("edited_first_frame_path")
        if guide_image_path is None:
            data_dir = "/opt/huawei/explorer-env/dataset/VidGen_data_chy/style-results-visual/style-video-bench-middle-frame-stylized"
            guide_image_path = os.path.join(data_dir, save_name.replace(".mp4", ".png"))
        else:
            guide_image_path = os.path.join(args.video_base_path, guide_image_path)

        guide_image_tensor, _ = load_image(guide_image_path, target_size=(target_height, target_width))
        guide_image_tensor = guide_image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
        guide_latents = process_vae_latent(guide_image_tensor, pipe.vae)

        # Load video frames from middle frame
        before_tensor, after_tensor, _ = load_video_frames_from_middle(
            full_video_path,
            middle_frame_index,
            frames_before,
            frames_after,
            (target_height, target_width)
        )
        
        # Move to device
        before_tensor = before_tensor.to(device=device, dtype=torch.float32)
        after_tensor = after_tensor.to(device=device, dtype=torch.float32)
        
        logger.info(
            f"[Process {process_index}]   Loaded frames - Before: {before_tensor.shape[2]}, "
            f"After: {after_tensor.shape[2]}"
        )

        # Inference forward (middle -> after)
        output_frames = []


        if after_tensor.shape[2] > 0:
            # Combine middle + after for forward inference
            forward_latent = process_vae_latent(after_tensor, pipe.vae)
            print("[DEBUG] video_latent shape:", forward_latent.shape, flush=True)
            logger.info(f"[Process {process_index}]   Running forward inference ({forward_latent.shape[2]} frames)...")
            forward_output = pipe(
                prompt=instruction,
                video=forward_latent,
                negative_prompt=args.negative_prompt,
                height=target_height,
                width=target_width,
                num_frames=forward_latent.shape[2],
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                guide_latents=guide_latents
            ).frames[0]
            output_frames.extend(forward_output)
        
        # Inference backward (before <- middle)
        if before_tensor.shape[2] > 0:
            # Reverse before frames for backward inference (middle -> before, reversed)
            before_reversed = torch.flip(before_tensor, dims=[2])
            backward_latent = process_vae_latent(before_reversed, pipe.vae)
            logger.info(f"[Process {process_index}]   Running backward inference ({backward_latent.shape[2]} frames)...")
            backward_output = pipe(
                prompt=instruction,
                video=backward_latent,
                negative_prompt=args.negative_prompt,
                height=target_height,
                width=target_width,
                num_frames=backward_latent.shape[2],
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                guide_latents=guide_latents
            ).frames[0]
            # Reverse back and skip middle frame (already in output_frames)
            backward_frames = backward_output[1:] if len(backward_output) > 1 else []
            backward_frames_reversed = list(reversed(backward_frames))
            output_frames = backward_frames_reversed + output_frames
        
        output = output_frames
        total_inference_frames = len(output_frames)
        
        # Save output video using save_name from JSON if available, else fallback to original stem
        if save_name:
            output_filename = save_name
            # Ensure extension is present if user forgot it
            if not output_filename.lower().endswith('.mp4'):
                output_filename += '.mp4'
        else:
            video_name = Path(sample['video_path']).stem
            output_filename = f"{video_name}_middle_{middle_frame_index}_edited.mp4"
            
        output_path = os.path.join(args.output_dir, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        export_to_video(output, output_path, fps=args.output_fps)
        
        result = {
            'input_video': sample['video_path'],
            'instruction': instruction,
            'output_video': output_path,
            'save_name': save_name,
            'status': 'success',
            'original_resolution': f"{original_width}x{original_height}",
            'original_pixels': original_pixels,
            'original_frames': original_frames,
            'middle_frame_index': middle_frame_index,
            'frames_before': frames_before,
            'frames_after': frames_after,
            'target_resolution': f"{target_width}x{target_height}",
            'target_frames': total_inference_frames,
            'process_index': process_index,
        }
        
        logger.info(f"[Process {process_index}]   ✓ Saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"[Process {process_index}]   ✗ Error processing {full_video_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        result = {
            'input_video': sample['video_path'],
            'instruction': instruction,
            'output_video': None,
            'save_name': save_name,
            'status': f'error: {str(e)}',
            'process_index': process_index,
        }
    
    return result


def main():
    args = parse_args()
    
    # Initialize accelerator for distributed inference
    accelerator = Accelerator()
    
    # Get distributed info
    process_index = accelerator.process_index
    num_processes = accelerator.num_processes
    is_main_process = accelerator.is_main_process
    device = accelerator.device
    
    # Setup output directory (only on main process to avoid race conditions)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Wait for main process to create directory
    accelerator.wait_for_everyone()
    
    # Setup logging to file (each process has its own log file)
    log_file = os.path.join(args.output_dir, f"inference_process_{process_index}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)
    
    if is_main_process:
        logger.info("=" * 80)
        logger.info("LucyEdit Distributed Inference")
        logger.info("=" * 80)
        logger.info(f"Arguments: {vars(args)}")
        logger.info(f"Number of processes: {num_processes}")
        logger.info(f"Distributed backend: {accelerator.distributed_type}")
    
    logger.info(f"[Process {process_index}] Initialized on device: {device}")
    
    # Setup dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Create pipeline on this process's device
    pipe = create_pipeline(
        base_model_path=args.base_model_path,
        transformer_path=args.transformer_path,
        lora_path=args.lora_path,
        dtype=dtype,
        device=device,
    )
    
    # Setup generator for reproducibility (different seed per process for diversity)
    generator = None
    if args.seed is not None:
        # Use different seed per process to ensure different outputs if needed
        process_seed = args.seed + process_index
        generator = torch.Generator(device=device).manual_seed(process_seed)
        logger.info(f"[Process {process_index}] Using seed: {process_seed}")
    
    # Read input JSON (all processes read the same file)
    logger.info(f"[Process {process_index}] Reading input JSON: {args.input_json}")
    samples = []
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Ensure data is a list
            if isinstance(data, list):
                samples = data
            else:
                logger.error("JSON root must be a list of objects")
                raise ValueError("JSON root must be a list of objects")
                
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        raise
    
    total_samples = len(samples)
    if is_main_process:
        logger.info(f"Total samples to process: {total_samples}")
    
    # Split samples across processes
    local_samples = split_samples_for_distributed(samples, accelerator)
    
    # Process local samples
    local_results = []
    for idx, sample in enumerate(tqdm(
        local_samples, 
        desc=f"Process {process_index}", 
        disable=not is_main_process,
        position=process_index
    )):
        result = process_single_sample(
            sample=sample,
            idx=idx,
            total=len(local_samples),
            pipe=pipe,
            args=args,
            generator=generator,
            process_index=process_index,
            device=device,
            is_main_process=is_main_process
        )
        local_results.append(result)
        
        # Clear GPU cache between samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    # Gather results from all processes
    logger.info(f"[Process {process_index}] Gathering results from all processes...")
    all_results = gather_object(local_results)
    
    # Save combined results (only on main process)
    if is_main_process:
        # Flatten the gathered results (gather_object returns list of lists)
        combined_results = []
        for process_results in all_results:
            if isinstance(process_results, list):
                combined_results.extend(process_results)
            else:
                combined_results.append(process_results)
        
        # Sort by input video to maintain consistent ordering
        combined_results.sort(key=lambda x: x['input_video'])
        
        # Save results summary
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        success_count = sum(1 for r in combined_results if r['status'] == 'success')
        
        logger.info("\n" + "=" * 80)
        logger.info("Distributed Inference Complete!")
        logger.info("=" * 80)
        logger.info(f"  Total processes: {num_processes}")
        logger.info(f"  Total samples: {len(combined_results)}")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Failed: {len(combined_results) - success_count}")
        logger.info(f"  Results saved to: {results_path}")
        logger.info("=" * 80)
        
        # Also print to console
        print("\n" + "=" * 80)
        print("Distributed Inference Complete!")
        print("=" * 80)
        print(f"  Total processes: {num_processes}")
        print(f"  Total samples: {len(combined_results)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {len(combined_results) - success_count}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Results saved to: {results_path}")
        print("=" * 80)
    
    # Final sync
    accelerator.wait_for_everyone()
    logger.info(f"[Process {process_index}] Done!")


if __name__ == "__main__":
    main()

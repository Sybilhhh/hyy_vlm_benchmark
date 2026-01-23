import os
import sys
import json
import copy
import random
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict
from einops import rearrange
from pathlib import Path

import pandas as pd
import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import Compose
import decord
from torchvision import transforms
from PIL import Image

class ShortVideoError(ValueError):
    """Video has fewer frames than required max_frames; skip this sample."""

# Handle both module import and script execution
try:
    from .sampler import MultiResJointSampler, MultiResJointSampler_distributed, MultiResMultiTaskJointSampler, MultiResMultiTaskJointSampler_distributed
    from .instruction_templates import CAMERA_MOTION_INSTRUCTION_TEMPLATES
except ImportError:
    # If running as a script, use absolute import
    from sampler import MultiResJointSampler, MultiResJointSampler_distributed, MultiResMultiTaskJointSampler, MultiResMultiTaskJointSampler_distributed
    from instruction_templates import CAMERA_MOTION_INSTRUCTION_TEMPLATES

from diffusers.utils import load_image


def is_main_process():
    """Check if current process is the main process."""
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def parse_bucket(bucket_str):
    """
    Parse bucket string (e.g., "848_480") to get (height, width)
    
    Args:
        bucket_str (str): Bucket string in format "H_W"
    
    Returns:
        tuple: (height, width) as integers
    """
    try:
        parts = bucket_str.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid bucket format: {bucket_str}")
        height, width = int(parts[0]), int(parts[1])

        assert height % 16 == 0, f"Height {height} must be divisible by 16"
        assert width % 16 == 0, f"Width {width} must be divisible by 16"
        
        return (height, width)
    except Exception as e:
        print(f"Error parsing bucket {bucket_str}: {str(e)}")
        return None


def is_video_file(file_path):
    """
    Check if a file is a video based on its extension.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        bool: True if the file is a video, False if it's an image
    """
    if not file_path:
        return False
    
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.ts', '.mts', '.m2ts'}
    
    # Get file extension
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions


def load_video_frames(video_path, target_fps=24, max_frames=81, target_size=(480, 848)):
    """
    Load video frames using decord and apply crop and resize transforms for VAE input
    
    Args:
        video_path (str): Path to the video file
        target_fps (int): Target fps for sampling frames
        max_frames (int): Maximum number of frames to extract
        target_size (tuple): Target size (height, width) for resizing
    
    Returns:
        tuple: (torch.Tensor, numpy.ndarray)
            - torch.Tensor: Video tensor of shape (1, 3, T, H, W) normalized to [-1, 1]
            - numpy.ndarray: First frame as numpy array of shape (H, W, 3) with values in [0, 255]
    """
    try:
        # Initialize video reader
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        # If the video is shorter than max_frames, skip it (do not pad / do not process).
        if total_frames < max_frames:
            raise ShortVideoError(
                f"Short video (<{max_frames} frames): total_frames={total_frames}, path={video_path}"
            )
        
        # Get original frame dimensions
        original_h, original_w = vr[0].shape[:2]
        
        if total_frames >= max_frames:
            if 81 < total_frames < 121:
                start_idx = total_frames - max_frames
                end_idx = total_frames
            else:
                start_idx = 0
                end_idx = max_frames
            frame_indices = list(range(start_idx, end_idx))
        else:
            # Short video: load all frames then pad
            frame_indices = list(range(0, total_frames))

        # Extract frames
        frames = vr.get_batch(frame_indices).asnumpy()  # Shape: (T, H, W, C)
        
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

        # Get first frame as numpy array (H, W, 3) for visualization
        # Convert from (T, C, H, W) to (H, W, C) for first frame
        # Values are still in [0, 255] range at this point
        first_frame_tensor = frames[0].permute(1, 2, 0)  # (H, W, C)
        # Convert to numpy and ensure uint8 format
        first_frame_numpy = first_frame_tensor.clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        # Normalize to [-1, 1] range (assuming input is in [0, 255])
        frames = frames / 127.5 - 1.0
        
        # Add batch dimension and rearrange to (B, C, T, H, W)
        frames = frames.unsqueeze(0)  # (1, T, C, H, W)
        frames = frames.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
        
        return frames, first_frame_numpy
        
    except ShortVideoError:
        raise
    except Exception as e:
        print(f"Error loading video {video_path}: {str(e)}")
        return None, None


def load_video_key_frames(video_path, max_frames=81, target_size=(480, 848)):
    """
    Load key frames (first, middle, last) from video using decord
    
    Args:
        video_path (str): Path to the video file
        target_size (tuple): Target size (height, width) for resizing
    
    Returns:
        tuple: (torch.Tensor, numpy.ndarray)
            - torch.Tensor: Key frames tensor of shape (1, 3, 3, H, W) normalized to [-1, 1]
            - numpy.ndarray: Key frames as numpy array of shape (3, H, W, 3) with values in [0, 255]
    """
    try:
        # Initialize video reader
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)

        # Skip short videos (do not process).
        if total_frames < max_frames:
            raise ShortVideoError(
                f"Short video (<{max_frames} frames): total_frames={total_frames}, path={video_path}"
            )
        
        # Get original frame dimensions
        original_h, original_w = vr[0].shape[:2]
        
        # Calculate key frame indices: first, middle, last
        if total_frames <= 0:
            raise ValueError(f"Video {video_path} has 0 frames.")
        if total_frames == 1:
            key_indices = [0]
        elif total_frames == 2:
            key_indices = [0, 1]
        else:
            # If video is longer than max_frames, mimic load_video_frames() cropping window
            if total_frames >= max_frames:
                if 81 < total_frames < 121:
                    start_idx = total_frames - max_frames
                    window_len = max_frames
                else:
                    start_idx = 0
                    window_len = max_frames
            else:
                start_idx = 0
                window_len = total_frames
            key_indices = [start_idx, start_idx + window_len // 2, start_idx + window_len - 1]
        
        # Extract key frames
        key_frames = vr.get_batch(key_indices).asnumpy()  # Shape: (3, H, W, C)
        
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
        key_frames = key_frames[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]  # (3, crop_h, crop_w, C)
        
        # Convert to torch tensor and change format
        key_frames_tensor = torch.from_numpy(key_frames).float()  # (3, crop_h, crop_w, C)
        key_frames_tensor = key_frames_tensor.permute(0, 3, 1, 2)  # (3, C, crop_h, crop_w)
        
        # Resize to target size
        resize_transform = transforms.Resize((target_h, target_w), antialias=True)
        key_frames_tensor = resize_transform(key_frames_tensor)
        
        # Get key frames as numpy array (3, H, W, 3) for vision encoder
        # Convert from (3, C, H, W) to (3, H, W, C)
        # Values are still in [0, 255] range at this point
        key_frames_numpy = key_frames_tensor.permute(0, 2, 3, 1)  # (3, H, W, C)
        # Convert to numpy and ensure uint8 format
        key_frames_numpy = key_frames_numpy.clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        # Normalize to [-1, 1] range (assuming input is in [0, 255])
        key_frames_tensor = key_frames_tensor / 127.5 - 1.0
        
        # Add batch dimension and rearrange to (B, C, T, H, W) where T=3
        key_frames_tensor = key_frames_tensor.unsqueeze(0)  # (1, 3, C, H, W)
        key_frames_tensor = key_frames_tensor.permute(0, 2, 1, 3, 4)  # (1, C, 3, H, W)
        
        return key_frames_tensor, key_frames_numpy
        
    except ShortVideoError:
        raise
    except Exception as e:
        print(f"Error loading key frames from video {video_path}: {str(e)}")
        return None, None


def load_image(image_path, target_size=(480, 848)):
    """
    Load image using PIL and apply crop and resize transforms for VAE input
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size (height, width) for resizing
    
    Returns:
        tuple: (torch.Tensor, numpy.ndarray)
            - torch.Tensor: Image tensor of shape (1, 3, H, W) normalized to [-1, 1]
            - numpy.ndarray: Image as numpy array of shape (H, W, 3) with values in [0, 255]
    """
    try:
        # Load image using PIL
        image = Image.open(image_path).convert("RGB")
        original_h, original_w = image.size[1], image.size[0]  # PIL returns (width, height)
        
        # Crop and resize
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
        
        # Apply center crop
        image = image.crop((start_w, start_h, start_w + crop_w, start_h + crop_h))
        
        # Resize to target size
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Convert to numpy array (H, W, 3) with values in [0, 255]
        image_numpy = np.array(image).astype(np.uint8)
        
        # Convert to torch tensor and normalize
        image_tensor = torch.from_numpy(image_numpy).float()  # (H, W, C)
        image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)
        
        # Normalize to [-1, 1] range
        image_tensor = image_tensor / 127.5 - 1.0
        
        # Add batch dimension: (1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)

        # Reshape to (C, 1, H, W)
        image_tensor = rearrange(image_tensor, "1 c h w -> c 1 h w")
        
        return image_tensor, image_numpy
        
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None, None


def process_vae_latent(input_tensor, vae):
    """Process input tensor through VAE encoder"""
    with torch.no_grad():
        latent = vae.encode(input_tensor).latent_dist.mode()
        latent.mul_(vae.config.scaling_factor)
    return latent


class MultiResVideoEditDatasetOnline(torch.utils.data.Dataset):
    """
    Multi-resolution dataset that returns raw data paths and metadata.
    Feature extraction is done in the training loop, not in the dataset.
    
    This dataset only loads and returns raw paths (videos, images, captions),
    and feature extraction happens during training for better efficiency.
    
    CSV format for tv2v task:
        video1_path, video2_path, video1_caption, video2_caption, 
        editing_instruction, bucket, frames, task_type, data_type
    
    CSV format for style_transfer task:
        video1_path, video2_path, video1_caption, video2_caption, 
        editing_instruction, bucket, frames, task_type, data_type
        (Same format as tv2v)
    
    CSV format for vv2v task:
        video1_path, video2_path, editing_instruction (optional),
        bucket, frames, task_type, data_type
    
    CSV format for prop task:
        video1_path, video2_path, img1_path, img2_path,
        forward_text_path, forward_text_byt5_path, bucket, frames, task_type, data_type
    
    CSV format for iv2v task:
        video1_path, video2_path, img1_path, conditional_img_path,
        forward_text_path, forward_text_byt5_path, bucket, frames, task_type, data_type
        (conditional_img_path can be a JSON list or single path)
    
    CSV format for dense_prediction task:
        video_path, depth_path, text_path, text_byt5_path, 
        img_path, img_siglip_path, bucket, frames, task_type, data_type
    
    CSV format for ti2i task:
        img1_path, img2_path, img2_caption, editing_instruction, 
        bucket, frames, task_type, data_type
    
    CSV format for t2v task:
        video_path, caption, bucket, frames, task_type, data_type
    
    CSV format for t2i task:
        img_path, caption, bucket, frames, task_type, data_type
    """
    
    def __init__(
        self,
        csv_path: Union[str, List[str]],
        data_root: Union[str, List[str], None] = None,
        max_frames: int = 81,
        **kwargs,
    ):
        self.csv_path = csv_path
        if isinstance(csv_path, str):
            csv_path = [csv_path]
        if data_root is None or isinstance(data_root, str):
            data_root = [data_root] if data_root is not None else [None]
        
        if len(data_root) == 1:
            data_root = data_root * len(csv_path)
        
        assert len(csv_path) == len(data_root), \
            "The number of csv files and data root should be the same."
        
        self.data_root = data_root
        self.max_frames = max_frames
        
        # Separate lists for video and image data
        self.video_data_list = []
        self.image_data_list = []
        self.gen_video_data_list = []
        self.gen_image_data_list = []
        self.data_list = []
        
        # Multi-resolution organization: (type, bucket) -> data indices
        self.video_bucket_to_indices = defaultdict(list)
        self.image_bucket_to_indices = defaultdict(list)
        self.gen_video_bucket_to_indices = defaultdict(list)
        self.gen_image_bucket_to_indices = defaultdict(list)
        self.buckets = []
        self.video_buckets = []
        self.image_buckets = []
        self.gen_video_buckets = []
        self.gen_image_buckets = []
        
        # Multi-task organization: (task, bucket) -> data indices
        self.task_bucket_to_indices = defaultdict(list)
        self.task_buckets = defaultdict(list)
        
        # Organization by (bucket, frames) for videos with different frame counts
        self.video_bucket_frames_to_indices = defaultdict(list)
        self.task_bucket_frames_to_indices = defaultdict(list)
        self.image_bucket_frames_to_indices = defaultdict(list)

        print(f"Total CSV PATH List: {csv_path}")

        self.load_annotations(csv_path, data_root)

    def load_annotations(self, csv_path, data_root):
        """
        Load annotations from CSV files for multi-resolution editing and generation.
        Supports both image and video data with automatic detection.
        
        Expected CSV columns for tv2v:
            - video1_path, video2_path: paths to source and target videos
            - video1_caption, video2_caption: captions for videos
            - editing_instruction: text instruction for editing
            - bucket: resolution bucket (e.g., "480_848")
            - frames: number of frames (e.g., 49, 81, 121)
            - task_type: "tv2v"
            - data_type: "video"
        """
        for i, path in enumerate(csv_path):
            # 检测文件扩展名并使用相应的读取方法
            file_path = Path(path)
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(path)
                print(f"Loaded CSV file: {path} with {len(df)} rows")
            elif file_extension in ['.parquet', '.pq']:
                df = pd.read_parquet(path, engine='pyarrow')
                print(f"Loaded Parquet file: {path} with {len(df)} rows")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Only .csv and .parquet are supported.")
            
            # Check if data_type column exists
            has_data_type_column = "data_type" in df.columns
            if not has_data_type_column:
                # Fallback: Detect data type from column names
                if "video1_path" in df.columns or "video_path" in df.columns:
                    default_data_type = "video"
                elif "img1_path" in df.columns or "img_path" in df.columns:
                    default_data_type = "image"
                else:
                    raise ValueError(f"Cannot determine data type from CSV columns in {path}. "
                                   f"Please add 'data_type' column or include type-specific columns.")
                print(f"Warning: 'data_type' column not found in {path}. "
                      f"Inferring type as '{default_data_type}' from column names.")
            
            # Check if task_type column exists
            has_task_type_column = "task_type" in df.columns
            if not has_task_type_column:
                default_task_type = "tv2v"
                print(f"Warning: 'task_type' column not found in {path}. "
                      f"Using default task type '{default_task_type}'.")
            else:
                default_task_type = None

            # Process each row based on data type
            for idx, row in df.iterrows():
                # Get data type from column or use inferred default
                if has_data_type_column:
                    data_type = str(row["data_type"]).lower().strip()
                    if data_type not in ["image", "video"]:
                        print(f"Warning: Invalid data_type '{data_type}' at row {idx} in {path}. Skipping.")
                        continue
                else:
                    data_type = default_data_type
                
                if has_task_type_column:
                    task_type = str(row["task_type"]).lower().strip()
                    if task_type not in ["tv2v", "ti2i", "iv2v", "ii2i", "vv2v", "t2v", "t2i", "s2v", "is2v", "style_transfer", "dense_prediction", "conditional_gen", "prop"]:
                        print(f"Warning: Invalid task type '{task_type}' at row {idx} in {path}. Skipping.")
                        continue
                else:
                    task_type = default_task_type
                
                # Process based on data type
                if data_type == "video":
                    if task_type in ["tv2v", "style_transfer"]:
                        data_dict = self._load_video_tv2v_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.video_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = data_dict.get("frames", self.max_frames)  # Use frames from data or default
                            self.video_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by (bucket, frames) combination
                            bucket_frames_key = (bucket, frames)
                            self.video_bucket_frames_to_indices[bucket_frames_key].append(data_dict["index"])
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])
                            
                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)
                    elif task_type == "prop":
                        data_dict = self._load_video_tv2v_prop_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.video_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = data_dict.get("frames", self.max_frames)  # Use frames from data or default
                            self.video_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by (bucket, frames) combination
                            bucket_frames_key = (bucket, frames)
                            self.video_bucket_frames_to_indices[bucket_frames_key].append(data_dict["index"])
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])
                            
                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)
                    elif task_type == "iv2v":
                        data_dict = self._load_video_iv2v_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.video_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = data_dict.get("frames", self.max_frames)  # Use frames from data or default
                            self.video_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by (bucket, frames) combination
                            bucket_frames_key = (bucket, frames)
                            self.video_bucket_frames_to_indices[bucket_frames_key].append(data_dict["index"])
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])
                            
                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)
                    elif task_type == "ii2i":
                        data_dict = self._load_video_ii2i_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.video_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = 1  # Use frames from data or default
                            self.video_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by (bucket, frames) combination
                            bucket_frames_key = (bucket, frames)
                            self.video_bucket_frames_to_indices[bucket_frames_key].append(data_dict["index"])
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])
                            
                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)
                    elif task_type == "vv2v":
                        data_dict = self._load_video_vv2v_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.video_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = data_dict.get("frames", self.max_frames)  # Use frames from data or default
                            self.video_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by (bucket, frames) combination
                            bucket_frames_key = (bucket, frames)
                            self.video_bucket_frames_to_indices[bucket_frames_key].append(data_dict["index"])
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])
                            
                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)
                    elif task_type == "dense_prediction":
                        data_dict = self._load_video_dense_prediction_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.video_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = data_dict.get("frames", self.max_frames)  # Use frames from data or default
                            self.video_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by (bucket, frames) combination
                            bucket_frames_key = (bucket, frames)
                            self.video_bucket_frames_to_indices[bucket_frames_key].append(data_dict["index"])
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])
                            
                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)
                    elif task_type == "t2v":
                        data_dict = self._load_video_t2v_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.gen_video_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = data_dict.get("frames", self.max_frames)  # Use frames from data or default
                            self.gen_video_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by (bucket, frames) combination
                            bucket_frames_key = (bucket, frames)
                            # Note: gen_video might not have bucket_frames_to_indices, but we can add it if needed
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])
                            
                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)
                
                elif data_type == "image":
                    if task_type == "ti2i":
                        data_dict = self._load_image_ti2i_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.image_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = data_dict.get("frames", 1)  # Get frames from data_dict, default to 1 for images
                            self.image_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by (bucket, frames) combination
                            bucket_frames_key = (bucket, frames)
                            self.image_bucket_frames_to_indices[bucket_frames_key].append(data_dict["index"])
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])

                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)
                    elif task_type == "t2i":
                        data_dict = self._load_image_t2i_annotation(row, data_root[i])
                        if data_dict is not None:
                            data_dict["index"] = len(self.data_list)
                            self.gen_image_data_list.append(data_dict)
                            self.data_list.append(data_dict)
                            
                            # Organize by bucket
                            bucket = data_dict["bucket"]
                            frames = data_dict.get("frames", 1)  # Get frames from data_dict, default to 1 for images
                            self.gen_image_bucket_to_indices[bucket].append(data_dict["index"])
                            
                            # Organize by task and bucket
                            task = data_dict["task"]
                            task_bucket_key = (task, bucket)
                            self.task_bucket_to_indices[task_bucket_key].append(data_dict["index"])

                            # Organize by (task, bucket, frames)
                            task_bucket_frames_key = (task, bucket, frames)
                            self.task_bucket_frames_to_indices[task_bucket_frames_key].append(data_dict["index"])
                            
                            if bucket not in self.task_buckets[task]:
                                self.task_buckets[task].append(bucket)

        # Get unique buckets
        self.video_buckets = sorted(self.video_bucket_to_indices.keys())
        self.image_buckets = sorted(self.image_bucket_to_indices.keys())
        self.gen_video_buckets = sorted(self.gen_video_bucket_to_indices.keys())
        self.gen_image_buckets = sorted(self.gen_image_bucket_to_indices.keys())
        
        # Sort buckets for each task
        for task in self.task_buckets:
            self.task_buckets[task] = sorted(self.task_buckets[task])
        
        # Print statistics only on main process
        if is_main_process():
            print(f"Loaded {len(self.video_data_list)} video edit samples, {len(self.image_data_list)} image edit samples")
            print(f"Loaded {len(self.gen_video_data_list)} video generation samples, {len(self.gen_image_data_list)} image generation samples")
            print(f"Total: {len(self.data_list)} samples")
            print(f"\nVideo Edit resolution buckets ({len(self.video_buckets)}):")
            for bucket in self.video_buckets:
                count = len(self.video_bucket_to_indices[bucket])
                print(f"  {bucket}: {count} samples")
            
            print(f"\nImage Edit resolution buckets ({len(self.image_buckets)}):")
            for bucket in self.image_buckets:
                count = len(self.image_bucket_to_indices[bucket])
                print(f"  {bucket}: {count} samples")
            
            print(f"\nVideo Generation resolution buckets ({len(self.gen_video_buckets)}):")
            for bucket in self.gen_video_buckets:
                count = len(self.gen_video_bucket_to_indices[bucket])
                print(f"  {bucket}: {count} samples")
            
            print(f"\nImage Generation resolution buckets ({len(self.gen_image_buckets)}):")
            for bucket in self.gen_image_buckets:
                count = len(self.gen_image_bucket_to_indices[bucket])
                print(f"  {bucket}: {count} samples")
            
            # Print statistics by (bucket, frames) combination
            if self.video_bucket_frames_to_indices:
                print(f"\nVideo Edit (bucket, frames) combinations ({len(self.video_bucket_frames_to_indices)}):")
                for (bucket, frames), indices in sorted(self.video_bucket_frames_to_indices.items()):
                    count = len(indices)
                    print(f"  {bucket}, {frames} frames: {count} samples")
            
            if self.image_bucket_frames_to_indices:
                print(f"\nImage Edit (bucket, frames) combinations ({len(self.image_bucket_frames_to_indices)}):")
                for (bucket, frames), indices in sorted(self.image_bucket_frames_to_indices.items()):
                    count = len(indices)
                    print(f"  {bucket}, {frames} frames: {count} samples")
            
            # Print task-based statistics
            print(f"\nTask-based organization ({len(self.task_buckets)} tasks):")
            for task in sorted(self.task_buckets.keys()):
                buckets = self.task_buckets[task]
                total_samples = sum(len(self.task_bucket_to_indices[(task, bucket)]) for bucket in buckets)
                print(f"  {task}: {total_samples} samples across {len(buckets)} buckets")
                for bucket in buckets:
                    count = len(self.task_bucket_to_indices[(task, bucket)])
                    print(f"    {bucket}: {count} samples")
    
    def _load_video_tv2v_annotation(self, row, data_root):
        """Load video annotation from a CSV row for tv2v or style_transfer task."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        video1_path = row.get("video1_path", None)
        video2_path = row.get("video2_path", None)
        editing_instruction = row.get("editing_instruction", None) if "editing_instruction" in row else row.get("instruction", None)
        video_caption = row.get("video2_caption", None)
        frames = row.get("frames", None)
        task_type = row.get("task_type", "tv2v")  # Get task type, default to tv2v

        # Skip invalid entries
        if pd.isna(video1_path) or pd.isna(video2_path) or pd.isna(editing_instruction):
            return None
        
        # CFG instruction selection
        if video_caption is not None:
            cfg_random = random.random()
            if cfg_random < 0.2:
                instruction = video_caption
            elif 0.2 <= cfg_random < 0.6:
                instruction = editing_instruction
            elif 0.6 <= cfg_random <= 1:
                instruction = f"{editing_instruction} {video_caption}"
            else:
                raise ValueError(f"Invalid cfg_random value: {cfg_random}")
        else:
            instruction = editing_instruction

        # Convert frames to int if present, otherwise use default
        if pd.isna(frames):
            frames = self.max_frames
        else:
            try:
                frames = int(frames)
            except (ValueError, TypeError):
                frames = self.max_frames

        # Determine task type
        if pd.isna(task_type):
            task = "tv2v"
        else:
            task = str(task_type).lower().strip()
            if task not in ["tv2v", "style_transfer"]:
                task = "tv2v"  # Default to tv2v if invalid

        # Build full paths
        data_dict = {
            "type": "video",
            "video1_path": os.path.join(data_root, video1_path) if data_root else video1_path,
            "video2_path": os.path.join(data_root, video2_path) if data_root else video2_path,
            "instruction": instruction,
            "bucket": str(bucket),
            "frames": frames,
            "task": task,
        }
        
        return data_dict
    
    def _load_image_ti2i_annotation(self, row, data_root):
        """Load image annotation from a CSV row for ti2i task."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        img1_path = row.get("img1_path", None)
        img2_path = row.get("img2_path", None)
        editing_instruction = row.get("editing_instruction", None) if "editing_instruction" in row else row.get("instruction", None)
        img_caption = row.get("img2_caption", None)

        # Skip invalid entries
        if pd.isna(img1_path) or pd.isna(img2_path) or pd.isna(editing_instruction):
            return None
        
        # CFG instruction selection
        if img_caption is not None:
            cfg_random = random.random()
            if cfg_random < 0.2:
                instruction = img_caption
            elif 0.2 <= cfg_random < 0.6:
                instruction = editing_instruction
            elif 0.6 <= cfg_random <= 1:
                instruction = f"{img_caption} {editing_instruction}"
            else:
                raise ValueError(f"Invalid cfg_random value: {cfg_random}")
        else:
            instruction = editing_instruction

        # Get frames from CSV, default to 1 for images
        frames = row.get("frames", None)
        if pd.isna(frames):
            frames = 1  # Default for images
        else:
            try:
                frames = int(frames)
            except (ValueError, TypeError):
                frames = 1  # Default for images

        # Build full paths
        data_dict = {
            "type": "image",
            "img1_path": os.path.join(data_root, img1_path) if data_root else img1_path,
            "img2_path": os.path.join(data_root, img2_path) if data_root else img2_path,
            "instruction": instruction,
            "bucket": str(bucket),
            "frames": frames,
            "task": "ti2i",
        }
        
        return data_dict
    
    def _load_video_dense_prediction_annotation(self, row, data_root):
        """Load video dense prediction annotation from a CSV row."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        video_path = row.get("video_path", None)
        dense_path = row.get("dense_path", None)
        editing_instruction = row.get("editing_instruction", None) if "editing_instruction" in row else row.get("instruction", None)
        frames = row.get("frames", None)

        # Skip invalid entries
        if pd.isna(video_path) or pd.isna(dense_path):
            return None

        instruction = editing_instruction

        # Convert frames to int if present, otherwise use default
        if pd.isna(frames):
            frames = self.max_frames
        else:
            try:
                frames = int(frames)
            except (ValueError, TypeError):
                frames = self.max_frames

        # Build full paths
        data_dict = {
            "type": "video",
            "video1_path": os.path.join(data_root, video_path) if data_root else video_path,
            "video2_path": os.path.join(data_root, dense_path) if data_root else dense_path,
            "instruction": instruction,
            "bucket": str(bucket),
            "frames": frames,
            "task": "dense_prediction",
        }
        
        return data_dict
    
    def _load_video_tv2v_prop_annotation(self, row, data_root):
        """Load video prop annotation from a CSV row."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        video1_path = row.get("video1_path", None)
        video2_path = row.get("video2_path", None)
        editing_instruction = row.get("editing_instruction", None) if "editing_instruction" in row else row.get("instruction", None)
        video_caption = row.get("video2_caption", None)
        frames = row.get("frames", None)

        # Skip invalid entries
        if pd.isna(video1_path) or pd.isna(video2_path) or pd.isna(editing_instruction):
            return None
        
        # CFG instruction selection
        if video_caption is not None:
            cfg_random = random.random()
            if cfg_random < 0.2:
                instruction = video_caption
            elif 0.2 <= cfg_random < 0.6:
                instruction = editing_instruction
            elif 0.6 <= cfg_random <= 1:
                instruction = f"{editing_instruction} {video_caption}"
            else:
                raise ValueError(f"Invalid cfg_random value: {cfg_random}")
        else:
            instruction = editing_instruction


        # Build full paths
        data_dict = {
            "type": "cond_video",
            "video1_path": os.path.join(data_root, video1_path) if data_root else video1_path,
            "video2_path": os.path.join(data_root, video2_path) if data_root else video2_path,
            "instruction": instruction,
            "bucket": str(bucket),
            "frames": frames,
            "task": "prop",
        }
        
        return data_dict
    
    def _load_video_iv2v_annotation(self, row, data_root):
        """Load video iv2v annotation from a CSV row."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        video1_path = row.get("video1_path", None)
        video2_path = row.get("video2_path", None)
        conditional_img_path = row.get("conditional_img_path", None)
        video_caption = row.get("video2_caption", None)
        editing_instruction = row.get("editing_instruction", None) if "editing_instruction" in row else row.get("instruction", None)
        frames = row.get("frames", None)

        # Skip invalid entries
        if pd.isna(video1_path) or pd.isna(video2_path):
            return None

        # Handle conditional_img_path - can be a string or JSON list
        if conditional_img_path is not None and not pd.isna(conditional_img_path):
            if isinstance(conditional_img_path, str):
                try:
                    conditional_img_paths = json.loads(conditional_img_path)
                except (json.JSONDecodeError, TypeError):
                    conditional_img_paths = [conditional_img_path]
            else:
                conditional_img_paths = [conditional_img_path]
        else:
            conditional_img_paths = []
        
        # CFG instruction selection
        if video_caption is not None:
            cfg_random = random.random()
            if cfg_random < 0.5:
                instruction = editing_instruction
            elif 0.5 <= cfg_random <= 1:
                instruction = f"{editing_instruction} {video_caption}"
            else:
                raise ValueError(f"Invalid cfg_random value: {cfg_random}")
        else:
            instruction = editing_instruction

        # Convert frames to int if present, otherwise use default
        if pd.isna(frames):
            frames = self.max_frames
        else:
            try:
                frames = int(frames)
            except (ValueError, TypeError):
                frames = self.max_frames

        # Build full paths
        data_dict = {
            "type": "cond_video",
            "video1_path": os.path.join(data_root, video1_path) if data_root else video1_path,
            "video2_path": os.path.join(data_root, video2_path) if data_root else video2_path,
            "conditional_img_paths": [os.path.join(data_root, path) if data_root else path for path in conditional_img_paths],
            "instruction": instruction,
            "bucket": str(bucket),
            "frames": frames,
            "task": "iv2v",
        }
        
        return data_dict
    
    def _load_video_ii2i_annotation(self, row, data_root):
        """Load video ii2i annotation from a CSV row."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        video1_path = row.get("video1_path", None)
        video2_path = row.get("video2_path", None)
        conditional_img_path = row.get("conditional_img_path", None)
        editing_instruction = row.get("editing_instruction", None) if "editing_instruction" in row else row.get("instruction", None)
        video_caption = row.get("video2_caption", None)
        frames = row.get("frames", None)

        # Skip invalid entries
        if pd.isna(video1_path) or pd.isna(video2_path):
            return None

        # Handle conditional_img_path - can be a string or JSON list
        if conditional_img_path is not None and not pd.isna(conditional_img_path):
            if isinstance(conditional_img_path, str):
                try:
                    conditional_img_paths = json.loads(conditional_img_path)
                except (json.JSONDecodeError, TypeError):
                    conditional_img_paths = [conditional_img_path]
            else:
                conditional_img_paths = [conditional_img_path]
        else:
            conditional_img_paths = []
        
        # CFG instruction selection
        if video_caption is not None:
            cfg_random = random.random()
            if cfg_random < 0.5:
                instruction = editing_instruction
            elif 0.5 <= cfg_random <= 1:
                instruction = f"{editing_instruction} {video_caption}"
            else:
                raise ValueError(f"Invalid cfg_random value: {cfg_random}")
        else:
            instruction = editing_instruction

        # Convert frames to int if present, otherwise use default
        if pd.isna(frames):
            frames = self.max_frames
        else:
            try:
                frames = int(frames)
            except (ValueError, TypeError):
                frames = self.max_frames

        # Build full paths
        data_dict = {
            "type": "cond_image",
            "video1_path": os.path.join(data_root, video1_path) if data_root else video1_path,
            "video2_path": os.path.join(data_root, video2_path) if data_root else video2_path,
            "conditional_img_paths": [os.path.join(data_root, path) if data_root else path for path in conditional_img_paths],
            "instruction": instruction,
            "bucket": str(bucket),
            "frames": frames,
            "task": "ii2i",
        }
        
        return data_dict
    
    def _load_video_vv2v_annotation(self, row, data_root):
        """Load video vv2v (video-to-video) annotation from a CSV row."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        video1_path = row.get("video1_path", None)
        video2_path = row.get("video2_path", None)
        conditional_img_path = row.get("conditional_video_path", None)
        frames = row.get("frames", None)
        # Optional instruction for vv2v
        editing_instruction = row.get("editing_instruction", None) if "editing_instruction" in row else row.get("instruction", None)

        # Skip invalid entries
        if pd.isna(video1_path) or pd.isna(video2_path):
            return None

        # Apply camera motion instruction template with 80% probability
        if editing_instruction is not None and not pd.isna(editing_instruction):
            camera_template = random.choice(CAMERA_MOTION_INSTRUCTION_TEMPLATES)
            if random.random() < 0.5:
                editing_instruction = f"{camera_template}"
            elif 0.5 <= random.random() < 0.8:
                editing_instruction = f"{camera_template} {editing_instruction}"

        # Handle conditional_img_path - can be a string or JSON list
        if conditional_img_path is not None and not pd.isna(conditional_img_path):
            if isinstance(conditional_img_path, str):
                try:
                    conditional_img_paths = json.loads(conditional_img_path)
                except (json.JSONDecodeError, TypeError):
                    conditional_img_paths = [conditional_img_path]
            else:
                conditional_img_paths = [conditional_img_path]
        else:
            conditional_img_paths = []

        # Convert frames to int if present, otherwise use default
        if pd.isna(frames):
            frames = self.max_frames
        else:
            try:
                frames = int(frames)
            except (ValueError, TypeError):
                frames = self.max_frames

        # Build full paths
        data_dict = {
            "type": "cond_video",
            "video1_path": os.path.join(data_root, video1_path) if data_root else video1_path,
            "video2_path": os.path.join(data_root, video2_path) if data_root else video2_path,
            "conditional_img_paths": [os.path.join(data_root, path) if data_root else path for path in conditional_img_paths],
            "instruction": editing_instruction,
            "bucket": str(bucket),
            "frames": frames,
            "task": "vv2v",
        }
        
        return data_dict
    
    def _load_video_t2v_annotation(self, row, data_root):
        """Load video generation (t2v) annotation from a CSV row."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        video_path = row.get("video_path", None)
        caption = row.get("caption", None)
        frames = row.get("frames", None)

        # Skip invalid entries
        if pd.isna(video_path) or pd.isna(caption):
            return None

        # Convert frames to int if present, otherwise use default
        if pd.isna(frames):
            frames = self.max_frames
        else:
            try:
                frames = int(frames)
            except (ValueError, TypeError):
                frames = self.max_frames

        # Build full paths
        data_dict = {
            "type": "gen_video",
            "video_path": os.path.join(data_root, video_path) if data_root else video_path,
            "caption": caption,
            "bucket": str(bucket),
            "frames": frames,
            "task": "t2v",
        }
        
        return data_dict
    
    def _load_image_t2i_annotation(self, row, data_root):
        """Load image generation (t2i) annotation from a CSV row."""
        # Required fields - use raw paths instead of feature paths
        bucket = row.get("bucket", None)
        img_path = row.get("img_path", None)
        caption = row.get("caption", None)

        # Skip invalid entries
        if pd.isna(img_path) or pd.isna(caption):
            return None

        # Get frames from CSV, default to 1 for images
        frames = row.get("frames", None)
        if pd.isna(frames):
            frames = 1  # Default for images
        else:
            try:
                frames = int(frames)
            except (ValueError, TypeError):
                frames = 1  # Default for images

        # Build full paths
        data_dict = {
            "type": "gen_image",
            "img_path": os.path.join(data_root, img_path) if data_root else img_path,
            "caption": caption,
            "bucket": str(bucket),
            "frames": frames,
            "task": "t2i",
        }
        
        return data_dict
    
    def _get_video_item(self, data):
        """
        Load and return video data as tensors/numpy arrays.
        Feature extraction will be done in training loop.
        
        Args:
            data: data dictionary with video paths and metadata
        
        Returns:
            dict with keys:
                - type: "video"
                - task: task type
                - video1: torch.Tensor of shape (1, 3, T, H, W) normalized to [-1, 1]
                - video2: torch.Tensor of shape (1, 3, T, H, W) normalized to [-1, 1]
                - video1_key_frames: numpy.ndarray of shape (3, H, W, 3) with values in [0, 255]
                - video2_key_frames: numpy.ndarray of shape (3, H, W, 3) with values in [0, 255] (optional)
                - editing_instruction: text instruction for editing
                - video1_caption: caption for source video
                - video2_caption: caption for target video
                - bucket: resolution bucket
        """
        # Parse bucket to get target size
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket format: {data['bucket']}")
        
        # Get frames from data, fallback to self.max_frames if not present
        max_frames = data.get("frames", self.max_frames)
        if not isinstance(max_frames, int):
            max_frames = self.max_frames
        
        # Load video frames
        video1_tensor, _ = load_video_frames(
            data["video1_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        video2_tensor, _ = load_video_frames(
            data["video2_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video1_tensor is None or video2_tensor is None:
            raise ValueError(f"Failed to load video frames from {data['video1_path']} or {data['video2_path']}")
        
        # Load key frames for vision encoder
        _, video1_key_frames_numpy = load_video_key_frames(
            data["video1_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video1_key_frames_numpy is None:
            raise ValueError(f"Failed to load key frames from {data['video1_path']}")
        
        result = {
            "type": "video",
            "task": data["task"],
            "training_type": "normal",
            "video1": video1_tensor.squeeze(0),  # (3, T, H, W) tensor, normalized to [-1, 1]
            "video2": video2_tensor.squeeze(0),  # (3, T, H, W) tensor, normalized to [-1, 1]
            "video1_key_frames": video1_key_frames_numpy,  # (3, H, W, 3) numpy array, values in [0, 255]
            "bucket": data["bucket"],
            "index": data["index"],
        }
        
        # Add optional instruction if available
        if "instruction" in data and data["instruction"] is not None:
            result["instruction"] = data["instruction"]
        
        return result
    
    def _get_prop_video_item(self, data):
        """
        Load and return prop video data as tensors/numpy arrays.
        Feature extraction will be done in training loop.
        
        Args:
            data: data dictionary with video paths and metadata
        
        Returns:
            dict with keys:
                - type: "cond_video_prop"
                - task: "prop"
                - video1: torch.Tensor of shape (3, T, H, W) normalized to [-1, 1]
                - video2: torch.Tensor of shape (3, T, H, W) normalized to [-1, 1]
                - video1_key_frames: numpy.ndarray of shape (3, H, W, 3) with values in [0, 255]
                - img1_numpy: numpy.ndarray of shape (H, W, 3) with values in [0, 255]
                - img2_numpy: numpy.ndarray of shape (H, W, 3) with values in [0, 255] (single frame selected)
                - forward_text_path: path to text embedding
                - forward_text_byt5_path: path to text byt5 embedding
                - bucket: resolution bucket
        """
        # Parse bucket to get target size
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket format: {data['bucket']}")
        
        # Get frames from data, fallback to self.max_frames if not present
        max_frames = data.get("frames", self.max_frames)
        if not isinstance(max_frames, int):
            max_frames = self.max_frames
        
        # Load video frames
        video1_tensor, _ = load_video_frames(
            data["video1_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        video2_tensor, _ = load_video_frames(
            data["video2_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video1_tensor is None or video2_tensor is None:
            raise ValueError(f"Failed to load video frames from {data['video1_path']} or {data['video2_path']}")
        
        # Load key frames for vision encoder
        video1_key_frames_tensor, video1_key_frames_numpy = load_video_key_frames(
            data["video1_path"], 
            target_size=target_size,
            max_frames=max_frames
        )

        video2_key_frames_tensor, video2_key_frames_numpy = load_video_key_frames(
            data["video2_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video1_key_frames_numpy is None or video2_key_frames_tensor is None:
            raise ValueError(f"Failed to load key frames from {data['video1_path']} or {data['video2_path']}")

        result = {
            "type": "cond_video",
            "task": data["task"],
            "training_type": "normal",
            "video1": video1_tensor.squeeze(0),  # (3, T, H, W) tensor, normalized to [-1, 1]
            "video2": video2_tensor.squeeze(0),  # (3, T, H, W) tensor, normalized to [-1, 1]
            "video1_key_frames_np": video1_key_frames_numpy,  # (3, H, W, 3) numpy array, values in [0, 255]
            "video2_key_frames_np": video2_key_frames_numpy[0],  # (3, H, W, 3) numpy array, values in [0, 255]
            "video2_key_frames_tensor": video2_key_frames_tensor.squeeze(0)[:,:1],
            "instruction": data["instruction"],
            "bucket": data["bucket"],
            "index": data["index"],
        }
        
        return result
    
    def _get_cond_video_item(self, data):
        """
        Load and return conditional video (iv2v) data as tensors/numpy arrays.
        Feature extraction will be done in training loop.
        
        Args:
            data: data dictionary with video paths and metadata
        
        Returns:
            dict with keys:
                - type: "cond_video"
                - task: "iv2v"
                - video1: torch.Tensor of shape (3, T, H, W) normalized to [-1, 1]
                - video2: torch.Tensor of shape (3, T, H, W) normalized to [-1, 1]
                - video1_key_frames: numpy.ndarray of shape (3, H, W, 3) with values in [0, 255]
                - img1_numpy: numpy.ndarray of shape (H, W, 3) with values in [0, 255]
                - conditional_img_numpy: list of numpy.ndarray, each of shape (H, W, 3) with values in [0, 255]
                - forward_text_path: path to text embedding
                - forward_text_byt5_path: path to text byt5 embedding
                - bucket: resolution bucket
        """
        # Parse bucket to get target size
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket format: {data['bucket']}")
        
        # Get frames from data, fallback to self.max_frames if not present
        max_frames = data.get("frames", self.max_frames)
        if not isinstance(max_frames, int):
            max_frames = self.max_frames
        
        # Load video frames
        video1_tensor, _ = load_video_frames(
            data["video1_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        video2_tensor, _ = load_video_frames(
            data["video2_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video1_tensor is None or video2_tensor is None:
            raise ValueError(f"Failed to load video frames from {data['video1_path']} or {data['video2_path']}")
        
        # Load key frames for vision encoder
        _, video1_key_frames_numpy = load_video_key_frames(
            data["video1_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video1_key_frames_numpy is None:
            raise ValueError(f"Failed to load key frames from {data['video1_path']}")
        
        # Load conditional images/videos if available
        conditional_img_numpy_list = []
        conditional_img_tensor_list = []
        for cond_img_path in data["conditional_img_paths"]:
            if is_video_file(cond_img_path):
                # Load first frame from video
                cond_img_tensor, _ = load_video_frames(
                    cond_img_path,
                    target_size=target_size,
                    max_frames=data.get("frames", self.max_frames)
                )
                _, cond_img_numpy = load_video_key_frames(
                    cond_img_path, 
                    target_size=target_size, 
                    max_frames=max_frames
                )
                if cond_img_tensor is None or cond_img_numpy is None:
                    raise ValueError(f"Failed to load video frames from {cond_img_path}")
                cond_img_tensor = cond_img_tensor.squeeze(0)
            else:
                cond_img_tensor, cond_img_numpy = load_image(cond_img_path, target_size=target_size)

            if cond_img_tensor is None or cond_img_numpy is None:
                raise ValueError(f"Failed to load image from {cond_img_path}")
                
            if cond_img_numpy is not None:
                conditional_img_numpy_list.append(cond_img_numpy)  # (H, W, 3) numpy array
                conditional_img_tensor_list.append(cond_img_tensor)  # (3, H, W) tensor, normalized to [-1, 1]
        
        result = {
            "type": "cond_video",
            "task": data["task"],
            "training_type": "normal",
            "video1": video1_tensor.squeeze(0),  # (3, T, H, W) tensor, normalized to [-1, 1]
            "video2": video2_tensor.squeeze(0),  # (3, T, H, W) tensor, normalized to [-1, 1]
            "video1_key_frames_np": video1_key_frames_numpy,  # (3, H, W, 3) numpy array, values in [0, 255]
            "conditional_img_numpy": conditional_img_numpy_list,  # (H, W, 3) numpy array, values in [0, 255]
            "conditional_img_tensor": conditional_img_tensor_list,  # (3, H, W) tensor, normalized to [-1, 1]
            "instruction": data["instruction"],
            "bucket": data["bucket"],
            "index": data["index"],
        }
        
        return result
    
    def _get_image_item(self, data):
        """
        Load and return image data as tensors/numpy arrays.
        Feature extraction will be done in training loop.
        
        Args:
            data: data dictionary with image paths and metadata
        
        Returns:
            dict with keys:
                - type: "image"
                - task: task type
                - img1: torch.Tensor of shape (1, 3, H, W) normalized to [-1, 1]
                - img2: torch.Tensor of shape (1, 3, H, W) normalized to [-1, 1]
                - img1_numpy: numpy.ndarray of shape (H, W, 3) with values in [0, 255]
                - img2_numpy: numpy.ndarray of shape (H, W, 3) with values in [0, 255]
                - instruction: text instruction for editing
                - bucket: resolution bucket
        """
        # Parse bucket to get target size
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket format: {data['bucket']}")
        
        # Load images
        img1_tensor, img1_numpy = load_image(
            data["img1_path"], 
            target_size=target_size
        )

        img2_tensor, _ = load_image(
            data["img2_path"], 
            target_size=target_size
        )
        
        if img1_tensor is None or img2_tensor is None:
            raise ValueError(f"Failed to load images from {data['img1_path']} or {data['img2_path']}")

        result = {
            "type": "image",
            "task": data["task"],
            "training_type": "normal",
            "img1": img1_tensor,  # (3, H, W) tensor, normalized to [-1, 1]
            "img2": img2_tensor,  # (3, H, W) tensor, normalized to [-1, 1]
            "img1_numpy": img1_numpy,  # (H, W, 3) numpy array, values in [0, 255]
            "instruction": data["instruction"],
            "bucket": data["bucket"],
            "index": data["index"],
        }
        
        return result
    
    def _get_cond_image_item(self, data):
        """
        Load and return conditional image (ii2i) data as tensors/numpy arrays.
        Feature extraction will be done in training loop.
        
        Args:
            data: data dictionary with video paths and metadata
        
        Returns:
            dict with keys:
                - type: "cond_video"
                - task: "iv2v"
                - video1: torch.Tensor of shape (3, T, H, W) normalized to [-1, 1]
                - video2: torch.Tensor of shape (3, T, H, W) normalized to [-1, 1]
                - video1_key_frames: numpy.ndarray of shape (3, H, W, 3) with values in [0, 255]
                - img1_numpy: numpy.ndarray of shape (H, W, 3) with values in [0, 255]
                - conditional_img_numpy: list of numpy.ndarray, each of shape (H, W, 3) with values in [0, 255]
                - forward_text_path: path to text embedding
                - forward_text_byt5_path: path to text byt5 embedding
                - bucket: resolution bucket
        """
        # Parse bucket to get target size
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket format: {data['bucket']}")
        
        # Get frames from data, fallback to self.max_frames if not present
        max_frames = data.get("frames", self.max_frames)
        if not isinstance(max_frames, int):
            max_frames = self.max_frames
        
        # Load video frames
        video1_tensor, _ = load_video_frames(
            data["video1_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        video2_tensor, _ = load_video_frames(
            data["video2_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video1_tensor is None or video2_tensor is None:
            raise ValueError(f"Failed to load video frames from {data['video1_path']} or {data['video2_path']}")
        
        # Load key frames for vision encoder
        _, video1_key_frames_numpy = load_video_key_frames(
            data["video1_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video1_key_frames_numpy is None:
            raise ValueError(f"Failed to load key frames from {data['video1_path']}")
        
        # Load conditional images/videos if available
        conditional_img_numpy_list = []
        conditional_img_tensor_list = []
        for cond_img_path in data["conditional_img_paths"]:
            if is_video_file(cond_img_path):
                # Load first frame from video
                cond_img_tensor, _ = load_video_frames(
                    cond_img_path,
                    target_size=target_size,
                    max_frames=data.get("frames", self.max_frames)
                )
                _, cond_img_numpy = load_video_key_frames(
                    cond_img_path, 
                    target_size=target_size, 
                    max_frames=max_frames
                )
                cond_img_tensor = cond_img_tensor.squeeze(0)
            else:
                cond_img_tensor, cond_img_numpy = load_image(cond_img_path, target_size=target_size)
                
            if cond_img_tensor is None or cond_img_numpy is None:
                raise ValueError(f"Failed to load image from {cond_img_path}")
                
            if cond_img_numpy is not None:
                conditional_img_numpy_list.append(cond_img_numpy)  # (H, W, 3) numpy array
                conditional_img_tensor_list.append(cond_img_tensor)  # (3, H, W) tensor, normalized to [-1, 1]

        result = {
            "type": "cond_video",
            "task": data["task"],
            "training_type": "normal",
            "img1": video1_tensor.squeeze(0)[:,:1],  # (3, 1, H, W) tensor, normalized to [-1, 1]
            "img2": video2_tensor.squeeze(0)[:,:1],  # (3, 1, H, W) tensor, normalized to [-1, 1]
            "img1_numpy": video1_key_frames_numpy[0],  # (3, H, W, 3) numpy array, values in [0, 255]
            "conditional_img_numpy": conditional_img_numpy_list,  # (H, W, 3) numpy array, values in [0, 255]
            "conditional_img_tensor": conditional_img_tensor_list,  # (3, H, W) tensor, normalized to [-1, 1]
            "instruction": data["instruction"],
            "bucket": data["bucket"],
            "index": data["index"],
        }
        
        return result
    
    def _get_gen_video_item(self, data):
        """
        Load and return video generation data as tensors/numpy arrays.
        Feature extraction will be done in training loop.
        
        Args:
            data: data dictionary with video paths and metadata
        
        Returns:
            dict with keys:
                - type: "gen_video"
                - task: "t2v"
                - video: torch.Tensor of shape (3, T, H, W) normalized to [-1, 1]
                - video_key_frames: numpy.ndarray of shape (3, H, W, 3) with values in [0, 255]
                - caption: text caption for generation
                - bucket: resolution bucket
        """
        # Parse bucket to get target size
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket format: {data['bucket']}")
        
        # Get frames from data, fallback to self.max_frames if not present
        max_frames = data.get("frames", self.max_frames)
        if not isinstance(max_frames, int):
            max_frames = self.max_frames
        
        # Load video frames
        video_tensor, _ = load_video_frames(
            data["video_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )

        _, video_numpy = load_video_key_frames(
            data["video_path"], 
            target_size=target_size, 
            max_frames=max_frames
        )
        
        if video_tensor is None:
            raise ValueError(f"Failed to load video frames from {data['video_path']}")
        
        result = {
            "type": "gen_video",
            "task": data["task"],
            "training_type": "normal",
            "tensor": video_tensor.squeeze(0),  # (3, T, H, W) tensor, normalized to [-1, 1]
            "vision_numpy": video_numpy,
            "instruction": data["caption"],
            "bucket": data["bucket"],
            "index": data["index"],
        }
        
        return result
    
    def _get_gen_image_item(self, data):
        """
        Load and return image generation data as tensors/numpy arrays.
        Feature extraction will be done in training loop.
        
        Args:
            data: data dictionary with image paths and metadata
        
        Returns:
            dict with keys:
                - type: "gen_image"
                - task: "t2i"
                - img: torch.Tensor of shape (3, 1, H, W) normalized to [-1, 1]
                - img_numpy: numpy.ndarray of shape (H, W, 3) with values in [0, 255]
                - caption: text caption for generation
                - bucket: resolution bucket
        """
        # Parse bucket to get target size
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket format: {data['bucket']}")
        
        # Load image
        img_tensor, img_numpy = load_image(
            data["img_path"], 
            target_size=target_size
        )
        
        if img_tensor is None:
            raise ValueError(f"Failed to load image from {data['img_path']}")
        
        result = {
            "type": "gen_image",
            "task": data["task"],
            "training_type": "normal",
            "tensor": img_tensor,  # (3, 1, H, W) tensor, normalized to [-1, 1]
            "vision_numpy": img_numpy,
            "instruction": data["caption"],
            "bucket": data["bucket"],
            "index": data["index"],
        }
        
        return result
    
    def __getitem__(self, index):
        """
        Get item by index with error handling.
        Routes to appropriate handler based on data type.
        
        Args:
            index: sample index
        
        Returns:
            dict containing vae1_latent, vae2_latent, text_feature, type, etc.
        """
        try:
            data = self.data_list[index]
            data_type = data["type"]
            
            # Route to appropriate handler based on data type
            if data_type == "video":
                return self._get_video_item(data)
            elif data_type == "cond_video":
                # Check task type for cond_video
                if data.get("task") == "prop":
                    return self._get_prop_video_item(data)
                else:
                    return self._get_cond_video_item(data)
            elif data_type == "image":
                return self._get_image_item(data)
            elif data_type == "cond_image":
                return self._get_cond_image_item(data)
            elif data_type == "gen_video":
                return self._get_gen_video_item(data)
            elif data_type == "gen_image":
                return self._get_gen_image_item(data)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
                
        except Exception as e:
            if is_main_process():
                if isinstance(e, ShortVideoError):
                    # Do not spam traceback for known short-video skip.
                    print(f"[skip] index {index}: {e}")
                else:
                    print(f"Error loading index {index}: {e}")
                    import traceback
                    traceback.print_exc()

            if len(self.data_list) < 1:
                raise RuntimeError(f"Failed to load sample at index {index}: {e}")

            # ------ same-bucket fallback ------
            bucket = None
            frames = None
            data_type = None
            task = None
            if data is not None:
                bucket = data.get("bucket", None)
                frames = data.get("frames", None)
                data_type = data.get("type", None)
                task = data.get("task", None)
            
            # 1. find the same bucket, frames, and same task (preferred)
            candidates = []
            if bucket is not None and frames is not None and task is not None:
                candidates = list(self.task_bucket_frames_to_indices.get((task, bucket, frames), []))
            
            # 2. if no candidates, try same bucket and task (without frames constraint)
            if not candidates and bucket is not None and task is not None:
                candidates = list(self.task_bucket_to_indices.get((task, bucket), []))
            
            # 2. remove self from candidates
            candidates = [idx for idx in candidates if idx != index]

            # 3. retry with max 50 times
            max_retries = 50
            tried = set([index])

            for _ in range(max_retries):
                if candidates:
                    new_idx = random.choice(candidates)
                else:
                    new_idx = random.randint(0, len(self.data_list) - 1)

                if new_idx in tried:
                    continue
                tried.add(new_idx)
                
                try:
                    return self.__getitem__(new_idx)
                except Exception as e:
                    if new_idx in candidates:
                        candidates.remove(new_idx)
                    continue

            raise RuntimeError(f"Failed to load sample at index {index} (Bucket: {bucket}, Task: {task}, Data Type: {data_type}): {e}")
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.data_list)
    
    def get_sampler(self, video_batch_size, image_batch_size, gen_video_batch_size, gen_image_batch_size, distributed, batchsize_config_path=None):
        if distributed:
            if is_main_process():
                print("Get Distributed Sampler")
            return MultiResMultiTaskJointSampler_distributed(
                self,
                video_batch_size=video_batch_size,
                image_batch_size=image_batch_size,
                gen_video_batch_size=gen_video_batch_size,
                gen_image_batch_size=gen_image_batch_size,
                shuffle=True,
                batchsize_config_path=batchsize_config_path
            )
        else:
            if is_main_process():
                print("Get Normal Sampler")
            return MultiResMultiTaskJointSampler(
                self,
                video_batch_size=video_batch_size,
                image_batch_size=image_batch_size,
                gen_video_batch_size=gen_video_batch_size,
                gen_image_batch_size=gen_image_batch_size,
                shuffle=True,
                batchsize_config_path=batchsize_config_path
            )


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Multi-Resolution Dataset with Online Feature Extraction Test")
    print("=" * 80)
    
    # Note: This requires a pipeline object to be initialized
    # For testing, you would need to load the pipeline first
    print("This dataset requires a pipeline object for online feature extraction.")
    print("Please initialize the pipeline and pass it to the dataset constructor.")
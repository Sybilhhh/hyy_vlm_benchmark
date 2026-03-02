"""Multi-resolution multi-task dataset for Kandinsky-5 unified training.

Supports task types: t2v, t2i, ti2i, tv2v.
Loads raw videos/images from CSV annotations; feature extraction happens
in the training loop (online mode).

Adapted from HunyuanVideo 1.5 MultiResVideoEditDatasetOnline, with:
- SigLIP / ByT5 fields removed (Kandinsky uses Qwen2.5-VL multimodal)
- Spatial downsampling adjusted to 8x (Kandinsky) vs 16x (HunyuanVideo)
- Resolution buckets adjusted for Kandinsky (512/1024)
"""

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import decord
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def is_main_process():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def parse_bucket(bucket_str):
    """Parse bucket string (e.g., '512_768') to (height, width)."""
    try:
        parts = bucket_str.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid bucket format: {bucket_str}")
        height, width = int(parts[0]), int(parts[1])
        assert height % 8 == 0, f"Height {height} must be divisible by 8"
        assert width % 8 == 0, f"Width {width} must be divisible by 8"
        return (height, width)
    except Exception as e:
        print(f"Error parsing bucket {bucket_str}: {e}")
        return None


def is_video_file(file_path):
    if not file_path:
        return False
    video_extensions = {
        ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v",
    }
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions


def load_video_frames(video_path, target_size=(512, 768), max_frames=31):
    """Load video frames, center-crop and resize.

    Returns:
        frames: (1, C, T, H, W) float tensor in [-1, 1]
        first_frame_numpy: (H, W, C) uint8 numpy
    """
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        orig_h, orig_w = vr[0].shape[:2]

        if total_frames >= max_frames:
            if 81 < total_frames < 121:
                start_idx = total_frames - max_frames
                end_idx = total_frames
            else:
                start_idx = 0
                end_idx = max_frames
            frame_indices = list(range(start_idx, end_idx))
        else:
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (max_frames - total_frames)

        frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
        target_h, target_w = target_size

        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h
        if orig_aspect > target_aspect:
            crop_w = int(orig_h * target_aspect)
            crop_h = orig_h
            start_w = (orig_w - crop_w) // 2
            start_h = 0
        else:
            crop_h = int(orig_w / target_aspect)
            crop_w = orig_w
            start_h = (orig_h - crop_h) // 2
            start_w = 0

        frames = frames[:, start_h : start_h + crop_h, start_w : start_w + crop_w, :]
        frames = torch.from_numpy(frames).float()  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

        resize_transform = transforms.Resize((target_h, target_w), antialias=True)
        frames = resize_transform(frames)

        first_frame_numpy = (
            frames[0].permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        )

        frames = frames / 127.5 - 1.0
        frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)

        return frames, first_frame_numpy
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None, None


def load_video_key_frames(video_path, max_frames=31, target_size=(512, 768)):
    """Load first/mid/last key frames from a video.

    Returns:
        key_frames: (1, C, 3, H, W) float tensor in [-1, 1]
        key_frames_numpy: (3, H, W, C) uint8 numpy
    """
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = min(max_frames, len(vr))

        orig_h, orig_w = vr[0].shape[:2]
        target_h, target_w = target_size

        indices = [0, total_frames // 2, total_frames - 1]
        raw_frames = vr.get_batch(indices).asnumpy()  # (3, H, W, C)

        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h
        if orig_aspect > target_aspect:
            crop_w = int(orig_h * target_aspect)
            crop_h = orig_h
            start_w = (orig_w - crop_w) // 2
            start_h = 0
        else:
            crop_h = int(orig_w / target_aspect)
            crop_w = orig_w
            start_h = (orig_h - crop_h) // 2
            start_w = 0

        raw_frames = raw_frames[
            :, start_h : start_h + crop_h, start_w : start_w + crop_w, :
        ]

        frames_t = torch.from_numpy(raw_frames).float().permute(0, 3, 1, 2)
        resize_transform = transforms.Resize((target_h, target_w), antialias=True)
        frames_t = resize_transform(frames_t)

        key_frames_numpy = (
            frames_t.permute(0, 2, 3, 1).clamp(0, 255).cpu().numpy().astype(np.uint8)
        )

        frames_t = frames_t / 127.5 - 1.0
        key_frames = frames_t.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, 3, H, W)

        return key_frames, key_frames_numpy
    except Exception as e:
        print(f"Error loading key frames {video_path}: {e}")
        return None, None


def load_image(image_path, target_size=(1024, 1024)):
    """Load and resize an image.

    Returns:
        image_tensor: (1, C, 1, H, W) float tensor in [-1, 1]
        image_numpy: (H, W, C) uint8 numpy
    """
    try:
        img = Image.open(image_path).convert("RGB")
        target_h, target_w = target_size
        img = img.resize((target_w, target_h), Image.LANCZOS)
        img_np = np.array(img)
        img_t = torch.from_numpy(img_np).float().permute(2, 0, 1)  # (C, H, W)
        img_t_normalized = img_t / 127.5 - 1.0
        img_t_normalized = img_t_normalized.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
        return img_t_normalized, img_np
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None


class MultiResDataset(Dataset):
    """Multi-resolution multi-task dataset for Kandinsky-5 unified training.

    CSV format for tv2v:
        video1_path, video2_path, editing_instruction, video2_caption,
        bucket, frames, task_type, data_type

    CSV format for ti2i:
        img1_path, img2_path, editing_instruction, img2_caption,
        bucket, task_type, data_type

    CSV format for t2v:
        video_path, caption, bucket, frames, task_type, data_type

    CSV format for t2i:
        img_path, caption, bucket, task_type, data_type
    """

    def __init__(
        self,
        csv_path: Union[str, List[str]],
        data_root: Union[str, List[str], None] = None,
        max_frames: int = 31,
        **kwargs,
    ):
        self.max_frames = max_frames

        if isinstance(csv_path, str):
            csv_path = [csv_path]
        if data_root is None or isinstance(data_root, str):
            data_root = [data_root] if data_root is not None else [None]
        if len(data_root) == 1:
            data_root = data_root * len(csv_path)
        assert len(csv_path) == len(data_root)

        self.data_list = []
        self.video_data_list = []
        self.image_data_list = []
        self.gen_video_data_list = []
        self.gen_image_data_list = []

        self.video_bucket_to_indices = defaultdict(list)
        self.image_bucket_to_indices = defaultdict(list)
        self.gen_video_bucket_to_indices = defaultdict(list)
        self.gen_image_bucket_to_indices = defaultdict(list)

        self.task_bucket_to_indices = defaultdict(list)
        self.task_buckets = defaultdict(list)

        self.video_bucket_frames_to_indices = defaultdict(list)
        self.task_bucket_frames_to_indices = defaultdict(list)
        self.image_bucket_frames_to_indices = defaultdict(list)

        self._load_annotations(csv_path, data_root)

        self.video_buckets = sorted(self.video_bucket_to_indices.keys())
        self.image_buckets = sorted(self.image_bucket_to_indices.keys())
        self.gen_video_buckets = sorted(self.gen_video_bucket_to_indices.keys())
        self.gen_image_buckets = sorted(self.gen_image_bucket_to_indices.keys())
        for task in self.task_buckets:
            self.task_buckets[task] = sorted(self.task_buckets[task])

        if is_main_process():
            print(f"Loaded {len(self.video_data_list)} video edit, "
                  f"{len(self.image_data_list)} image edit, "
                  f"{len(self.gen_video_data_list)} video gen, "
                  f"{len(self.gen_image_data_list)} image gen samples. "
                  f"Total: {len(self.data_list)}")
            for task in sorted(self.task_buckets.keys()):
                buckets = self.task_buckets[task]
                total = sum(
                    len(self.task_bucket_to_indices[(task, b)]) for b in buckets
                )
                print(f"  Task '{task}': {total} samples across {len(buckets)} buckets")

    def _load_annotations(self, csv_paths, data_roots):
        for i, path in enumerate(csv_paths):
            file_ext = Path(path).suffix.lower()
            if file_ext == ".csv":
                df = pd.read_csv(path)
            elif file_ext in (".parquet", ".pq"):
                df = pd.read_parquet(path, engine="pyarrow")
            else:
                raise ValueError(f"Unsupported format: {file_ext}")

            has_data_type = "data_type" in df.columns
            has_task_type = "task_type" in df.columns

            if not has_data_type:
                if "video1_path" in df.columns or "video_path" in df.columns:
                    default_data_type = "video"
                elif "img1_path" in df.columns or "img_path" in df.columns:
                    default_data_type = "image"
                else:
                    raise ValueError(f"Cannot determine data type from {path}")
            else:
                default_data_type = None

            default_task_type = "tv2v" if not has_task_type else None

            for _, row in df.iterrows():
                data_type = (
                    str(row["data_type"]).lower().strip()
                    if has_data_type
                    else default_data_type
                )
                task_type = (
                    str(row["task_type"]).lower().strip()
                    if has_task_type
                    else default_task_type
                )

                if task_type not in ("tv2v", "ti2i", "t2v", "t2i", "style_transfer", "prop"):
                    continue

                data_dict = None
                if data_type == "video":
                    if task_type in ("tv2v", "style_transfer", "prop"):
                        data_dict = self._load_tv2v(row, data_roots[i])
                    elif task_type == "t2v":
                        data_dict = self._load_t2v(row, data_roots[i])
                elif data_type == "image":
                    if task_type == "ti2i":
                        data_dict = self._load_ti2i(row, data_roots[i])
                    elif task_type == "t2i":
                        data_dict = self._load_t2i(row, data_roots[i])

                if data_dict is None:
                    continue

                data_dict["index"] = len(self.data_list)
                self.data_list.append(data_dict)

                task = data_dict["task"]
                bucket = data_dict["bucket"]
                frames = data_dict.get("frames", 1)
                idx = data_dict["index"]

                task_bucket_key = (task, bucket)
                self.task_bucket_to_indices[task_bucket_key].append(idx)
                if bucket not in self.task_buckets[task]:
                    self.task_buckets[task].append(bucket)

                task_bucket_frames_key = (task, bucket, frames)
                self.task_bucket_frames_to_indices[task_bucket_frames_key].append(idx)

                if task in ("tv2v", "style_transfer", "prop"):
                    self.video_data_list.append(data_dict)
                    self.video_bucket_to_indices[bucket].append(idx)
                    self.video_bucket_frames_to_indices[(bucket, frames)].append(idx)
                elif task == "ti2i":
                    self.image_data_list.append(data_dict)
                    self.image_bucket_to_indices[bucket].append(idx)
                    self.image_bucket_frames_to_indices[(bucket, frames)].append(idx)
                elif task == "t2v":
                    self.gen_video_data_list.append(data_dict)
                    self.gen_video_bucket_to_indices[bucket].append(idx)
                elif task == "t2i":
                    self.gen_image_data_list.append(data_dict)
                    self.gen_image_bucket_to_indices[bucket].append(idx)

    def _load_tv2v(self, row, data_root):
        bucket = row.get("bucket", None)
        video1_path = row.get("video1_path", None)
        video2_path = row.get("video2_path", None)
        instruction = row.get("editing_instruction", None) or row.get("instruction", None)
        video_caption = row.get("video2_caption", None)
        frames = row.get("frames", None)
        task_type = row.get("task_type", "tv2v")

        if pd.isna(video1_path) or pd.isna(video2_path) or pd.isna(instruction):
            return None

        if video_caption is not None and not pd.isna(video_caption):
            r = random.random()
            if r < 0.2:
                instruction = video_caption
            elif r < 0.6:
                pass
            else:
                instruction = f"{instruction} {video_caption}"

        frames = self.max_frames if pd.isna(frames) else int(frames)
        task = str(task_type).lower().strip() if not pd.isna(task_type) else "tv2v"
        if task not in ("tv2v", "style_transfer", "prop"):
            task = "tv2v"

        return {
            "type": "video",
            "video1_path": os.path.join(data_root, video1_path) if data_root else video1_path,
            "video2_path": os.path.join(data_root, video2_path) if data_root else video2_path,
            "instruction": instruction,
            "bucket": str(bucket),
            "frames": frames,
            "task": task,
        }

    def _load_ti2i(self, row, data_root):
        bucket = row.get("bucket", None)
        img1_path = row.get("img1_path", None)
        img2_path = row.get("img2_path", None)
        instruction = row.get("editing_instruction", None) or row.get("instruction", None)
        img_caption = row.get("img2_caption", None)

        if pd.isna(img1_path) or pd.isna(img2_path) or pd.isna(instruction):
            return None

        if img_caption is not None and not pd.isna(img_caption):
            r = random.random()
            if r < 0.2:
                instruction = img_caption
            elif r < 0.6:
                pass
            else:
                instruction = f"{img_caption} {instruction}"

        return {
            "type": "image",
            "img1_path": os.path.join(data_root, img1_path) if data_root else img1_path,
            "img2_path": os.path.join(data_root, img2_path) if data_root else img2_path,
            "instruction": instruction,
            "bucket": str(bucket),
            "frames": 1,
            "task": "ti2i",
        }

    def _load_t2v(self, row, data_root):
        bucket = row.get("bucket", None)
        video_path = row.get("video_path", None)
        caption = row.get("caption", None)
        frames = row.get("frames", None)

        if pd.isna(video_path) or pd.isna(caption):
            return None

        frames = self.max_frames if pd.isna(frames) else int(frames)

        return {
            "type": "gen_video",
            "video_path": os.path.join(data_root, video_path) if data_root else video_path,
            "instruction": caption,
            "bucket": str(bucket),
            "frames": frames,
            "task": "t2v",
        }

    def _load_t2i(self, row, data_root):
        bucket = row.get("bucket", None)
        img_path = row.get("img_path", None)
        caption = row.get("caption", None)

        if pd.isna(img_path) or pd.isna(caption):
            return None

        return {
            "type": "gen_image",
            "img_path": os.path.join(data_root, img_path) if data_root else img_path,
            "instruction": caption,
            "bucket": str(bucket),
            "frames": 1,
            "task": "t2i",
        }

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        task = data["task"]

        if task in ("tv2v", "style_transfer", "prop"):
            return self._get_video_item(data)
        elif task == "ti2i":
            return self._get_image_item(data)
        elif task == "t2v":
            return self._get_gen_video_item(data)
        elif task == "t2i":
            return self._get_gen_image_item(data)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _get_video_item(self, data):
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket: {data['bucket']}")

        max_frames = data.get("frames", self.max_frames)
        if not isinstance(max_frames, int):
            max_frames = self.max_frames

        video1_tensor, _ = load_video_frames(
            data["video1_path"], target_size=target_size, max_frames=max_frames,
        )
        video2_tensor, _ = load_video_frames(
            data["video2_path"], target_size=target_size, max_frames=max_frames,
        )
        if video1_tensor is None or video2_tensor is None:
            raise ValueError(f"Failed to load videos for {data['video1_path']}")

        _, video1_key_frames_numpy = load_video_key_frames(
            data["video1_path"], target_size=target_size, max_frames=max_frames,
        )
        if video1_key_frames_numpy is None:
            raise ValueError(f"Failed to load key frames from {data['video1_path']}")

        _, video2_key_frames_numpy = load_video_key_frames(
            data["video2_path"], target_size=target_size, max_frames=max_frames,
        )

        return {
            "type": "video",
            "task": data["task"],
            "video1": video1_tensor.squeeze(0),  # (C, T, H, W)
            "video2": video2_tensor.squeeze(0),  # (C, T, H, W)
            "video1_key_frames": video1_key_frames_numpy,  # (3, H, W, C) uint8
            "video2_key_frames": video2_key_frames_numpy,  # (3, H, W, C) uint8, Hunyuan-style for prop
            "bucket": data["bucket"],
            "instruction": data["instruction"],
            "index": data["index"],
        }

    def _get_image_item(self, data):
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket: {data['bucket']}")

        img1_tensor, img1_numpy = load_image(data["img1_path"], target_size=target_size)
        img2_tensor, img2_numpy = load_image(data["img2_path"], target_size=target_size)
        if img1_tensor is None or img2_tensor is None:
            raise ValueError(f"Failed to load images for {data['img1_path']}")

        return {
            "type": "image",
            "task": data["task"],
            "img1": img1_tensor.squeeze(0),  # (C, 1, H, W)
            "img2": img2_tensor.squeeze(0),  # (C, 1, H, W)
            "img1_numpy": img1_numpy,  # (H, W, C) uint8
            "bucket": data["bucket"],
            "instruction": data["instruction"],
            "index": data["index"],
        }

    def _get_gen_video_item(self, data):
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket: {data['bucket']}")

        max_frames = data.get("frames", self.max_frames)
        if not isinstance(max_frames, int):
            max_frames = self.max_frames

        video_tensor, _ = load_video_frames(
            data["video_path"], target_size=target_size, max_frames=max_frames,
        )
        if video_tensor is None:
            raise ValueError(f"Failed to load video {data['video_path']}")

        return {
            "type": "gen_video",
            "task": "t2v",
            "video": video_tensor.squeeze(0),  # (C, T, H, W)
            "bucket": data["bucket"],
            "instruction": data["instruction"],
            "index": data["index"],
        }

    def _get_gen_image_item(self, data):
        target_size = parse_bucket(data["bucket"])
        if target_size is None:
            raise ValueError(f"Invalid bucket: {data['bucket']}")

        img_tensor, _ = load_image(data["img_path"], target_size=target_size)
        if img_tensor is None:
            raise ValueError(f"Failed to load image {data['img_path']}")

        return {
            "type": "gen_image",
            "task": "t2i",
            "image": img_tensor.squeeze(0),  # (C, 1, H, W)
            "bucket": data["bucket"],
            "instruction": data["instruction"],
            "index": data["index"],
        }

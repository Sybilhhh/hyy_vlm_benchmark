import json
import logging
import struct
from typing import List, Optional, Union

import decord
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import transformers
from PIL import Image
from peft import (
    LoraConfig,
    PeftConfig,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pil_to_tensor

from .generation_utils import (
    encode_video,
    generate_unified_sample,
    resize_video,
)

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True

# Support all unified tasks: t2v / t2i / ti2i / tv2v / i2v / prop / keyframe
SUPPORTED_TASK_TYPES = ("t2v", "t2i", "ti2i", "tv2v", "i2v", "prop", "keyframe")

VIDEO_RESOLUTIONS = {
    512: [(512, 512), (512, 768), (768, 512)],
    1024: [
        (1024, 1024), (640, 1408), (1408, 640),
        (768, 1280), (1280, 768), (896, 1152), (1152, 896),
    ],
}

IMAGE_RESOLUTIONS = {
    1024: [
        (1024, 1024), (640, 1408), (1408, 640),
        (768, 1280), (1280, 768), (896, 1152), (1152, 896),
    ],
}

DEFAULT_NEGATIVE_VIDEO = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
    "worst quality, low quality, ugly, deformed, walking backwards"
)
DEFAULT_NEGATIVE_IMAGE = ""


def _find_nearest_resolution(available_res, real_res):
    nearest_index = np.argmin(
        [abs((x[0] / x[1]) - (real_res[0] / real_res[1])) for x in available_res]
    )
    return available_res[nearest_index]


def load_video_frames(video_path, target_size=(512, 768), max_frames=31):
    """Load video frames using decord, center-crop and resize to target_size.

    Returns:
        frames: (T, H, W, C) float tensor in [-1, 1]
        frames_numpy: (T, H, W, C) uint8 numpy for Qwen multimodal input
    """
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    n_frames = min(total_frames, max_frames)
    frame_indices = list(range(n_frames))
    frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

    target_h, target_w = target_size
    orig_h, orig_w = frames.shape[1], frames.shape[2]
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
    frames_numpy = frames.copy()

    frames_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2)  # (T, C, H, W)
    frames_t = torch.nn.functional.interpolate(
        frames_t, size=(target_h, target_w), mode="bilinear", align_corners=False,
    )

    frames_numpy_resized = frames_t.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()
    frames_normalized = frames_t / 127.5 - 1.0  # (T, C, H, W)
    frames_normalized = frames_normalized.permute(0, 2, 3, 1)  # (T, H, W, C)

    return frames_normalized, frames_numpy_resized


def load_video_key_frames(video_path, target_size=(512, 768), max_frames=31):
    """Load first/mid/last key frames from a video.

    Returns:
        key_frames: (3, H, W, C) float tensor in [-1, 1]
        key_frames_numpy: (3, H, W, C) uint8 numpy
    """
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = min(len(vr), max_frames)
    indices = [0, total_frames // 2, total_frames - 1]

    target_h, target_w = target_size
    orig_h, orig_w = vr[0].shape[:2]
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

    key_frames = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        frame = frame[start_h : start_h + crop_h, start_w : start_w + crop_w, :]
        key_frames.append(frame)

    key_frames = np.stack(key_frames, axis=0)  # (3, H, W, C)
    frames_t = torch.from_numpy(key_frames).float().permute(0, 3, 1, 2)
    frames_t = torch.nn.functional.interpolate(
        frames_t, size=(target_h, target_w), mode="bilinear", align_corners=False,
    )

    key_frames_numpy = frames_t.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()
    key_frames_normalized = (frames_t / 127.5 - 1.0).permute(0, 2, 3, 1)

    return key_frames_normalized, key_frames_numpy


def _read_safetensors_json(file_path):
    with open(file_path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("Q", header_size_bytes)[0]
        header_bytes = f.read(header_size)
        header_str = header_bytes.decode("utf-8")
        header = json.loads(header_str)
        return header


class Kandinsky5UnifiedPipeline:
    """Unified inference pipeline for Kandinsky-5 supporting t2v, t2i, ti2i, tv2v.

    All tasks use a single DiT with instruct_type='channel' that accepts
    33-channel input: [noise(16) + cond_latent(16) + mask(1)].
    """

    def __init__(
        self,
        device_map: Union[str, torch.device, dict],
        dit,
        text_embedder,
        vae,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf=None,
        offload: bool = False,
        device_mesh: DeviceMesh = None,
    ):
        if not isinstance(device_map, dict):
            device_map = {
                "dit": device_map,
                "vae": device_map,
                "text_embedder": device_map,
            }

        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae

        self.device_map = device_map
        self.local_dit_rank = local_dit_rank
        self.world_size = world_size
        self.conf = conf
        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self.offload = offload
        self.device_mesh = device_mesh

        self._hf_peft_config_loaded = False
        self.peft_config = {}
        self.peft_triggers = {}
        self.peft_trigger = ""

    def _get_default_resolution(self, task_type):
        # Video-style tasks: t2v, tv2v, i2v, prop, keyframe
        if task_type in ("t2v", "tv2v", "i2v", "prop", "keyframe"):
            return VIDEO_RESOLUTIONS
        else:
            return IMAGE_RESOLUTIONS

    def _get_default_negative(self, task_type):
        if task_type in ("t2v", "tv2v", "i2v", "prop", "keyframe"):
            return DEFAULT_NEGATIVE_VIDEO
        return DEFAULT_NEGATIVE_IMAGE

    def _get_default_scheduler_scale(self, task_type):
        if task_type in ("t2v", "tv2v", "i2v", "prop", "keyframe"):
            return 10.0
        return 3.0

    def _encode_source_image(self, image, height, width):
        """Encode a source image to VAE latent space for ti2i.
        Uses the HunyuanVideo 3D VAE with T=1."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        img_tensor = pil_to_tensor(image)[None]  # (1, C, H, W)
        img_tensor = resize_video(img_tensor, (height * 8, width * 8))
        img_tensor = img_tensor.to(
            device=self.device_map["vae"], dtype=torch.bfloat16,
        ) / 127.5 - 1.0
        # (1, C, H, W) -> (1, C, 1, H, W) for 3D VAE
        with torch.no_grad():
            cond_latent = encode_video(img_tensor[:, :, None], self.vae, False).squeeze(0)
        return cond_latent, img_tensor

    def _encode_source_video(self, video_path, height, width, num_frames):
        """Encode a source video to VAE latent space for tv2v/prop/keyframe."""
        target_size = (height * 8, width * 8)
        frames, _ = load_video_frames(video_path, target_size=target_size, max_frames=num_frames)
        frames_bcthw = frames.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W) -> need (1, C, T, H, W)
        frames_bcthw = frames_bcthw.permute(0, 2, 1, 3, 4)
        frames_bcthw = frames_bcthw.to(
            device=self.device_map["vae"], dtype=torch.bfloat16,
        )
        with torch.no_grad():
            cond_latent = encode_video(frames_bcthw, self.vae, False).squeeze(0)
        return cond_latent

    def _load_first_frame_from_video(self, video_path, height, width):
        """Load first frame from video as PIL Image (used for prop guide_video)."""
        target_size = (height * 8, width * 8)
        frames, _ = load_video_frames(video_path, target_size=target_size, max_frames=1)
        frame0 = frames[0]
        arr = (np.clip((frame0.cpu().numpy() * 0.5 + 0.5) * 255, 0, 255)).astype(np.uint8)
        return Image.fromarray(arr)

    def expand_prompt_video(self, prompt):
        """Expand a text prompt for video generation using Qwen."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            'You are a prompt beautifier that transforms short user video '
                            'descriptions into rich, detailed English prompts specifically '
                            'optimized for video generation models.\n'
                            'Rewrite Prompt: "' + prompt + '". Answer only with expanded prompt.'
                        ),
                    },
                ],
            }
        ]
        text = self.text_embedder.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.text_embedder.embedder.processor(
            text=[text], images=None, videos=None, padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.text_embedder.embedder.model.device)
        generated_ids = self.text_embedder.embedder.model.generate(
            **inputs, max_new_tokens=256,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_embedder.embedder.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def expand_prompt_image_edit(self, prompt, image):
        """Expand an editing prompt using Qwen with the source image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        w, h = image.size
        image_small = image.resize((w // 4, h // 4))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            'Rewrite and enhance the original editing instruction with richer '
                            'detail, clearer structure, and improved descriptive quality. '
                            'When adding text that should appear inside an image, place that '
                            'text inside double quotes and in capital letters. Explain what '
                            'needs to be changed and what needs to be left unchanged.\n'
                            'Rewrite Prompt: "' + prompt + '". Answer only with expanded prompt.'
                        ),
                    },
                    {"type": "image", "image": image_small},
                ],
            }
        ]
        text = self.text_embedder.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.text_embedder.embedder.processor(
            text=[text], images=image_small, videos=None,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.text_embedder.embedder.model.device)
        generated_ids = self.text_embedder.embedder.model.generate(
            **inputs, max_new_tokens=256,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_embedder.embedder.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def __call__(
        self,
        text: str,
        task_type: str = "t2v",
        image: Optional[Union[str, Image.Image]] = None,
        video: Optional[str] = None,
        guide_image: Optional[Union[str, Image.Image]] = None,
        guide_video: Optional[str] = None,
        time_length: int = 5,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_weight: Optional[float] = None,
        scheduler_scale: Optional[float] = None,
        negative_caption: Optional[str] = None,
        expand_prompts: bool = True,
        save_path: Optional[str] = None,
        progress: bool = True,
    ):
        """Unified inference entry point.

        Args:
            text: text prompt or editing instruction
            task_type: "t2v" | "t2i" | "ti2i" | "tv2v" | "i2v" | "prop" | "keyframe"
            image: source image for ti2i / keyframe (path or PIL Image)
            video: source video path for tv2v / keyframe / prop
            guide_image: guide keyframe image for prop (path or PIL Image); first frame is fixed to this
            guide_video: optional guide video for prop; first frame used as keyframe when guide_image is None
            time_length: video duration in seconds (ignored for image tasks)
            width, height: output resolution (auto-detected for ti2i from image)
            seed: random seed
            num_steps: denoising steps (default from config)
            guidance_weight: CFG weight (default from config)
            scheduler_scale: timestep warping (default per task)
            negative_caption: negative prompt (default per task)
            expand_prompts: use Qwen to expand the prompt
            save_path: path to save output
            progress: show tqdm progress bar
        """
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {SUPPORTED_TASK_TYPES}, got '{task_type}'"
            )

        if task_type == "ti2i" and image is None:
            raise ValueError("ti2i requires an 'image' argument")
        if task_type == "tv2v" and video is None:
            raise ValueError("tv2v requires a 'video' argument")
        if task_type in ("prop", "keyframe"):
            if video is None:
                raise ValueError("prop/keyframe requires a 'video' argument")
            # 至少需要 guide_image 或 guide_video 之一；两个都为 None 才报错
            if guide_image is None and guide_video is None:
                raise ValueError("prop/keyframe requires a 'guide_image' or 'guide_video' argument (keyframe for first frame)")

        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        if scheduler_scale is None:
            scheduler_scale = self._get_default_scheduler_scale(task_type)
        if negative_caption is None:
            negative_caption = self._get_default_negative(task_type)

        # Seed handling
        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**32 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)
            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)
            seed = seed.item()

        resolutions = self._get_default_resolution(task_type)
        resolution_key = 1024 if task_type in ("t2i", "ti2i") else self.conf.metrics.resolution

        # Resolve dimensions
        if task_type in ("t2i", "ti2i"):
            num_frames = 1
            if height is None or width is None:
                if task_type == "ti2i" and image is not None:
                    if isinstance(image, str):
                        pil_img = Image.open(image).convert("RGB")
                    else:
                        pil_img = image
                    height, width = _find_nearest_resolution(
                        resolutions[resolution_key], pil_img.size[::-1],
                    )
                else:
                    if height is None:
                        height = 480
                    if width is None:
                        width = 832
        else:
            num_frames = 1 if time_length == 0 else time_length * 24 // 4 + 1
            if width is None:
                width = 768 if resolution_key == 512 else 1024
            if height is None:
                height = 512 if resolution_key == 512 else 1024

        spatial_div = 8
        shape = (1, num_frames, height // spatial_div, width // spatial_div, 16)

        # Encode conditioning
        cond_latent = None
        images_for_text_encoder = None
        keyframe_idx = None

        if task_type == "ti2i":
            if self.offload:
                self.vae.to(self.device_map["vae"])
            cond_latent, img_tensor = self._encode_source_image(
                image, height // spatial_div, width // spatial_div,
            )
            if self.offload:
                self.vae.to("cpu")
            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            else:
                pil_img = image
            images_for_text_encoder = [resize_video(pil_to_tensor(pil_img)[None], (height, width))]

        elif task_type == "tv2v":
            if self.offload:
                self.vae.to(self.device_map["vae"])
            cond_latent = self._encode_source_video(
                video, height // spatial_div, width // spatial_div, num_frames,
            )
            if self.offload:
                self.vae.to("cpu")

        elif task_type == "keyframe":
            # Keyframe propagation: encode source video as cond_latent, then
            # overwrite a single temporal slice with the keyframe image latent.
            if self.offload:
                self.vae.to(self.device_map["vae"])
            # Source video latent (conditioning backbone)
            cond_latent = self._encode_source_video(
                video, height // spatial_div, width // spatial_div, num_frames,
            )
            # Keyframe image latent (T=1)
            guide_latent, _ = self._encode_source_image(
                guide_image, height // spatial_div, width // spatial_div,
            )
            latent_T = cond_latent.shape[0]
            # Try to infer frame index from filename pattern "..._idxXX.png"
            frame_idx = None
            if isinstance(guide_image, str) and "_idx" in guide_image:
                try:
                    frame_idx_str = guide_image.split("_idx")[-1]
                    frame_idx_str = frame_idx_str.split(".")[0]
                    frame_idx = int(frame_idx_str)
                except Exception:
                    frame_idx = None
            if frame_idx is not None:
                # Map original frame index to latent index (temporal downsample ~4x)
                keyframe_idx = (frame_idx + 3) // 4
            else:
                keyframe_idx = latent_T // 2
            keyframe_idx = max(0, min(latent_T - 1, int(keyframe_idx)))
            cond_latent[keyframe_idx : keyframe_idx + 1] = guide_latent.to(
                device=cond_latent.device, dtype=cond_latent.dtype
            )  # TODO:check this, check dim
            if self.offload:
                self.vae.to("cpu")

        elif task_type == "prop":
            # Propagation with explicit guide:
            # - encode source video as cond_latent (backbone motion/structure)
            # - encode guide image/video first frame as a separate guide_latent
            if self.offload:
                self.vae.to(self.device_map["vae"])
            cond_latent = self._encode_source_video(
                video, height // spatial_div, width // spatial_div, num_frames,
            )
            guide_img = guide_image
            if guide_img is None and guide_video is not None:
                guide_img = self._load_first_frame_from_video(
                    guide_video, height // spatial_div, width // spatial_div,
                )
            guide_latent, _ = self._encode_source_image(
                guide_img, height // spatial_div, width // spatial_div,
            )
            if self.offload:
                self.vae.to("cpu")

        # Prompt expansion
        caption = text
        if expand_prompts:
            transformers.set_seed(seed)
            if self.local_dit_rank == 0:
                if self.offload:
                    self.text_embedder = self.text_embedder.to(
                        self.device_map["text_embedder"],
                    )
                if task_type == "ti2i" and image is not None:
                    caption = self.expand_prompt_image_edit(caption, image=image)
                elif task_type in ("tv2v", "i2v", "prop", "keyframe"):
                    caption = self.expand_prompt_video(caption)
                else:
                    caption = self.peft_trigger + self.expand_prompt_video(caption)
            if self.world_size > 1:
                caption = [caption]
                torch.distributed.broadcast_object_list(caption, 0)
                caption = caption[0]

        # Generate
        output = generate_unified_sample(
            shape,
            caption,
            self.dit,
            self.vae,
            self.conf,
            text_embedder=self.text_embedder,
            task_type=task_type,
            cond_latent=cond_latent,
            guide_latent=guide_latent if task_type in ("prop", "keyframe") else None,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            seed=seed,
            device=self.device_map["dit"],
            vae_device=self.device_map["vae"],
            text_embedder_device=self.device_map["text_embedder"],
            progress=progress,
            offload=self.offload,
            images_for_text_encoder=images_for_text_encoder,
            tp_mesh=self.device_mesh,
            keyframe_idx=keyframe_idx if task_type == "keyframe" else None,
        )
        torch.cuda.empty_cache()

        if self.offload:
            self.text_embedder = self.text_embedder.to(
                device=self.device_map["text_embedder"],
            )

        # Post-process and save
        if self.local_dit_rank == 0:
            if task_type in ("t2i", "ti2i"):
                return_images = []
                for img in output.cpu():
                    return_images.append(ToPILImage()(img))
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    for path, img in zip(save_path, return_images):
                        img.save(path)
                return return_images
            else:
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    for path, vid in zip(save_path, output):
                        torchvision.io.write_video(
                            path,
                            vid.float().permute(1, 2, 3, 0).cpu().numpy(),
                            fps=24,
                            options={"crf": "5"},
                        )
                return output

    # ---- LoRA Adapter Support ----

    def load_adapter(
        self,
        adapter_config: Union[PeftConfig, str],
        adapter_path: Optional[str] = None,
        adapter_name: Optional[str] = None,
        trigger: Optional[str] = None,
    ) -> None:
        if adapter_name is None:
            adapter_name = "default"
        if self._hf_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} already exists. "
                "Please use a different name."
            )

        if not isinstance(adapter_config, PeftConfig):
            try:
                with open(adapter_config, "r") as f:
                    adapter_config = json.load(f)
                adapter_config = LoraConfig(**adapter_config)
            except Exception:
                raise TypeError(
                    "adapter_config should be a PeftConfig or path to a JSON file."
                )
        self.peft_config[adapter_name] = adapter_config

        inject_adapter_in_model(adapter_config, self.dit, adapter_name)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        adapter_state_dict = load_file(adapter_path)
        adapter_metadata = _read_safetensors_json(adapter_path)
        if trigger is not None:
            self.peft_trigger = trigger
        else:
            if (
                "__metadata__" in adapter_metadata
                and "trigger" in adapter_metadata["__metadata__"]
            ):
                self.peft_trigger = adapter_metadata["__metadata__"]["trigger"]
            else:
                self.peft_trigger = ""
        self.peft_triggers[adapter_name] = self.peft_trigger

        processed_adapter_state_dict = {}
        for key, value in adapter_state_dict.items():
            new_key = key
            for prefix in ["base_model.model.", "transformer."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            new_key = new_key.replace(".default", "")
            processed_adapter_state_dict[new_key] = value

        incompatible_keys = set_peft_model_state_dict(
            self.dit, processed_adapter_state_dict, adapter_name,
        )

        if incompatible_keys is not None:
            err_msg = ""
            if (
                hasattr(incompatible_keys, "unexpected_keys")
                and len(incompatible_keys.unexpected_keys) > 0
            ):
                err_msg = (
                    "Loading adapter weights led to unexpected keys: "
                    f"{', '.join(incompatible_keys.unexpected_keys)}. "
                )
            missing_keys = getattr(incompatible_keys, "missing_keys", None)
            if missing_keys:
                lora_missing = [
                    k for k in missing_keys if "lora_" in k and adapter_name in k
                ]
                if lora_missing:
                    err_msg += (
                        "Loading adapter weights led to missing keys: "
                        f"{', '.join(lora_missing)}"
                    )
            if err_msg:
                logging.warning(err_msg)

        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[list, str]) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        if isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if missing:
                raise ValueError(
                    f"Adapter(s) not found: {', '.join(missing)}. "
                    f"Loaded: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter '{adapter_name}' not found. "
                f"Available: {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _set = False
        for _, module in self.dit.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    module.disable_adapters = False
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _set = True

        if not _set:
            raise ValueError(
                "Failed to set adapter. Ensure the model supports adapters."
            )
        self.peft_trigger = self.peft_triggers[adapter_name]

    def disable_adapters(self) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.dit.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    module.disable_adapters = True
        self.peft_trigger = ""

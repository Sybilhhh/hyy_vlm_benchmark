import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import torch
from torch.distributed import all_gather
from tqdm import tqdm 

from .models.utils import fast_sta_nabla
import torchvision.transforms.functional as F


_DEBUG_PROP_KEYFRAME = os.environ.get("K5_DEBUG_PROP_KEYFRAME", "0") == "1"
_DEBUG_PROP_KEYFRAME_DONE = False


def get_sparse_params(conf, batch_embeds, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, _ = batch_embeds["visual"].shape
    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )
    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(T, H // 8, W // 8, conf.model.attention.wT,
                                  conf.model.attention.wH, conf.model.attention.wW, device=device)
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
            "wT": conf.model.attention.wT,
            "wW": conf.model.attention.wW,
            "wH": conf.model.attention.wH,
            "add_sta": conf.model.attention.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(conf.model.attention, "method", "topcdf"),
        }
    else:
        sparse_params = None

    return sparse_params

def adaptive_mean_std_normalization(source, reference):
    source_mean = source.mean(dim=(1,2,3),keepdim=True)
    source_std = source.std(dim=(1,2,3),keepdim=True)
    #magic constants - limit changes in latents
    clump_mean_low = 0.05
    clump_mean_high = 0.1
    clump_std_low = 0.1
    clump_std_high = 0.25

    reference_mean = torch.clamp(reference.mean(), source_mean - clump_mean_low, source_mean + clump_mean_high)
    reference_std = torch.clamp(reference.std(), source_std - clump_std_low, source_std + clump_std_high)

    # normalization
    normalized = (source - source_mean) / source_std
    normalized = normalized * reference_std + reference_mean
    
    return normalized

def normalize_first_frame(latents, reference_frames=5, clump_values=False):
    latents_copy = latents.clone()
    samples = latents_copy
    
    if samples.shape[0] <= 1:
        return (latents, "Only one frame, no normalization needed")
    nFr = 4
    first_frames = samples[:nFr]
    reference_frames_data = samples[nFr:nFr+min(reference_frames, samples.shape[0]-1)]
    
    # print("First frame stats - Mean:", first_frames.mean(dim=(1,2,3)), "Std: ", first_frames.std(dim=(1,2,3)))
    # print(f"Reference frames stats - Mean: {reference_frames_data.mean().item():.4f}, Std: {reference_frames_data.std().item():.4f}")
    
    normalized_first = adaptive_mean_std_normalization(first_frames, reference_frames_data)
    if clump_values:
        min_val = reference_frames_data.min()
        max_val = reference_frames_data.max()
        normalized_first = torch.clamp(normalized_first, min_val, max_val)
    
    samples[:nFr] = normalized_first
    
    return samples

@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
    attention_mask=None,
    null_attention_mask=None,
):
    with torch._dynamo.utils.disable_cache_limit():
        pred_velocity = dit(
            x,
            text_embeds["text_embeds"],
            text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
        )
        if abs(guidance_weight - 1.0) > 1e-6:
            uncond_pred_velocity = dit(
                x,
                null_text_embeds["text_embeds"],
                null_text_embeds["pooled_embed"],
                t * 1000,
                visual_rope_pos,
                null_text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=null_attention_mask,
            )
            pred_velocity = uncond_pred_velocity + guidance_weight * (
                pred_velocity - uncond_pred_velocity
            )
    return pred_velocity


@torch.no_grad()
def generate(
    model,
    device,
    img,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    first_frames,
    conf,
    progress=False,
    seed=6554,
    tp_mesh=None,
    attention_mask=None,
    null_attention_mask=None,
    prop_first_frame_latent=None,
):
    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    if tp_mesh and first_frames is None: # do not split on gpus for i2v
        tp_rank = tp_mesh["tensor_parallel"].get_local_rank()
        tp_world_size = tp_mesh["tensor_parallel"].size()
        img = torch.chunk(img, tp_world_size, dim=1)[tp_rank]

    for timestep, timestep_diff in tqdm(list(zip(timesteps[:-1], torch.diff(timesteps)))):
        time = timestep.unsqueeze(0)
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            if first_frames is not None:
                first_frames = first_frames.to(device=visual_cond.device, dtype=visual_cond.dtype)
                img[:1] = first_frames
                visual_cond_mask[:1] = 1
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        pred_velocity = get_velocity(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
        )
        # Prop: do not update first frame so it stays exactly the guide latent (avoids blur/mosaic)
        if prop_first_frame_latent is not None:
            pred_velocity = pred_velocity.clone()
            pred_velocity[0] = 0
        img[..., :pred_velocity.shape[-1]] += timestep_diff * pred_velocity
        # Prop: re-pin first frame every step as safeguard
        if prop_first_frame_latent is not None:
            img[0, :, :, 0 : pred_velocity.shape[-1]] = prop_first_frame_latent
        # NOTE: remove extra channels that can be added in Image Editing (I2I)
    out = img[..., :pred_velocity.shape[-1]]
    # Prop: guarantee first frame is guide latent at return (no drift before decode)
    if prop_first_frame_latent is not None:
        out[0] = prop_first_frame_latent
    return out


def resize_video(video, visual_size):
    height, width = video.shape[-2:]
    nearest_height, nearest_width = visual_size

    scale_factor = min(height / nearest_height, width / nearest_width)
    video = F.resize(video, (int(height / scale_factor), int(width / scale_factor)))

    height, width = video.shape[-2:]
    video = F.crop(
        video,
        (height - nearest_height) // 2,
        (width - nearest_width) // 2,
        nearest_height,
        nearest_width,
    )
    return video


def encode_video(data, vae, image_vae=False): # batch, channels, time, h, w
    """Encode video (or single-frame image) using HunyuanVideo 3D VAE.
    For images, pass data with T=1: (B, C, 1, H, W).
    image_vae param is kept for backward compat but ignored -- always uses 3D VAE."""
    encoded = vae.encode(data)
    # Handle multiple return styles across Diffusers/custom VAE wrappers.
    if hasattr(encoded, "latent_dist"):
        data = encoded.latent_dist.sample()
    elif isinstance(encoded, tuple):
        first = encoded[0]
        data = first.sample() if hasattr(first, "sample") else first
    else:
        data = encoded.sample() if hasattr(encoded, "sample") else encoded
    data *= vae.config.scaling_factor
    return data.permute(0, 2, 3, 4, 1) # batch, time, h, w, channels


def generate_sample(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    tp_mesh=None,
):
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)

    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    if offload:
        text_embedder = text_embedder.to('cpu')

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)
        
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                tp_mesh=tp_mesh,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
            )
            
    if tp_mesh:
        tensor_list = [
        torch.zeros_like(latent_visual, device=latent_visual.device) for _ in range(tp_mesh["tensor_parallel"].size())
        ]
        all_gather(
            tensor_list,
            latent_visual.contiguous(),
            group=tp_mesh.get_group(mesh_dim="tensor_parallel")
        )
        latent_visual = torch.cat(tensor_list, dim=1)

    if offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images

def generate_sample_ti2i(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    image_vae=False,
    image=None
):
    bs, duration, height, width, dim = shape
    
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)
    
    if duration == 1:
        if image is None:
            type_of_content = "image"
        else:
            type_of_content = 'image_edit'
    else:
        type_of_content = "video"

    
    if image is not None:
        image = [resize_video(image, (height * 8, width * 8))]

    if dit.instruct_type == 'channel':
        if image is not None:
            if offload:
                vae.to(vae_device)
            edit_latent = [(i.to(device=vae_device, dtype=torch.bfloat16) / 127.5 - 1.0) for i in image]
            edit_latent = torch.cat([encode_video(i[:,:,None], vae, image_vae).squeeze(0) for i in edit_latent], 0)
            edit_latent = torch.cat([edit_latent, torch.ones_like(img[...,:1])],-1)
            if offload:
                vae.to('cpu')
        else:
            edit_latent = torch.cat([torch.zeros_like(img), torch.zeros_like(img[...,:1])],-1)
        img = torch.cat([img, edit_latent],dim=-1)
    
    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, images=image
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content, images=image
        )

    if offload:
        text_embedder = text_embedder.to('cpu')

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device,dtype=torch.bfloat16)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device,dtype=torch.bfloat16)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)
        
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
            )
            
    if offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            if image_vae:
                images = images[:,:,0]
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images

def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    """Merge two tensors using a binary mask. Selects tensor_2 where mask=1, tensor_1 otherwise."""
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp


def get_task_mask(task_type, latent_T, guide_latents_num: int = 0,keyframe_idx: int = None):
    """Returns a temporal binary mask based on task type.

    For prop with guide_latents_num > 0 (Hunyuan-style): first guide_latents_num frames = 1, rest = 0.
    Otherwise prop/tv2v/ti2i-style tasks: all ones.
    """
    if task_type in ("t2v", "t2i"):
        return torch.zeros(latent_T)
    elif task_type in ("ti2i", "tv2v", "recon_v", "recon_i", "i2v"):
        # keyframe defaults to all-ones mask when no explicit keyframe_idx is provided
        return torch.ones(latent_T)
    elif task_type in ("prop"):
        if guide_latents_num > 0:
            mask = torch.zeros(latent_T)
            mask[: min(guide_latents_num, latent_T)] = 1.0
            return mask
        return torch.ones(latent_T)
    elif task_type in ("keyframe"):
        if keyframe_idx is not None:
            mask = torch.zeros(latent_T)
            mask[keyframe_idx] = 1.0
            return mask
        return torch.ones(latent_T)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def prepare_cond_input(task_type, cond_latent, noise_img, task_mask):
    """
    Build the 33-channel DiT input: [noise(16) | cond(16) | mask(1)].
    cond_latent: (T, H, W, 16) or None
    noise_img: (T, H, W, 16)
    task_mask: (T,) binary mask
    """
    T, H, W, C = noise_img.shape

    if cond_latent is not None:
        latent_concat = cond_latent.to(device=noise_img.device, dtype=noise_img.dtype)
    else:
        latent_concat = torch.zeros_like(noise_img)

    mask_ones = torch.ones(T, H, W, 1, device=noise_img.device, dtype=noise_img.dtype)
    mask_zeros = torch.zeros(T, H, W, 1, device=noise_img.device, dtype=noise_img.dtype)
    mask_channel = merge_tensor_by_mask(mask_zeros, mask_ones, mask=task_mask, dim=0)

    return torch.cat([noise_img, latent_concat, mask_channel], dim=-1)


def generate_unified_sample(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    task_type="t2v",
    cond_latent=None,
    guide_latent=None,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=10.0,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    images_for_text_encoder=None,
    tp_mesh=None,
    keyframe_idx: int = None,
):
    """
    Unified sampling function for all task types (t2v, t2i, ti2i, tv2v).
    Always produces 33-channel DiT input: [noise(16) + cond(16) + mask(1)].
    Uses a single HunyuanVideo 3D VAE for all tasks (images are T=1 videos).

    Args:
        shape: (bs, duration, height, width, dim) latent shape
        caption: text prompt
        dit: DiffusionTransformer3D with instruct_type='channel'
        vae: HunyuanVideo 3D VAE for encoding/decoding all tasks
        conf: OmegaConf config
        text_embedder: Kandinsky5TextEmbedder (Qwen + CLIP)
        task_type: one of "t2v", "t2i", "ti2i", "tv2v"
        cond_latent: (T, H, W, 16) VAE-encoded conditioning latent, or None for generation
        num_steps: number of denoising steps
        guidance_weight: classifier-free guidance weight
        scheduler_scale: timestep warping factor
        negative_caption: negative prompt for CFG
        seed: random seed
        device: DiT device
        vae_device: VAE device
        text_embedder_device: text embedder device
        progress: show progress bar
        offload: enable model offloading
        images_for_text_encoder: raw images/tensors for Qwen image_edit mode
        tp_mesh: tensor parallel mesh
    """
    bs, duration, height, width, dim = shape

    # Align cond_latent temporal length with diffusion duration (Hunyuan/Lucy-style).
    # For example, source video frames or VAE temporal downsampling may lead to
    # cond_latent.shape[0] != duration. We keep diffusion duration fixed and
    # crop/upsample cond_latent to match, to avoid later cat mismatches.
    if cond_latent is not None:
        cond_T = cond_latent.shape[0]
        if cond_T != duration:
            if cond_T > duration:
                # More conditioning frames than needed: crop.
                cond_latent = cond_latent[:duration]
            else:
                # Fewer conditioning frames: upsample along time to duration.
                # (T, H, W, C) -> (1, C, T, H, W)
                cond = cond_latent.unsqueeze(0).permute(0, 4, 1, 2, 3)
                cond = torch.nn.functional.interpolate(
                    cond, size=(duration, height, width), mode="nearest"
                )
                # (1, C, T, H, W) -> (T, H, W, C)
                cond_latent = cond.permute(0, 2, 3, 4, 1).squeeze(0)

    # Use same dtype as DiT weights to avoid dtype mismatches.
    weight_dtype = next(dit.parameters()).dtype

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    img = torch.randn(
        bs * duration, height, width, dim,
        device=device, generator=g, dtype=weight_dtype,
    )

    # modified 0311-1457, support recon_i, recon_v, i2v, keyframe, prop
    if task_type in ("ti2i", "recon_i"):
        type_of_content = "image_edit"
    elif task_type in ("tv2v", "recon_v", "i2v"):
        type_of_content = "video"
    elif task_type in ["keyframe","prop"]:
        type_of_content = "keyframe"
    elif duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video" # default
 
    # Task mask:
    # - prop: first guide_latents_num frames are treated as "given" (Hunyuan-style)
    # - keyframe: delegate to get_task_mask's keyframe branch (one-hot at keyframe_idx)
    if task_type == "prop" and guide_latent is not None:
        task_mask = get_task_mask(task_type, duration, guide_latents_num=1)
    elif task_type == "keyframe" and keyframe_idx is not None and keyframe_idx is not None:
        task_mask = get_task_mask(task_type, duration, keyframe_idx=int(keyframe_idx))
    else:
        task_mask = get_task_mask(task_type, duration) # default

    cond_for_input = cond_latent
    if task_type == "prop" and cond_latent is not None and guide_latent is not None:
        # For prop, blend guide into cond_latent on early frames:
        # cond_t = alpha * cond_t + (1 - alpha) * guide, 1 <= t < 1 + n_guide_frames.
        # 这样在前几帧轻量注入 guide 外观，同时保留 video1 的运动结构。
        cond = cond_latent.to(device=img.device)
        T, H, W, C = cond.shape
        g = guide_latent.to(device=img.device, dtype=cond.dtype)
        if g.dim() == 5:
            g = g.reshape(-1, g.shape[-3], g.shape[-2], g.shape[-1])[0:1]
        else:
            g = g[0:1]
        g_hw = g.squeeze(0)  # (H, W, C)
        if T > 1:
            alpha = 0.8
            n_guide_frames = min(8, T - 1)
            cond[1 : 1 + n_guide_frames] = (
                alpha * cond[1 : 1 + n_guide_frames]
                + (1.0 - alpha) * g_hw.unsqueeze(0)
            )
        cond_for_input = cond
    # Keyframe: guide_latent is baked into cond_latent by the pipeline (方案 A),
    # so no separate injection needed here. cond_for_input = cond_latent is correct.

    img = prepare_cond_input(task_type, cond_for_input, img, task_mask)

    # Optional debug trace for prop/keyframe inference (first call only)
    global _DEBUG_PROP_KEYFRAME_DONE
    if _DEBUG_PROP_KEYFRAME and not _DEBUG_PROP_KEYFRAME_DONE and task_type in ("prop", "keyframe"):
        debug_lines = []

        def _add(name, t):
            if isinstance(t, torch.Tensor):
                debug_lines.append(f"{name}: shape={tuple(t.shape)} dtype={t.dtype}")
            else:
                debug_lines.append(f"{name}: {t}")

        debug_lines.append(f"[DEBUG-PROP-KEYFRAME] task_type={task_type} keyframe_idx={keyframe_idx}")
        debug_lines.append(f"[DEBUG-PROP-KEYFRAME] shape(arg)={shape}")
        _add("cond_latent", cond_latent)
        _add("guide_latent", guide_latent)
        _add("cond_for_input", cond_for_input)
        _add("task_mask", task_mask)
        _add("img (DiT 33ch input)", img)

        debug_path = os.path.join(os.getcwd(), "debug_prop_keyframe_infer.txt")
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write("\n".join(debug_lines))
        except Exception:
            pass
        _DEBUG_PROP_KEYFRAME_DONE = True

    # Prop: fix first frame to guide image; keep condition (not zeroed) for attention coherence
    prop_first_frame_latent = None
    if task_type == "prop" and guide_latent is not None:
        # img: (bs * duration, H, W, 33) = [noise(16) | cond(16) | mask(1)]
        # guide_latent: (1, H, W, 16) or (1, 1, H, W, 16)
        g = guide_latent.to(device=img.device, dtype=img.dtype)
        if g.dim() == 5:
            g = g.reshape(-1, g.shape[-3], g.shape[-2], g.shape[-1])[0:1]
        else:
            g = g[0:1]
        img[0, :, :, 0:16] = g.squeeze(0)
        # Keep condition channel as guide_latent (set by prepare_cond_input) — training
        # uses cond[0] = target_latent (not zeros), so inference should match.
        prop_first_frame_latent = img[0, :, :, 0:16].clone()

    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, images=images_for_text_encoder,
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content, images=images_for_text_encoder,
        )

    if offload:
        text_embedder = text_embedder.to("cpu")

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device, dtype=torch.bfloat16)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device, dtype=torch.bfloat16)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                tp_mesh=tp_mesh,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                prop_first_frame_latent=prop_first_frame_latent,
            )

    if tp_mesh:
        tensor_list = [
            torch.zeros_like(latent_visual, device=latent_visual.device)
            for _ in range(tp_mesh["tensor_parallel"].size())
        ]
        all_gather(
            tensor_list,
            latent_visual.contiguous(),
            group=tp_mesh.get_group(mesh_dim="tensor_parallel"),
        )
        latent_visual = torch.cat(tensor_list, dim=1)

    if offload:
        dit = dit.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

    vae_dtype = vae.dtype if hasattr(vae, "dtype") else torch.float16
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=vae_dtype):
            images = latent_visual.reshape(
                bs, -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device, dtype=vae_dtype)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        vae = vae.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    return images


def generate_sample_i2v(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    images,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    tp_mesh=None
):
    text_embedder.embedder.mode = "i2v"
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)
    
    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"
        
    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    if offload:
        text_embedder = text_embedder.to('cpu')
        
    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)

    first_frames = images

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                first_frames,
                conf,
                seed=seed,
                progress=progress,
                tp_mesh=tp_mesh,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
            )

    if images is not None:
        images = images.to(device=latent_visual.device, dtype=latent_visual.dtype)
        latent_visual[:1] = images
    latent_visual = normalize_first_frame(latent_visual)

    if offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images

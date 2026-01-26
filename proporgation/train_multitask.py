"""
Training script for WanVideoPipeline TV2V + Propagation (prop) finetuning.

This is a refactor of the original Qwen multi-task script:
- Removes Qwen3VL and VLM projector (connector)
- Keeps a pure TV2V/prop training path
- Uses umT5 text encoder only

Note: TV2V conditioning is implemented by encoding the source video through VAE and
concatenating reference latents with the target latents (channel-wise) before DiT.
"""
import torch, os, argparse, accelerate, warnings, sys, glob, random
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from diffsynth.pipelines.wan_video_tv2v import WanVideoPipeline, ModelConfig  # type: ignore
from diffsynth.diffusion import *
from diffsynth.core.data import MultiResVideoEditDatasetOnline
from safetensors.torch import load_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_gpu_memory_info(device=None):
    """
    Mimic `hunyuan_edit/train_edit_v4_online.py` GPU memory stats.
    Returns values in GB.
    """
    if not torch.cuda.is_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "max_allocated": 0.0,
            "max_reserved": 0.0,
        }
    if device is None:
        device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return {
        "allocated": allocated,
        "reserved": reserved,
        "max_allocated": max_allocated,
        "max_reserved": max_reserved,
    }


def log_memory_info(logger, prefix="", device=None, accelerator=None):
    """
    Mimic `hunyuan_edit/train_edit_v4_online.py` log_memory_info().
    """
    mem_info = get_gpu_memory_info(device)
    msg = (
        f"{prefix}GPU Memory - "
        f"Allocated: {mem_info['allocated']:.2f} GB, "
        f"Reserved: {mem_info['reserved']:.2f} GB, "
        f"Max Allocated: {mem_info['max_allocated']:.2f} GB, "
        f"Max Reserved: {mem_info['max_reserved']:.2f} GB"
    )
    # If a logger-like object is provided, use it; otherwise print (main process only if accelerator provided).
    if logger is not None:
        logger.info(msg)
    elif accelerator is not None:
        if accelerator.is_main_process:
            accelerator.print(msg)
    else:
        print(msg)


def merge_tensor_by_mask(tensor_1: torch.Tensor, tensor_2: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Utility copied from hunyuan_edit/train_edit_v4_online.py.
    Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1.
    """
    assert tensor_1.shape == tensor_2.shape
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    else:
        raise ValueError(f"Unsupported dim={dim} for merge_tensor_by_mask")
    return tmp


def prepare_cond_latents(
    task_type: str,
    cond_latents: torch.Tensor | None,
    latents: torch.Tensor,
    multitask_mask: torch.Tensor,
    guide_latents_num: int = 0,
    empty_v_prob: float = 0.4,
) -> torch.Tensor:
    """
    Copied from hunyuan_edit/train_edit_v4_online.py.

    NOTE: This returns a tensor shaped like (B, C+1, T, H, W) = [cond_latents, mask].
    Our Wan FlowMatch training path does NOT currently feed this tensor into WanVideoPipeline
    (Wan DiT expects a fixed latent channel count), but we keep this utility here because
    propagation ("prop") data provides extra conditional frames (e.g. video2_key_frames_tensor)
    and future training variants may want this exact (cond + mask) formulation.
    """
    latents_concat: torch.Tensor | None = None

    if cond_latents is not None and task_type in ["i2v", "is2v"]:
        latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)  # B C T H W
        latents_concat[:, :, 1:, :, :] = 0.0
    elif cond_latents is not None and task_type in [
        "tv2v",
        "ti2i",
        "iv2v",
        "ii2i",
        "vv2v",
        "prop",
        "dense_prediction",
        "style_transfer",
        "conditional_gen",
    ]:
        latents_concat = cond_latents
        if guide_latents_num > 0:
            latents_concat[:, :, :guide_latents_num] = latents[:, :, :guide_latents_num]
            multitask_mask[:guide_latents_num] = 1
            multitask_mask[guide_latents_num:] = 0
            if random.random() < empty_v_prob:
                latents_concat[:, :, guide_latents_num:] = 0
    else:
        latents_concat = torch.zeros_like(latents)

    mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
    mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
    mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=latents.device)

    return torch.concat([latents_concat, mask_concat], dim=1)


def load_checkpoint_for_resume(accelerator, model, checkpoint_path, resume_models=None):
    """
    Load checkpoint with selective component loading.
    
    Args:
        accelerator: Accelerator instance
        model: Training model
        checkpoint_path: Path to checkpoint .safetensors file
        resume_models: List of model names to load (e.g., ['vlm_projector', 'denoising_model'])
                      If None, loads all available components
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"Loading checkpoint: {checkpoint_path}")
        if resume_models:
            print(f"Resume models: {', '.join(resume_models)}")
        else:
            print("Resume models: all available")
        print(f"{'='*80}\n")
    
    # Load checkpoint
    state_dict = load_file(checkpoint_path)
    
    # Detect available components
    has_denoising_model = any('denoising_model' in k or 'dit' in k or k.startswith('pipe.dit.') for k in state_dict.keys())
    
    if accelerator.is_main_process:
        print("📦 Available in checkpoint:")
        print(f"  {'✓' if has_denoising_model else '✗'} denoising_model")
        print()
    
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Load Denoising Model (DiT)
    if has_denoising_model and (resume_models is None or 'denoising_model' in resume_models):
        # Support both old names (denoising_model) and current WanVideoPipeline name (dit)
        target = None
        if hasattr(unwrapped_model.pipe, 'denoising_model') and unwrapped_model.pipe.denoising_model is not None:
            target = unwrapped_model.pipe.denoising_model
        elif hasattr(unwrapped_model.pipe, 'dit') and unwrapped_model.pipe.dit is not None:
            target = unwrapped_model.pipe.dit
        
        if target is not None:
            dit_keys = {k: v for k, v in state_dict.items() 
                       if 'denoising_model' in k or 'dit' in k or k.startswith('pipe.dit.')}
            
            # Clean keys
            clean_dict = {}
            for k, v in dit_keys.items():
                clean_key = k.replace('pipe.denoising_model.', '').replace('pipe.dit.', '').replace('denoising_model.', '').replace('dit.', '')
                clean_dict[clean_key] = v
            
            if clean_dict:
                load_result = target.load_state_dict(clean_dict, strict=False)
                if accelerator.is_main_process:
                    print(f"✓ denoising_model loaded: {len(clean_dict)} keys")
                    if len(load_result[0]) > 0:
                        print(f"  ⚠ Missing: {load_result[0][:5]}..." if len(load_result[0]) > 5 else f"  ⚠ Missing: {load_result[0]}")
        else:
            if accelerator.is_main_process:
                print("⚠ denoising_model/dit not initialized in model")
    elif resume_models and 'denoising_model' in resume_models:
        if accelerator.is_main_process:
            print("⚠ denoising_model requested but not found in checkpoint")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print("Checkpoint loading complete")
        print(f"{'='*80}\n")


class WanTV2VTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        t5_path: str | None = None,
        vae_path: str | None = None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        reference_concat_method="channel",  # "channel"
        use_prepare_cond_latents: bool = True,
        guide_latents_num: int = 1,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        def _resolve_from_arg(arg_path: str, candidates: list[str], label: str) -> str:
            """
            Resolve an explicit component path (file or directory).
            IMPORTANT: we only ever load the resolved *single file*; we do NOT glob for other
            diffusion_pytorch_model*.safetensors under the given directory.
            """
            if not arg_path:
                raise ValueError(f"{label} path is empty.")
            if os.path.isfile(arg_path):
                return arg_path
            if os.path.isdir(arg_path):
                for name in candidates:
                    p = os.path.join(arg_path, name)
                    if os.path.exists(p):
                        return p
                raise FileNotFoundError(
                    f"Could not find {label} under directory: {arg_path}. "
                    f"Tried: {candidates}"
                )
            raise FileNotFoundError(f"{label} path does not exist: {arg_path}")
        
        # Custom model loading logic for WanVideo folder
        if model_paths and not model_paths.strip().startswith("[") and os.path.isdir(model_paths):
            print(f"Loading WanVideo models from directory: {model_paths}")
            # Find sharded DiT
            dit_files = sorted(glob.glob(os.path.join(model_paths, "diffusion_pytorch_model-*.safetensors")))
            if not dit_files:
                # Fallback to single file
                if os.path.exists(os.path.join(model_paths, "diffusion_pytorch_model.safetensors")):
                    dit_files = [os.path.join(model_paths, "diffusion_pytorch_model.safetensors")]
            
            # Find T5 and VAE
            t5_file = os.path.join(model_paths, "models_t5_umt5-xxl-enc-bf16.pth")
            if not os.path.exists(t5_file):
                 # Try finding safetensors version if pth not found
                 t5_file = os.path.join(model_paths, "models_t5_umt5-xxl-enc-bf16.safetensors")
            if not os.path.exists(t5_file):
                if t5_path is None:
                    raise FileNotFoundError(
                        "T5 file not found under the training checkpoint folder. "
                        "Your checkpoint folder looks like a DiT-only export. "
                        "Please pass `--t5_path` pointing to the T5 file (or a directory containing it). "
                        f"Missing: {t5_file}"
                    )
                t5_file = _resolve_from_arg(
                    t5_path,
                    candidates=[
                        "models_t5_umt5-xxl-enc-bf16.safetensors",
                        "models_t5_umt5-xxl-enc-bf16.pth",
                    ],
                    label="T5",
                )

            vae_file = os.path.join(model_paths, "Wan2.2_VAE.pth")
            if not os.path.exists(vae_file):
                vae_file = os.path.join(model_paths, "Wan2.1_VAE.pth")
            if not os.path.exists(vae_file):
                if vae_path is None:
                    raise FileNotFoundError(
                        "VAE file not found under the training checkpoint folder. "
                        "Your checkpoint folder looks like a DiT-only export. "
                        "Please pass `--vae_path` pointing to the VAE file (or a directory containing it). "
                        f"Missing: {vae_file}"
                    )
                vae_file = _resolve_from_arg(
                    vae_path,
                    candidates=[
                        "Wan2.2_VAE.safetensors",
                        "Wan2.2_VAE.pth",
                        "Wan2.1_VAE.safetensors",
                        "Wan2.1_VAE.pth",
                    ],
                    label="VAE",
                )
            
            # Construct model configs manually
            model_configs = [
                ModelConfig(path=dit_files),
                ModelConfig(path=t5_file),
                ModelConfig(path=vae_file),
            ]
            
            # Helper for vram config
            vram_config = self.parse_vram_config(device=device)
            # Apply vram config to all
            for config in model_configs:
                config.offload_device = vram_config.get("offload_device")
                config.onload_device = vram_config.get("onload_device")
                # ... or easier: use parse_vram_config on each if needed, but here simple default is fine?
                # Actually parse_vram_config uses args.fp8_models etc. 
                # Ideally we should respect fp8/offload settings.
                
                # Check if this model is in offload list (by name? impossible now as we have paths)
                # But we can assume default behavior or apply global offload if user requested "all"?
                # Current parse_model_configs checks 'path in offload_models'. 
                # Here path is a list for DiT. 
                pass 

        else:
            model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        # Tokenizer config (use local folder if provided)
        tokenizer_config = None if tokenizer_path is None else ModelConfig(path=tokenizer_path)

        # Load WanVideoPipeline (no Qwen / no VLM projector)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device=device, 
            model_configs=model_configs, 
            tokenizer_config=tokenizer_config,
            redirect_common_files=False,
        )
        
        # If using token-concat reference conditioning (or hybrid), WanModel must have `ref_conv`.
        # Some checkpoints/configs instantiate WanModel with has_ref_conv=False, in which case
        # `model_fn_wan_video` will crash with AttributeError. We add it here (before optimizer
        # / deepspeed wrapping) so it becomes a proper trainable parameter.
        if reference_concat_method in ["token", "hybrid", "channel_real"]:
            import torch.nn as nn

            def _ensure_ref_conv(dit, vae):
                if dit is None:
                    return
                if hasattr(dit, "ref_conv") and getattr(dit, "ref_conv") is not None:
                    return
                # Get VAE z_dim dynamically (WanVideoVAE: z_dim=16, WanVideoVAE38: z_dim=48)
                in_ch = getattr(vae, 'z_dim', 16) if vae is not None else 16
                k = 2
                s = 2
                # Keep consistent with WanModel(has_ref_conv=True)
                dit.ref_conv = nn.Conv2d(in_ch, dit.dim, kernel_size=(k, k), stride=(s, s))
                dit.has_ref_conv = True
                print(f"[Token-Concat] Created ref_conv for DiT: in_ch={in_ch}, out_ch={dit.dim}, kernel={k}, stride={s}")

            vae = getattr(self.pipe, "vae", None)
            _ensure_ref_conv(getattr(self.pipe, "dit", None), vae)
            _ensure_ref_conv(getattr(self.pipe, "dit2", None), vae)
        
        # If using channel_real mode, modify DiT's patch_embedding to accept doubled channels (48 -> 96)
        if reference_concat_method == "channel_real":
            import torch.nn as nn
            
            def _modify_patch_embedding_for_channel_concat(dit, vae):
                if dit is None:
                    return
                # Get VAE z_dim (48 for Wan2.2)
                z_dim = getattr(vae, 'z_dim', 48) if vae is not None else 48
                new_in_dim = z_dim * 2  # 48 * 2 = 96
                
                old_patch_emb = dit.patch_embedding
                old_in_dim = old_patch_emb.in_channels
                out_dim = old_patch_emb.out_channels
                kernel_size = old_patch_emb.kernel_size
                stride = old_patch_emb.stride
                
                if old_in_dim == new_in_dim:
                    print(f"[Channel-Real] patch_embedding already has {new_in_dim} input channels")
                    return
                
                # Create new patch_embedding with doubled input channels
                new_patch_emb = nn.Conv3d(
                    new_in_dim, out_dim,
                    kernel_size=kernel_size,
                    stride=stride
                )
                
                # Initialize: copy weights for original channels, zero-init for new channels
                with torch.no_grad():
                    # Initialize new weights to zero
                    new_patch_emb.weight.zero_()
                    if new_patch_emb.bias is not None:
                        new_patch_emb.bias.zero_()
                    
                    # Copy original weights to the second half (for input_latents)
                    # new_patch_emb.weight shape: (out_dim, new_in_dim, *kernel_size)
                    # We want: [0:z_dim] = reference_latents (zero-init), [z_dim:2*z_dim] = input_latents (original weights)
                    new_patch_emb.weight[:, z_dim:, :, :, :] = old_patch_emb.weight
                    if old_patch_emb.bias is not None and new_patch_emb.bias is not None:
                        new_patch_emb.bias.copy_(old_patch_emb.bias)
                
                # Move to same device and dtype
                device = old_patch_emb.weight.device
                dtype = old_patch_emb.weight.dtype
                new_patch_emb = new_patch_emb.to(device=device, dtype=dtype)
                
                # Replace
                dit.patch_embedding = new_patch_emb
                dit.in_dim = new_in_dim
                
                print(f"[Channel-Real] Modified patch_embedding: in_channels {old_in_dim} -> {new_in_dim}, "
                      f"out_channels={out_dim}, kernel={kernel_size}, stride={stride}")
            
            vae = getattr(self.pipe, "vae", None)
            _modify_patch_embedding_for_channel_concat(getattr(self.pipe, "dit", None), vae)
            _modify_patch_embedding_for_channel_concat(getattr(self.pipe, "dit2", None), vae)
        
        # Split pipeline units for training
        if task == "sft:data_process" or task == "direct_distill:data_process":
            self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode setup
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.reference_concat_method = reference_concat_method  # "channel" or "token"
        # Use hunyuan-style prepare_cond_latents to build a latent-space timestep mask (first_frame_mask).
        # This does NOT change DiT input channels; it only controls per-token timesteps when
        # `dit.seperated_timestep` and `dit.fuse_vae_embedding_in_latents` are enabled.
        self.use_prepare_cond_latents = bool(use_prepare_cond_latents)
        self.guide_latents_num = int(guide_latents_num) if guide_latents_num is not None else 1
        
        def debug_loss_wrapper(loss_fn, name, pipe, inputs_shared, inputs_posi, inputs_nega):
            # print(f"\n[DEBUG] === Training Step ({name}) ===")
            # print(f"[DEBUG] Task: {getattr(pipe, '_current_task', 'unknown')}")
            
            if "input_video" in inputs_shared:
                v = inputs_shared["input_video"]
                # if isinstance(v, torch.Tensor):
                #     print(f"[DEBUG] Input Video (Raw): {v.shape} | Dev: {v.device} | Type: {v.dtype}")
            
            if "input_latents" in inputs_shared: # VAE output
                l = inputs_shared["input_latents"]
                # print(f"[DEBUG] Input Latents (VAE): {l.shape} | Dev: {l.device} | Type: {l.dtype}")
            
            if "context" in inputs_posi:
                c = inputs_posi["context"]
                # print(f"[DEBUG] Prompt Context (Merged): {c.shape} | Dev: {c.device} | Type: {c.dtype}")
            
            # Execute actual loss
            loss = loss_fn(pipe, **inputs_shared, **inputs_posi)
            # print(f"[DEBUG] Loss: {loss.item()}")
            return loss

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: debug_loss_wrapper(FlowMatchSFTLoss, "sft", pipe, inputs_shared, inputs_posi, inputs_nega),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: debug_loss_wrapper(FlowMatchSFTLoss, "sft:train", pipe, inputs_shared, inputs_posi, inputs_nega),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def extract_reference_for_vae(self, data, task, data_type):
        """
        Extract reference video/image for VAE encoding in TI2I/TV2V/prop tasks.
        
        Args:
            data: batch data
            task: task type ('ti2i', 'tv2v', or 'prop')
            data_type: data type ('image', 'video', or 'cond_video')
            
        Returns:
            reference tensor or None
        """
        if task not in ['ti2i', 'tv2v', 'prop']:
            return None
        
        if data_type == "image" or data_type == "gen_image":
            # TI2I: use img1 (source image)
            if "img1" in data:
                # img1: [B, C, H, W]
                # Add time dimension: [B, C, 1, H, W]
                reference = [data["img1"][i:i+1].unsqueeze(2) for i in range(len(data["img1"]))]
                return reference
            return None
        
        elif data_type in ("video", "gen_video", "cond_video"):
            # TV2V/prop: use video1 (source video)
            if "video1" in data:
                # video1: list of [C, T, H, W] or [B, C, T, H, W]
                reference = [data["video1"][i:i+1] if len(data["video1"].shape) == 5 
                           else data["video1"][i].unsqueeze(0) 
                           for i in range(len(data["video1"]))]
                return reference
            return None
        
        return None
    
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        """Parse extra inputs from data batch."""
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        """
        Prepare pipeline inputs from batch data.
        NOTE (customized):
        We treat TV2V batches with the *propagation-style* conditioning logic (mimicking
        `hunyuan_edit/train_edit_v4_online.py` prop branch):
        - use a conditional key frame tensor (video2_key_frames_tensor) when available
        - otherwise derive it from the first frame of video2
        - pass it through Wan LongCatVideo path as `longcat_video`
        This disables the classic TV2V reference-latent concatenation path.
        
        Args:
            data: batch data from MultiResVideoEditDatasetOnline
                - type: "video"
                - task: "tv2v"
                - video1/video2: tensors (B, 3, T, H, W)
                - instruction: text instruction/caption
        
        Returns:
            tuple: (inputs_shared, inputs_posi, inputs_nega)
        """
        data_type = data.get("type", ["video"])[0] if isinstance(data.get("type", "video"), list) else data.get("type", "video")
        task = data.get("task", ["tv2v"])[0] if isinstance(data.get("task", "tv2v"), list) else data.get("task", "tv2v")
        # Allow TV2V and Propagation tasks.
        # dataset_multires_online.py returns:
        # - tv2v: type="video", task="tv2v"
        # - prop: type="cond_video", task="prop"
        if task not in ["tv2v", "prop"] or data_type not in ["video", "cond_video"]:
            raise ValueError(
                f"Expected TV2V/prop batch, but got type={data_type}, task={task}. "
                "Please provide a dataset containing only tv2v/prop samples."
            )
        
        # Get prompt/instruction
        if "instruction" in data:
            prompt = data["instruction"]
        elif "caption" in data:
            prompt = data["caption"]
        else:
            prompt = [""] * len(data.get("video1", data.get("tensor", torch.zeros(1))))
        
        # Build inputs
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        
        # Propagation task (CORRECTED):
        # INPUTS:
        #   - video1: SOURCE video (provides structure/motion)
        #   - video2's first frame: TARGET style (the edited appearance to propagate)
        #   - prompt: editing instruction
        # OUTPUT:
        #   - video2: TARGET video (the full edited video we train to generate)
        #
        # Training objective: Given video1 (structure) + video2[0] (style) + prompt → generate video2
        
        if "video1" not in data:
            raise ValueError("TV2V/prop batch must contain 'video1' (source video for structure).")
        if "video2" not in data:
            raise ValueError("TV2V/prop batch must contain 'video2' (target video to generate).")
        
        # DataLoader collates tensors -> (B, 3, T, H, W)
        # video1: SOURCE (provides structure/motion)
        video1 = data["video1"]
        if not isinstance(video1, torch.Tensor):
            raise ValueError("Expected video1 to be torch.Tensor with shape (B,3,T,H,W).")
        
        # video2: TARGET (what we train the model to generate)
        video2 = data["video2"]
        if not isinstance(video2, torch.Tensor):
            raise ValueError("Expected video2 to be torch.Tensor with shape (B,3,T,H,W).")
        
        # Handle shape: could be (B, 3, T, H, W) or (3, T, H, W)
        if video1.dim() == 4:
            video1 = video1.unsqueeze(0)
        if video2.dim() == 4:
            video2 = video2.unsqueeze(0)

        # input_video = video2 (TARGET) - this gets encoded, noised, and trained to denoise
        inputs_shared = {
            "input_video": video2,  # TARGET: video2 (what we generate)
            "height": int(video2.shape[-2]),
            "width": int(video2.shape[-1]),
            "num_frames": int(video2.shape[2]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Reference conditioning:
        # - video1: provides structure/motion (channel concat in hybrid mode)
        # - video2's first frame: provides target style to propagate (token/channel concat)
        #
        # Get video2's first frame as the style condition
        video2_first_frame = data.get("video2_key_frames_tensor", None)
        if isinstance(video2_first_frame, torch.Tensor):
            if video2_first_frame.dim() == 4:  # (3,1,H,W) -> (1,3,1,H,W)
                video2_first_frame = video2_first_frame.unsqueeze(0)
            style_condition = video2_first_frame
        else:
            # Extract first frame from video2
            style_condition = video2[:, :, :1].contiguous()  # (B, 3, 1, H, W)
        
        # video1 provides temporal structure/motion; keep the full video for conditioning
        video1_full = video1
        
        # Set reference_concat_method and corresponding inputs
        inputs_shared["reference_concat_method"] = self.reference_concat_method
        
        if self.reference_concat_method == "channel":
            # Channel concat: use video2's first frame as style condition
            inputs_shared["reference_video"] = style_condition
            inputs_shared["conditional_image"] = None
        elif self.reference_concat_method == "token":
            # Token concat: use video2's first frame as style condition
            inputs_shared["reference_video"] = style_condition
            inputs_shared["conditional_image"] = style_condition
        elif self.reference_concat_method == "hybrid":
            # Hybrid mode: BOTH token and channel concat
            # - conditional_image (video2's first frame) → token concat (target style via ref_conv)
            # - reference_video (video1) → channel concat (source structure/motion)
            inputs_shared["conditional_image"] = style_condition  # video2's first frame for style
            inputs_shared["reference_video"] = video1_full  # full video1 for temporal structure
        elif self.reference_concat_method == "channel_real":
            # TRUE channel concat for temporal structure + token concat for global style:
            # - reference_video = video1 (FULL)  -> channel_real concat provides temporal structure
            # - conditional_image = video2's first frame -> token concat provides target style
            inputs_shared["reference_video"] = video1_full
            inputs_shared["conditional_image"] = style_condition
        else:
            raise ValueError(f"Unknown reference_concat_method: {self.reference_concat_method}")
        
        # Also pass longcat_video for compatibility
        inputs_shared["longcat_video"] = style_condition

        # ------------------------------------------------------------------
        # Optional: use prepare_cond_latents (hunyuan_edit style) to build a
        # latent-space per-frame mask for separated-timestep conditioning.
        #
        # In Wan's model_fn_wan_video, `first_frame_mask` (latent-space) controls
        # which tokens use timestep=0 (clean condition) vs timestep=t (noisy target).
        # We expose `--guide_latents_num` to condition the first N frames.
        # ------------------------------------------------------------------
        if self.use_prepare_cond_latents:
            try:
                pipe = self.pipe
                # Ensure VAE is available on device (vram management may offload it).
                pipe.load_models_to_device(["vae"])
                with torch.no_grad():
                    # Encode target video to latents to get (B, C, T, H, W) shape.
                    v2 = pipe.preprocess_video(inputs_shared["input_video"])
                    v2_lat = pipe.vae.encode(v2, device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype, device=pipe.device)
                    T_lat = int(v2_lat.shape[2])
                    n = max(0, min(self.guide_latents_num, T_lat))
                    # multitask_mask: 0 -> condition tokens (t=0), 1 -> target tokens (t=timestep)
                    multitask_mask = torch.ones((T_lat,), device=v2_lat.device, dtype=torch.long)
                    if n > 0:
                        multitask_mask[:n] = 0
                    cond_plus_mask = prepare_cond_latents(
                        task_type=task,
                        cond_latents=None,
                        latents=v2_lat,
                        multitask_mask=multitask_mask,
                        guide_latents_num=0,
                        empty_v_prob=0.0,
                    )
                    inputs_shared["first_frame_mask"] = cond_plus_mask[:, -1:, :, :, :].contiguous()
            except Exception as e:
                # Don't hard-fail training if mask preparation fails; fallback to pipeline default.
                if not hasattr(self, "_warned_prepare_cond_latents"):
                    self._warned_prepare_cond_latents = True
                    print(f"[warn] prepare_cond_latents disabled due to error: {e}")
        
        # Parse extra inputs if provided
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        """
        Forward pass with multi-task training.
        
        Args:
            data: batch data from dataloader
            inputs: optional pre-prepared inputs
        
        Returns:
            loss: training loss
        """
        if inputs is None: 
            inputs = self.get_pipeline_inputs(data)
        
        # Transfer to device
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        
        # Debug: print input shapes before pipeline units
        inputs_shared, inputs_posi, inputs_nega = inputs
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            print("\n" + "="*80)
            print("[DEBUG] Input shapes before pipeline units:")
            if "input_video" in inputs_shared and inputs_shared["input_video"] is not None:
                print(f"  input_video: {inputs_shared['input_video'].shape}")
            if "reference_video" in inputs_shared and inputs_shared["reference_video"] is not None:
                ref = inputs_shared["reference_video"]
                if isinstance(ref, torch.Tensor):
                    print(f"  reference_video: {ref.shape}")
                elif isinstance(ref, list) and len(ref) > 0:
                    print(f"  reference_video: list of {len(ref)} items, first: {type(ref[0])}")
            if "conditional_image" in inputs_shared and inputs_shared["conditional_image"] is not None:
                cond = inputs_shared["conditional_image"]
                if isinstance(cond, torch.Tensor):
                    print(f"  conditional_image: {cond.shape}")
            print(f"  reference_concat_method: {inputs_shared.get('reference_concat_method', 'N/A')}")
            print("="*80)
        
        # Run pipeline units
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            
            # Debug: print latent shapes after key units
            if not hasattr(self, '_debug_printed') or not self._debug_printed:
                inputs_shared, inputs_posi, inputs_nega = inputs
                unit_name = unit.__class__.__name__
                if "latents" in inputs_shared and inputs_shared["latents"] is not None:
                    latents = inputs_shared["latents"]
                    print(f"[DEBUG] After {unit_name}:")
                    print(f"  latents: {latents.shape} (B={latents.shape[0]}, C={latents.shape[1]}, T={latents.shape[2]}, H={latents.shape[3]}, W={latents.shape[4]})")
                    if "input_latents" in inputs_shared and inputs_shared["input_latents"] is not None:
                        il = inputs_shared["input_latents"]
                        print(f"  input_latents: {il.shape}")
                    if "reference_latents" in inputs_shared and inputs_shared["reference_latents"] is not None:
                        rl = inputs_shared["reference_latents"]
                        print(f"  reference_latents: {rl.shape}")
                    if "first_frame_latents" in inputs_shared and inputs_shared["first_frame_latents"] is not None:
                        fl = inputs_shared["first_frame_latents"]
                        print(f"  first_frame_latents: {fl.shape}")
        
        # Mark debug as printed (only print once)
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print("="*80 + "\n")
        
        # Compute loss
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        
        return loss


def wan_tv2v_parser():
    """Command line argument parser for pure WanVideo TV2V/prop training (no Qwen / no projector)."""
    parser = argparse.ArgumentParser(description="Training script for WanVideo TV2V/prop (no Qwen / no projector).")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    
    # Model paths
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument(
        "--t5_path",
        type=str,
        default=None,
        help=(
            "Explicit path to T5 weights (file or directory). "
            "Used when --model_paths points to a DiT-only checkpoint folder that does not contain "
            "models_t5_umt5-xxl-enc-bf16.(safetensors|pth). "
            "NOTE: we will ONLY load the resolved T5 file and will NOT glob any diffusion_pytorch_model*.safetensors under this path."
        ),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=(
            "Explicit path to VAE weights (file or directory). "
            "Used when --model_paths points to a DiT-only checkpoint folder that does not contain "
            "Wan2.(1|2)_VAE.(safetensors|pth). "
            "NOTE: we will ONLY load the resolved VAE file and will NOT glob any diffusion_pytorch_model*.safetensors under this path."
        ),
    )
    
    # Dataset paths
    parser.add_argument("--dataset_csv_paths", type=str, required=True, help="Comma-separated CSV paths for dataset.")
    parser.add_argument("--dataset_data_roots", type=str, required=True, help="Comma-separated data root paths.")
    parser.add_argument("--prop_train", type=str, default="false", choices=["true", "false", "all"], help="Propagation training mode.")
    
    # Batch sizes (this script expects tv2v/prop dataset; other batch sizes kept for sampler compatibility)
    parser.add_argument("--video_batch_size", type=int, default=1, help="Batch size for TV2V/prop.")
    parser.add_argument("--image_batch_size", type=int, default=8, help="Unused (keep for sampler).")
    parser.add_argument("--gen_video_batch_size", type=int, default=1, help="Unused (keep for sampler).")
    parser.add_argument("--gen_image_batch_size", type=int, default=8, help="Unused (keep for sampler).")
    
    # Training parameters
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Initialize models on CPU.")
    parser.add_argument("--max_frames", type=int, default=81, help="Maximum number of frames for video tasks.")
    parser.add_argument(
        "--use_prepare_cond_latents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use hunyuan-style prepare_cond_latents to build latent-space first_frame_mask "
             "(controls separated-timestep conditioning; does NOT change DiT input channels).",
    )
    parser.add_argument(
        "--guide_latents_num",
        type=int,
        default=1,
        help="Number of initial latent frames to treat as clean condition (mask=0) when use_prepare_cond_latents is enabled.",
    )
    parser.add_argument("--reference_concat_method", type=str, default="hybrid", choices=["channel", "token", "hybrid", "channel_real"],
                       help="Concat method for reference latents (TV2V): 'channel' (time concat), 'token', 'hybrid' (both), or 'channel_real' (true channel concat, 48->96).")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint file (.safetensors) to resume from.")
    parser.add_argument("--resume_models", type=str, default=None, help="Comma-separated model names to load from checkpoint (e.g., 'vlm_projector,denoising_model').")
    
    # Logging arguments
    parser.add_argument("--use_tensorboard", default=False, action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Enable WandB logging.")
    parser.add_argument("--wandb_project", type=str, default="wan-tv2v-training", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (auto-generated if None).")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/team name.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--log_grad_norm", default=False, action="store_true", help="Log gradient norms.")
    
    return parser


def launch_training_task(accelerator, dataloader, model, model_logger, args, tb_writer=None, use_wandb=False):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        log_interval = getattr(args, 'log_interval', 10)
        log_grad_norm = getattr(args, 'log_grad_norm', False)
        save_projector_every_n_steps = None
    else:
        learning_rate = 1e-4
        weight_decay = 1e-2
        save_steps = 1000
        num_epochs = 1
        log_interval = 10
        log_grad_norm = False
        save_projector_every_n_steps = None
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # Resume from checkpoint if specified
    if args is not None and args.resume_from_checkpoint is not None:
        resume_models = args.resume_models.split(",") if args.resume_models else None
        load_checkpoint_for_resume(accelerator, model, args.resume_from_checkpoint, resume_models)
    
    global_step = 0
    
    for epoch_id in range(num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process)
        progress_bar.set_description(f"Epoch {epoch_id}")
        log_memory_info(logger=None, prefix=f"[epoch {epoch_id}] ", device=getattr(accelerator, "device", None), accelerator=accelerator)
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        for data in progress_bar:
            
            # Extract task info for logging
            task = data.get("task", ["unknown"])[0] if isinstance(data.get("task", "unknown"), list) else data.get("task", "unknown")
            data_type = data.get("type", ["unknown"])[0] if isinstance(data.get("type", "unknown"), list) else data.get("type", "unknown")
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                
                # Compute gradient norm if requested
                grad_norm = None
                if log_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_modules(), max_norm=float('inf'))
                
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                
                # (Removed) VLM projector saving
                
                # Accumulate loss for epoch average
                epoch_loss += loss.item()
                epoch_steps += 1
                
                # Logging
                if accelerator.is_main_process and global_step % log_interval == 0:
                    log_memory_info(logger=None, prefix=f"[step {global_step}] ", device=getattr(accelerator, "device", None), accelerator=accelerator)
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch_id,
                        "train/global_step": global_step,
                        f"train/loss_{task}": loss.item(),
                        f"train/loss_{data_type}": loss.item(),
                    }
                    
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    
                    # TensorBoard logging
                    if tb_writer is not None:
                        for key, value in log_dict.items():
                            tb_writer.add_scalar(key, value, global_step)
                    
                    # WandB logging
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log(log_dict, step=global_step)
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "task": task,
                    "lr": scheduler.get_last_lr()[0],
                })
                
                global_step += 1
        
        # Log epoch statistics
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        if accelerator.is_main_process:
            log_dict = {
                "epoch/avg_loss": avg_epoch_loss,
                "epoch/epoch_id": epoch_id,
            }
            
            if tb_writer is not None:
                for key, value in log_dict.items():
                    tb_writer.add_scalar(key, value, epoch_id)
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log(log_dict, step=global_step)
        
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    
    # Log final summary
    if accelerator.is_main_process:
        print(f"\nTraining completed: {num_epochs} epochs, {global_step} steps")
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"train/completed": True}, step=global_step)
            
    model_logger.on_training_end(accelerator, model, save_steps)


if __name__ == "__main__":
    parser = wan_tv2v_parser()
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    accelerator.even_batches = False
    
    # Check if DeepSpeed is used and fix batch size
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.video_batch_size
    
    # Parse CSV paths and data roots
    csv_paths = args.dataset_csv_paths.split(",")
    data_roots = args.dataset_data_roots.split(",")
    
    # Initialize dataset with online feature extraction
    dataset = MultiResVideoEditDatasetOnline(
        csv_path=csv_paths,
        data_root=data_roots,
        prop_train=args.prop_train,
        max_frames=args.max_frames,
    )

    # Safety check: this script supports TV2V and Propagation only.
    dataset_tasks = sorted(list(set(d.get("task") for d in dataset.data_list if isinstance(d, dict) and "task" in d)))
    dataset_types = sorted(list(set(d.get("type") for d in dataset.data_list if isinstance(d, dict) and "type" in d)))
    allowed_tasks = {"tv2v", "prop"}
    allowed_types = {"video", "cond_video"}
    if any(t not in allowed_tasks for t in dataset_tasks) or any(tp not in allowed_types for tp in dataset_types):
        raise RuntimeError(
            "This script only supports tv2v/prop datasets.\n"
            f"Detected tasks={dataset_tasks}, types={dataset_types}.\n"
            "Please provide csv(s) that only contain tv2v and/or prop samples."
        )
    
    # Get multi-resolution sampler
    sampler = dataset.get_sampler(
        video_batch_size=args.video_batch_size,
        image_batch_size=args.image_batch_size,
        gen_video_batch_size=args.gen_video_batch_size,
        gen_image_batch_size=args.gen_image_batch_size,
        distributed=accelerator.num_processes > 1,
    )
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        pin_memory=True,
    )
    
    # Initialize training module
    model = WanTV2VTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        t5_path=getattr(args, "t5_path", None),
        vae_path=getattr(args, "vae_path", None),
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        reference_concat_method=args.reference_concat_method,
        use_prepare_cond_latents=args.use_prepare_cond_latents,
        guide_latents_num=args.guide_latents_num,
    )
    
    # Setup model logger
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    
    # Initialize logging
    tb_writer = None
    use_wandb = False
    
    if accelerator.is_main_process:
        # Setup TensorBoard
        if args.use_tensorboard:
            tensorboard_dir = os.path.join(args.output_path, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"TensorBoard logging enabled: {tensorboard_dir}")
        
        # Setup WandB
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else f"tv2v-{args.task}-{os.path.basename(args.output_path)}"
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                entity=args.wandb_entity,
                config=vars(args),
                dir=args.output_path,
            )
            use_wandb = True
            print(f"WandB logging enabled: {args.wandb_project}/{wandb_run_name}")
            
            # Log model architecture info
            wandb.config.update({
                "trainable_params": sum(p.numel() for p in model.trainable_modules() if hasattr(p, 'numel')),
                "num_workers": args.num_workers if hasattr(args, 'num_workers') else 4,
                "dataset_size": len(dataset),
            })
        elif args.use_wandb and not WANDB_AVAILABLE:
            print("Warning: WandB logging requested but wandb is not installed.")
        
        # Log training configuration
        if tb_writer is not None:
            config_text = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
            tb_writer.add_text("config", config_text, 0)
        
        print("\n=== Training Configuration ===")
        print(f"Task: {args.task}")
        print(f"Output path: {args.output_path}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Batch sizes - Video: {args.video_batch_size}, Image: {args.image_batch_size}")
        print(f"Gen batch sizes - Video: {args.gen_video_batch_size}, Image: {args.gen_image_batch_size}")
        print(f"Epochs: {args.num_epochs}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Dataset size: {len(dataset)}")
        print(f"Max frames: {args.max_frames}")
        print("==============================\n")
    
    # Launch training
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": lambda acc, dl, m, ml, args: launch_training_task(acc, dl, m, ml, args, tb_writer=tb_writer, use_wandb=use_wandb),
        "sft:train": lambda acc, dl, m, ml, args: launch_training_task(acc, dl, m, ml, args, tb_writer=tb_writer, use_wandb=use_wandb),
        "direct_distill": lambda acc, dl, m, ml, args: launch_training_task(acc, dl, m, ml, args, tb_writer=tb_writer, use_wandb=use_wandb),
        "direct_distill:train": lambda acc, dl, m, ml, args: launch_training_task(acc, dl, m, ml, args, tb_writer=tb_writer, use_wandb=use_wandb),
    }
    
    launcher_map[args.task](accelerator, train_dataloader, model, model_logger, args=args)
    
    # Cleanup logging
    if accelerator.is_main_process:
        if tb_writer is not None:
            tb_writer.close()
            print("TensorBoard writer closed.")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            print("WandB run finished.")

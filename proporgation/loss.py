from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # Check if using channel_real mode (doubled channels)
    reference_concat_method = inputs.get("reference_concat_method", "channel")

    if reference_concat_method == "channel_real":
        # channel_real mode: input_latents has shape (B, 96, T, H, W) = [ref_latents | target_latents]
        # We add noise and compute loss ONLY on the target portion (channels z_dim:2*z_dim).
        input_latents = inputs["input_latents"]
        z_dim = input_latents.shape[1] // 2  # 96 // 2 = 48

        # Split into reference and target portions
        ref_latents = input_latents[:, :z_dim]      # (B, 48, T, H, W) - structure reference (no noise)
        target_latents = input_latents[:, z_dim:]   # (B, 48, T, H, W) - target video2 (add noise)

        # Only add noise to target portion
        noise = torch.randn_like(target_latents)
        noisy_target = pipe.scheduler.add_noise(target_latents, noise, timestep)

        # ------------------------------------------------------------------
        # Align with the "expand_timesteps" reference logic (screenshot):
        #
        #   latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
        #   temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
        #   timestep = temp_ts.unsqueeze(0).expand(B, -1)
        #
        # In our training setup:
        # - `condition` comes from the style reference image latent (inputs["reference_latents"]) when present
        # - we use it to REPLACE the first frame of the target latents (mask=0 at frame 0)
        # - we pass `first_frame_mask` into model_fn so timestep expansion stays consistent
        # ------------------------------------------------------------------
        ref_style_latents = inputs.get("reference_latents", None)
        if ref_style_latents is not None:
            # Normalize style latents to shape (B, z_dim, 1, H, W)
            if ref_style_latents.dim() == 5 and ref_style_latents.shape[2] > 1:
                ref_style_latents = ref_style_latents[:, :, :1]
            ref_style_latents = ref_style_latents.to(dtype=target_latents.dtype, device=target_latents.device)

            B, _, T, H, W = target_latents.shape
            first_frame_mask = torch.ones((B, 1, T, H, W), device=target_latents.device, dtype=target_latents.dtype)
            first_frame_mask[:, :, 0:1] = 0.0

            # condition tensor: same shape as latents (B,z_dim,T,H,W), only first frame replaced
            condition = noisy_target.clone()
            condition[:, :, 0:1] = ref_style_latents

            # latent_model_input: replace first frame with condition, keep other frames as noisy_target
            noisy_target = (1.0 - first_frame_mask[:, 0:1]) * condition + first_frame_mask[:, 0:1] * noisy_target

            # pass mask down so model_fn can expand per-token timestep consistently
            inputs["first_frame_mask"] = first_frame_mask

        # Concatenate back: [clean structure ref | noisy target]
        inputs["latents"] = torch.cat([ref_latents, noisy_target], dim=1)

        # Training target is only for the target portion
        training_target = pipe.scheduler.training_target(target_latents, noise, timestep)
    else:
        # Original behavior for other modes
        noise = torch.randn_like(inputs["input_latents"])
        inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    # If we used a first-frame condition mask, ignore the first frame in loss (it wasn't noised).
    if reference_concat_method == "channel_real" and inputs.get("first_frame_mask", None) is not None:
        ffm = inputs["first_frame_mask"].to(dtype=training_target.dtype, device=training_target.device)  # (B,1,T,H,W)
        # broadcast to channels
        ffm = ffm.expand(-1, training_target.shape[1], -1, -1, -1)
        per_elem = (noise_pred.float() - training_target.float()) ** 2
        loss = (per_elem * ffm).sum() / (ffm.sum().clamp_min(1.0))
    else:
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss

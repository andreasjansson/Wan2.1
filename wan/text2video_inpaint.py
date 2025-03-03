import random
import sys
from contextlib import contextmanager
import gc
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .text2video import WanT2V


class WanT2VInpaint(WanT2V):
    def preprocess_video(self, video: list[Image.Image]) -> torch.Tensor:
        video_frames = []
        for frame in video:
            frame = frame.convert("RGB")
            frame = np.array(frame).astype(np.float32) / 255.0
            video_frames.append(frame)

        # TODO: so much permutation mess, could be much simplified
        video_np = np.stack(video_frames)
        video_np = video_np.transpose(0, 3, 1, 2)
        video_tensor = torch.from_numpy(video_np)
        video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)

        return 2.0 * video_tensor - 1.0

    def preprocess_mask(self, mask: list[Image.Image]) -> torch.Tensor:
        mask_frames = []
        for frame in mask:
            frame = frame.convert("L")
            frame = np.array(frame).astype(np.float32) / 255.0
            frame = frame[..., None]
            mask_frames.append(frame)

        mask_np = np.stack(mask_frames)
        mask_np = mask_np.transpose(0, 3, 1, 2)
        mask_tensor = torch.from_numpy(mask_np)

        # Invert mask: 1 for areas to inpaint, 0 for areas to keep
        mask_tensor = 1 - mask_tensor
        return mask_tensor.unsqueeze(0)

    def get_timesteps(self, scheduler, num_inference_steps, strength):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start * scheduler.order :].to(self.device)

        return timesteps, t_start

    def prepare_latents(
        self,
        scheduler,
        init_video,
        timestep,
        generator,
    ):
        init_video = init_video.to(device=self.device, dtype=self.param_dtype)

        video_frames = init_video[0].permute(1, 0, 2, 3)
        latent = self.vae.encode([video_frames])[0]
        latents = torch.stack([latent])

        init_latents_orig = latents.clone()
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=self.device,
            dtype=self.param_dtype,
        )
        latents = scheduler.add_noise(latents, noise, timestep)

        return latents, init_latents_orig, noise

    def prepare_mask_latents(self, mask):
        mask = mask.to(device=self.device, dtype=self.param_dtype)

        _, frames, _, height, width = mask.shape
        temporal_factor = self.vae_stride[0]

        frame_masks = []
        for f in range(0, frames, temporal_factor):
            # Take one frame per temporal_factor frames
            if f < frames:
                frame_mask = mask[0, f]

                # Resize to latent space dimensions for height and width
                resized_frame_mask = F.interpolate(
                    frame_mask.unsqueeze(0),
                    size=(
                        height // self.vae_stride[1],
                        width // self.vae_stride[2],
                    ),
                    mode="nearest",
                ).squeeze(0)

                frame_masks.append(resized_frame_mask)

        final_masks = torch.stack(frame_masks).unsqueeze(0)
        final_masks = final_masks.permute(0, 2, 1, 3, 4)
        final_masks = final_masks.repeat(1, self.vae.model.z_dim, 1, 1, 1)

        return final_masks

    def generate_inpaint(
        self,
        input_prompt,
        init_video: list[Image.Image],
        mask: list[Image.Image],
        size=(1280, 720),
        strength=0.8,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        inpaint_fixup_steps=1,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=False,
    ):
        frame_num = len(init_video)

        # arbitrary
        if frame_num > 200:
            raise ValueError(
                f"Maximum number of frames is 200, but your video has {frame_num} frames"
            )

        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        init_video_tensor = self.preprocess_video(init_video)
        init_video_tensor = init_video_tensor.permute(0, 2, 1, 3, 4)

        mask_tensor = self.preprocess_mask(mask)

        if init_video_tensor.shape[1] != mask_tensor.shape[1]:
            raise ValueError(
                f"Video and mask must have same frame count. Got {init_video_tensor.shape[1]} and {mask_tensor.shape[1]}"
            )

        target_shape = (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = scheduler.timesteps
        elif sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                scheduler, device=self.device, sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError("Unsupported solver.")

        timesteps, t_start = self.get_timesteps(scheduler, sampling_steps, strength)
        latent_timestep = timesteps[:1]
        scheduler.set_begin_index(t_start)

        latents, init_latents_orig, noise = self.prepare_latents(
            scheduler,
            init_video_tensor,
            latent_timestep,
            generator,
        )

        # Prepare mask latents
        mask_latents = self.prepare_mask_latents(mask_tensor)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        with (
            torch.cuda.amp.autocast(dtype=self.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            for i, t in enumerate(tqdm(timesteps)):
                timestep = torch.stack([t])

                noise_pred_cond = self.model(latents, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(latents, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                latents = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents,
                    return_dict=False,
                    generator=generator,
                )[0].squeeze(0)

                init_latents_proper = scheduler.add_noise(
                    init_latents_orig, noise, torch.tensor([t])
                )

                if i + 1 == len(timesteps) - inpaint_fixup_steps:
                    mask_latents *= 0

                latents = (
                    1 - mask_latents
                ) * latents + mask_latents * init_latents_proper

            if offload_model:
                self.model.cpu()

            decoded = self.vae.decode(latents)
            video_frames = decoded[0].permute(1, 0, 2, 3)
            video_frames = (video_frames / 2 + 0.5).clamp(0, 1)
            video_frames = video_frames.squeeze(0).permute(1, 0, 2, 3)

        # Clean up
        del latents, init_latents_orig, noise, mask_latents
        del scheduler

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        return video_frames

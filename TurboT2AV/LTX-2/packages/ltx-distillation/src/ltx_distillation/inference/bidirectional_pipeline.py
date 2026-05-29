"""Bidirectional audio-video inference pipeline."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class BidirectionalAVInferencePipeline:
    """
    Pipeline for few-step bidirectional inference.

    Generates audio-video samples using a distilled model.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        use_trigflow: bool = False,
        use_euler: bool = False,
    ):
        """
        Args:
            generator: Distilled LTX2DiffusionWrapper
            add_noise_fn: Callable[[original, noise, sigma], noisy_sample]
            denoising_sigmas: Sigma values for few-step denoising
            use_euler: If True, use deterministic Euler stepping.
        """
        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.use_trigflow = use_trigflow
        self.use_euler = use_euler

    @staticmethod
    def _trig_recorrupt(
        clean: torch.Tensor,
        noise: torch.Tensor,
        trig_t: torch.Tensor,
    ) -> torch.Tensor:
        cos_t = torch.cos(trig_t).to(device=clean.device, dtype=clean.dtype)
        sin_t = torch.sin(trig_t).to(device=clean.device, dtype=clean.dtype)
        return cos_t * clean + sin_t * noise

    @torch.no_grad()
    def generate(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Tuple[int, ...],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Optional[Dict[str, Any]] = None,
        video_guidance_scale: float = 1.0,
        audio_guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate video and audio using few-step denoising.

        Args:
            video_shape: (B, F_v, C, H, W) video latent shape
            audio_shape: (B, F_a, C) audio latent shape
            conditional_dict: Text conditioning
            unconditional_dict: Negative/unconditional text conditioning for CFG
            video_guidance_scale: CFG scale for video predictions
            audio_guidance_scale: CFG scale for audio predictions. Defaults to
                video_guidance_scale when unconditional_dict is provided.
            seed: Random seed (optional)

        Returns:
            Tuple of (video_latent, audio_latent)
        """
        B = video_shape[0]
        F_v = video_shape[1]
        F_a = audio_shape[1]

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        device = next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype
        if audio_guidance_scale is None:
            audio_guidance_scale = video_guidance_scale

        # Initialize with noise
        video = torch.randn(video_shape, device=device, dtype=dtype)
        audio = torch.randn(audio_shape, device=device, dtype=dtype)

        # Few-step denoising
        for i, sigma in enumerate(self.denoising_sigmas[:-1]):
            sigma = sigma.to(device=device, dtype=dtype)
            video_sigma = sigma * torch.ones([B, F_v], device=device, dtype=dtype)
            audio_sigma = sigma * torch.ones([B, F_a], device=device, dtype=dtype)

            # Predict x0
            pred_video_cond, pred_audio_cond = self.generator(
                noisy_image_or_video=video,
                conditional_dict=conditional_dict,
                timestep=video_sigma,
                noisy_audio=audio,
                audio_timestep=audio_sigma,
            )
            if unconditional_dict is None:
                pred_video, pred_audio = pred_video_cond, pred_audio_cond
            else:
                pred_video_uncond, pred_audio_uncond = self.generator(
                    noisy_image_or_video=video,
                    conditional_dict=unconditional_dict,
                    timestep=video_sigma,
                    noisy_audio=audio,
                    audio_timestep=audio_sigma,
                )
                pred_video = pred_video_uncond + video_guidance_scale * (
                    pred_video_cond - pred_video_uncond
                )
                pred_audio = pred_audio_uncond + audio_guidance_scale * (
                    pred_audio_cond - pred_audio_uncond
                )

            # Get next sigma
            next_sigma = self.denoising_sigmas[i + 1]

            if next_sigma > 0:
                if self.use_euler:
                    # Deterministic Euler: best quality.
                    video_velocity = (video.float() - pred_video.float()) / sigma.float()
                    audio_velocity = (audio.float() - pred_audio.float()) / sigma.float()
                    dt = (next_sigma - sigma).float()
                    video = (video.float() + video_velocity * dt).to(dtype)
                    audio = (audio.float() + audio_velocity * dt).to(dtype)
                else:
                    fresh_noise_video = torch.randn_like(video)
                    fresh_noise_audio = torch.randn_like(audio)
                    if self.use_trigflow:
                        video = self._trig_recorrupt(pred_video, fresh_noise_video, next_sigma)
                        audio = self._trig_recorrupt(pred_audio, fresh_noise_audio, next_sigma)
                    else:
                        next_video_sigma = next_sigma * torch.ones([B, F_v], device=device)
                        next_audio_sigma = next_sigma * torch.ones([B, F_a], device=device)

                        video = self.add_noise_fn(
                            pred_video.flatten(0, 1),
                            fresh_noise_video.flatten(0, 1),
                            next_video_sigma.flatten(0, 1),
                        ).unflatten(0, (B, F_v))

                        audio = self.add_noise_fn(
                            pred_audio, fresh_noise_audio, next_audio_sigma
                        )
            else:
                video = pred_video
                audio = pred_audio

        return video, audio

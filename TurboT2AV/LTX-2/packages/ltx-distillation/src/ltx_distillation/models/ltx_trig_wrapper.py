"""
LTX-2 Diffusion Model Wrapper with native TrigFlow parameterization.

This wrapper keeps the underlying LTX velocity backbone unchanged, but exposes
the model under the TrigFlow semantics used by rCM:

    x_t = cos(t) * x0 + sin(t) * eps

The wrapped backbone still predicts the RectifiedFlow velocity on the scaled
input/state, while this wrapper handles the TrigFlow <-> RectifiedFlow
preconditioning coefficients directly in the forward pass.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer import LTXModel

from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper


class LTX2TrigFlowDiffusionWrapper(LTX2DiffusionWrapper):
    """
    LTX wrapper that exposes the backbone under native TrigFlow semantics.

    The wrapped `velocity_model` continues to run in the RectifiedFlow/LTX
    parameterization. This wrapper applies the same TrigFlow -> RectifiedFlow
    preconditioning used in rCM so callers can pass TrigFlow time directly.
    """

    def __init__(
        self,
        velocity_model: LTXModel,
        video_height: int = 512,
        video_width: int = 768,
        vae_spatial_compression: int = 32,
    ):
        super().__init__(
            model=velocity_model,
            video_height=video_height,
            video_width=video_width,
            vae_spatial_compression=vae_spatial_compression,
        )

    @staticmethod
    def _trig_coefficients(trig_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_t = torch.cos(trig_t)
        sin_t = torch.sin(trig_t)
        denom = (cos_t + sin_t).clamp_min(1e-8)
        return cos_t, sin_t, denom

    @staticmethod
    def _reshape_time_for_latent(
        trig_t: torch.Tensor,
        latent_dim: int,
    ) -> torch.Tensor:
        if trig_t.dim() == 1:
            return trig_t.reshape(-1, *[1] * (latent_dim - 1))
        if trig_t.dim() == 2:
            return trig_t.reshape(*trig_t.shape, *[1] * (latent_dim - 2))
        raise ValueError(f"Unsupported TrigFlow timestep shape: {trig_t.shape}")

    @staticmethod
    def _rf_time_from_trig(
        trig_t: torch.Tensor,
    ) -> torch.Tensor:
        _, sin_t, denom = LTX2TrigFlowDiffusionWrapper._trig_coefficients(trig_t)
        return sin_t / denom

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, Any],
        timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        use_causal_timestep: bool = False,  # ignored, for API compatibility
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = noisy_image_or_video.shape[0]
        num_video_frames = noisy_image_or_video.shape[1]

        # Scale TrigFlow latent back to RF scale: x_rf = x_trig / (cos+sin).
        video_trig = timestep
        video_trig_bcast = self._reshape_time_for_latent(video_trig, noisy_image_or_video.dim()).to(
            dtype=noisy_image_or_video.dtype,
            device=noisy_image_or_video.device,
        )
        cos_v = torch.cos(video_trig_bcast)
        sin_v = torch.sin(video_trig_bcast)
        denom_v = (cos_v + sin_v).clamp_min(1e-8)
        video_rf_latent = (noisy_image_or_video / denom_v).to(dtype=noisy_image_or_video.dtype)
        # RF timestep: u = sin/(cos+sin) = sigma. Two versions:
        # 1) broadcast shape for x0 = x_rf - u * v
        video_rf_time = (sin_v / denom_v).to(dtype=noisy_image_or_video.dtype)
        # 2) [B] or [B, F_v] shape for model timestep tokens
        video_rf_time_tokens = self._rf_time_from_trig(
            video_trig.to(dtype=noisy_image_or_video.dtype, device=noisy_image_or_video.device)
        )
        if video_rf_time_tokens.dim() == 2 and video_rf_time_tokens.shape[1] == 1:
            video_rf_time_tokens = video_rf_time_tokens[:, 0]

        video_flat = self._flatten_video_latent(video_rf_latent)
        num_video_tokens = video_flat.shape[1]
        video_positions = self._compute_video_positions(noisy_image_or_video)
        video_timesteps = self._compute_timesteps_for_tokens(
            video_rf_time_tokens, num_video_tokens, self.video_frame_seqlen,
        )

        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            latent=video_flat, timesteps=video_timesteps, positions=video_positions,
            context=conditional_dict["video_context"],
            context_mask=conditional_dict.get("attention_mask"), enabled=True,
        )

        audio_modality = None
        audio_rf_time = None
        audio_rf_latent = None
        if noisy_audio is not None:
            if audio_timestep is None:
                audio_timestep = timestep if timestep.dim() == 1 else timestep[:, 0]
            audio_trig = audio_timestep
            audio_trig_bcast = self._reshape_time_for_latent(audio_trig, noisy_audio.dim()).to(
                dtype=noisy_audio.dtype, device=noisy_audio.device,
            )
            cos_a = torch.cos(audio_trig_bcast)
            sin_a = torch.sin(audio_trig_bcast)
            denom_a = (cos_a + sin_a).clamp_min(1e-8)
            audio_rf_latent = (noisy_audio / denom_a).to(dtype=noisy_audio.dtype)
            audio_rf_time = (sin_a / denom_a).to(dtype=noisy_audio.dtype)
            audio_rf_time_tokens = self._rf_time_from_trig(
                audio_trig.to(dtype=noisy_audio.dtype, device=noisy_audio.device)
            )
            if audio_rf_time_tokens.dim() == 2 and audio_rf_time_tokens.shape[1] == 1:
                audio_rf_time_tokens = audio_rf_time_tokens[:, 0]

            num_audio_tokens = audio_rf_latent.shape[1]
            audio_timesteps = self._compute_timesteps_for_tokens(
                audio_rf_time_tokens, num_audio_tokens, 1,
            )
            audio_positions = self._compute_audio_positions(audio_rf_latent)

            audio_modality = Modality(
                latent=audio_rf_latent, timesteps=audio_timesteps, positions=audio_positions,
                context=conditional_dict["audio_context"],
                context_mask=conditional_dict.get("attention_mask"), enabled=True,
            )

        perturbations = BatchedPerturbationConfig.empty(batch_size=B)
        video_v, audio_v = self.model(
            video=video_modality, audio=audio_modality, perturbations=perturbations,
        )

        video_x0 = None
        if video_v is not None:
            video_v = self._unflatten_video_latent(video_v, num_video_frames)
            video_x0 = (video_rf_latent - video_rf_time * video_v).to(dtype=noisy_image_or_video.dtype)

        audio_x0 = None
        if audio_v is not None:
            audio_x0 = (audio_rf_latent - audio_rf_time * audio_v).to(dtype=noisy_audio.dtype)

        return video_x0, audio_x0

    def forward_rf(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, Any],
        timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        use_causal_timestep: bool = False,  # ignored, for API compatibility
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run the wrapper under native RectifiedFlow/LTX semantics.

        This is mainly used by teacher benchmark generation, which still relies
        on the original LTX sigma/RF scheduler and should not reinterpret its
        timesteps as TrigFlow time.
        """
        B = noisy_image_or_video.shape[0]
        num_video_frames = noisy_image_or_video.shape[1]

        video_flat = self._flatten_video_latent(noisy_image_or_video)
        num_video_tokens = video_flat.shape[1]
        video_positions = self._compute_video_positions(noisy_image_or_video)
        video_timesteps = self._compute_timesteps_for_tokens(
            timestep,
            num_video_tokens,
            self.video_frame_seqlen,
        )

        from ltx_core.model.transformer.modality import Modality  # local import to avoid extra top-level churn

        video_modality = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=conditional_dict["video_context"],
            context_mask=conditional_dict.get("attention_mask"),
            enabled=True,
        )

        audio_modality = None
        if noisy_audio is not None:
            num_audio_tokens = noisy_audio.shape[1]

            if audio_timestep is None:
                if timestep.dim() == 1:
                    audio_timestep = timestep
                else:
                    audio_timestep = timestep[:, 0]

            audio_timesteps = self._compute_timesteps_for_tokens(
                audio_timestep,
                num_audio_tokens,
                1,
            )
            audio_positions = self._compute_audio_positions(noisy_audio)

            audio_modality = Modality(
                latent=noisy_audio,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=conditional_dict["audio_context"],
                context_mask=conditional_dict.get("attention_mask"),
                enabled=True,
            )

        perturbations = BatchedPerturbationConfig.empty(batch_size=B)
        video_v, audio_v = self.model(
            video=video_modality,
            audio=audio_modality,
            perturbations=perturbations,
        )

        video_x0 = None
        if video_v is not None:
            video_v = self._unflatten_video_latent(video_v, num_video_frames)
            video_rf_time = self._reshape_time_for_latent(timestep, noisy_image_or_video.dim()).to(
                dtype=noisy_image_or_video.dtype,
                device=noisy_image_or_video.device,
            )
            video_x0 = (noisy_image_or_video - video_rf_time * video_v).to(dtype=noisy_image_or_video.dtype)

        audio_x0 = None
        if audio_v is not None:
            if audio_timestep is None:
                raise ValueError("audio_timestep should be populated before audio_x0 reconstruction.")
            audio_rf_time = self._reshape_time_for_latent(audio_timestep, noisy_audio.dim()).to(
                dtype=noisy_audio.dtype,
                device=noisy_audio.device,
            )
            audio_x0 = (noisy_audio - audio_rf_time * audio_v).to(dtype=noisy_audio.dtype)

        return video_x0, audio_x0


def create_ltx2_trig_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    video_height: int = 512,
    video_width: int = 768,
    registry: Registry | None = None,
) -> LTX2TrigFlowDiffusionWrapper:
    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        registry=registry,
    )

    x0_model = ledger.transformer()
    velocity_model = x0_model.velocity_model.to(device=device, dtype=dtype)

    wrapper = LTX2TrigFlowDiffusionWrapper(
        velocity_model=velocity_model,
        video_height=video_height,
        video_width=video_width,
    )

    return wrapper

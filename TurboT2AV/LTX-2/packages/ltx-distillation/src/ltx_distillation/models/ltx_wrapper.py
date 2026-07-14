"""
LTX-2 diffusion model wrapper for TurboT2AV inference.

This wrapper adapts LTX-2's audio-video joint generation model to the latent
layout used by the TurboT2AV inference runner.

Model Architecture:
- patch_size = (1, 1, 1): No spatial/temporal grouping
- Patchification: Simple reshape [B, C, F, H, W] → [B, F*H*W, C]
- Each token: 128-dimensional latent vector (one per spatial-temporal position)
- Model input projection: Linear(128, 4096)
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from ltx_core.model.transformer import LTXModel, X0Model
from ltx_core.model.transformer.modality import Modality
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.components.patchifiers import (
    VideoLatentPatchifier,
    AudioPatchifier,
    get_pixel_coords,
)
from ltx_core.types import (
    VideoLatentShape,
    AudioLatentShape,
    SpatioTemporalScaleFactors,
)


class LTX2DiffusionWrapper(nn.Module):
    """
    Wrapper for LTX-2 model inference.

    Handles:
    - Input format conversion: [B, F, C, H, W] -> Modality
    - Timestep handling: sigma values for all tokens
    - Position computation for video (3D) and audio (1D)
    - Output format: x0 predictions for both video and audio

    Uses official LTX-2 patchifiers (patch_size=1) to ensure consistency
    with the pretrained model weights.
    """

    # Time alignment constants
    VIDEO_LATENT_FPS = 3.0  # 24fps / 8 (VAE compression)
    AUDIO_LATENT_FPS = 25.0  # 16kHz / 160 / 4 (mel hop / VAE compression)
    ALIGNMENT_RATIO = AUDIO_LATENT_FPS / VIDEO_LATENT_FPS  # ~8.33

    # Video FPS for position computation
    VIDEO_FPS = 24.0

    # VAE scale factors (temporal=8, height=32, width=32)
    DEFAULT_SCALE_FACTORS = SpatioTemporalScaleFactors.default()

    def __init__(
        self,
        model: LTXModel,
        video_height: int = 512,
        video_width: int = 768,
        vae_spatial_compression: int = 32,
    ):
        """
        Args:
            model: X0Model instance (wraps velocity model, returns x0 predictions)
            video_height: Video height in pixels
            video_width: Video width in pixels
            vae_spatial_compression: VAE spatial compression factor
        """
        super().__init__()
        self.model = model
        self.video_height = video_height
        self.video_width = video_width
        self.vae_spatial_compression = vae_spatial_compression

        # Compute latent dimensions
        self.latent_height = video_height // vae_spatial_compression  # 16
        self.latent_width = video_width // vae_spatial_compression    # 24

        # Official patchifiers with patch_size=1 (no spatial grouping)
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)

        # Frame sequence length: with patch_size=1, each spatial position is one token
        # For 512x768: H'*W' = 16*24 = 384 tokens per frame
        self.video_frame_seqlen = self.latent_height * self.latent_width  # 384

    def set_module_grad(self, module_grad: Dict[str, bool]) -> None:
        """
        Set gradient requirements for model components.

        Args:
            module_grad: Dict mapping component names to requires_grad flags
        """
        if module_grad.get("model", True):
            self.model.requires_grad_(True)
        else:
            self.model.requires_grad_(False)
            self.model.eval()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, "velocity_model"):
            self.model.velocity_model.set_gradient_checkpointing(True)
        elif hasattr(self.model, "set_gradient_checkpointing"):
            self.model.set_gradient_checkpointing(True)

    def _flatten_video_latent(
        self,
        video_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flatten video latent from [B, F, C, H, W] to [B, T, C] using patch_size=1.

        With patch_size=1, this is a simple reshape — no spatial grouping.
        The official VideoLatentPatchifier(patch_size=1) does:
            "b c (f 1) (h 1) (w 1) -> b (f h w) (c 1 1 1)" = "b c f h w -> b (f h w) c"

        Args:
            video_latent: Shape [B, F, C, H, W] where
                - F: number of latent frames
                - C: latent channels (128)
                - H, W: latent spatial dimensions (16, 24)

        Returns:
            Flattened tensor [B, T, C] where:
            - T = F * H * W (e.g., 16 * 16 * 24 = 6144)
            - C = 128 (unchanged, since patch_size=1)
        """
        B, F, C, H, W = video_latent.shape
        assert C == 128, (
            f"Expected video latent C=128 at dim 2, got shape {video_latent.shape}. "
            f"Input should be [B, F, C, H, W] with C=128."
        )

        # Convert from [B, F, C, H, W] to [B, C, F, H, W] (official format)
        video_latent = video_latent.permute(0, 2, 1, 3, 4)

        # Use official patchifier: [B, C, F, H, W] -> [B, F*H*W, C]
        # With patch_size=1 this is equivalent to:
        #   einops.rearrange(x, "b c f h w -> b (f h w) c")
        video_latent = self.video_patchifier.patchify(video_latent)

        return video_latent

    def _unflatten_video_latent(
        self,
        flat_latent: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Unflatten video latent from [B, T, C] back to [B, F, C, H, W].

        Args:
            flat_latent: Shape [B, T, C] where C = 128 (patch_size=1)
            num_frames: Number of latent frames F

        Returns:
            Video latent [B, F, C, H, W]
        """
        B, T, C = flat_latent.shape
        H = self.latent_height
        W = self.latent_width
        F = num_frames

        # Use official unpatchifier: [B, T, C] -> [B, C, F, H, W]
        output_shape = VideoLatentShape(
            batch=B, channels=C, frames=F, height=H, width=W
        )
        video_latent = self.video_patchifier.unpatchify(flat_latent, output_shape)

        # Convert from [B, C, F, H, W] to [B, F, C, H, W].
        video_latent = video_latent.permute(0, 2, 1, 3, 4)

        return video_latent

    def _compute_video_positions(
        self,
        video_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 3D position indices for video tokens with [start, end) bounds.

        Uses the official VideoLatentPatchifier.get_patch_grid_bounds() and
        get_pixel_coords() to ensure consistency with the pretrained model.

        The RoPE computation expects positions in the format [B, 3, T, 2] where:
        - dim 1 (size 3): temporal, height, width dimensions
        - dim 3 (size 2): [start, end) bounds for each patch

        Returns:
            Position tensor of shape [B, 3, T, 2] with patch bounds in pixel space
        """
        B, F, C, H, W = video_latent.shape
        device = video_latent.device

        # Build VideoLatentShape for the patchifier
        video_shape = VideoLatentShape(
            batch=B, channels=C, frames=F, height=H, width=W
        )

        # Get patch grid bounds in latent coordinates: [B, 3, T, 2]
        # With patch_size=1, each token covers [i, i+1) in each dimension
        latent_coords = self.video_patchifier.get_patch_grid_bounds(
            output_shape=video_shape,
            device=device,
        )

        # Convert to pixel coordinates using official helper
        # Applies scale_factors (temporal=8, height=32, width=32)
        # and causal_fix (first frame temporal offset)
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.DEFAULT_SCALE_FACTORS,
            causal_fix=True,
        ).float()

        # Convert temporal dimension from frames to seconds (divide by fps=24)
        # This matches VideoLatentTools.create_initial_state
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / self.VIDEO_FPS

        return pixel_coords

    # Audio timing constants (from AudioPatchifier defaults)
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_HOP_LENGTH = 160
    AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
    AUDIO_IS_CAUSAL = True

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts latent indices into real-time seconds while honoring causal
        offsets and the configured hop length.

        Matches AudioPatchifier._get_audio_latent_time_in_sec exactly.
        """
        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)
        audio_mel_frame = audio_latent_frame * self.AUDIO_LATENT_DOWNSAMPLE_FACTOR

        if self.AUDIO_IS_CAUSAL:
            # Frame offset for causal alignment.
            causal_offset = 1
            audio_mel_frame = (audio_mel_frame + causal_offset - self.AUDIO_LATENT_DOWNSAMPLE_FACTOR).clip(min=0)

        return audio_mel_frame * self.AUDIO_HOP_LENGTH / self.AUDIO_SAMPLE_RATE

    def _compute_audio_positions(
        self,
        audio_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 1D temporal positions for audio tokens with [start, end) bounds.

        The RoPE computation expects positions in the format [B, 1, T, 2] where:
        - dim 1 (size 1): temporal dimension only (audio is 1D)
        - dim 3 (size 2): [start, end) bounds in seconds

        Returns:
            Position tensor of shape [B, 1, T, 2] with temporal bounds in seconds
        """
        B, T, C = audio_latent.shape
        device = audio_latent.device

        # Compute start timings for each audio frame
        start_timings = self._get_audio_latent_time_in_sec(
            0, T, torch.float32, device
        )
        start_timings = start_timings.unsqueeze(0).expand(B, -1).unsqueeze(1)  # [B, 1, T]

        # Compute end timings for each audio frame (shifted by 1)
        end_timings = self._get_audio_latent_time_in_sec(
            1, T + 1, torch.float32, device
        )
        end_timings = end_timings.unsqueeze(0).expand(B, -1).unsqueeze(1)  # [B, 1, T]

        # Stack to create [B, 1, T, 2] with [start, end) bounds
        positions = torch.stack([start_timings, end_timings], dim=-1)

        return positions

    def _compute_timesteps_for_tokens(
        self,
        sigma: torch.Tensor,
        num_tokens: int,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        """
        Expand sigma to per-token timesteps.

        In the official pipeline, timesteps = denoise_mask * sigma, producing
        shape [B, T, 1]. Here we replicate sigma to each token belonging to
        the same frame and add a trailing dimension for broadcasting with
        the latent channels.

        Args:
            sigma: Shape [B] or [B, F] - sigma values per frame
            num_tokens: Total number of tokens
            tokens_per_frame: Number of tokens per frame

        Returns:
            Timesteps tensor [B, T, 1] for correct broadcasting with [B, T, C]
        """
        B = sigma.shape[0]

        if sigma.dim() == 1:
            # Single sigma per sample -> expand to all tokens
            return sigma.view(B, 1, 1).expand(B, num_tokens, 1)
        else:
            # Per-frame sigma [B, F] -> expand to per-token [B, T, 1]
            F = sigma.shape[1]
            expanded = sigma.unsqueeze(2).expand(B, F, tokens_per_frame).reshape(B, -1)
            return expanded.unsqueeze(-1)  # [B, T, 1]

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
        """
        Forward pass for audio-video inference.

        Args:
            noisy_image_or_video: Noisy video latent [B, F, C, H, W]
            conditional_dict: Dictionary containing:
                - video_context: [B, seq_len, dim]
                - audio_context: [B, seq_len, dim]
                - attention_mask: [B, seq_len]
            timestep: Sigma values [B] or [B, F]
            noisy_audio: Noisy audio latent [B, F_a, C_audio] (optional)
                where C_audio = 128 (= 8 channels * 16 mel_bins, post-patchify)
            audio_timestep: Audio sigma values [B] or [B, F_a] (optional)

        Returns:
            Tuple of (video_x0_pred, audio_x0_pred)
            - video_x0_pred: [B, F, C, H, W]
            - audio_x0_pred: [B, F_a, C_audio] or None
        """
        B = noisy_image_or_video.shape[0]
        num_video_frames = noisy_image_or_video.shape[1]
        device = noisy_image_or_video.device

        # Flatten video latent: [B, F, C, H, W] -> [B, T, C]
        # With patch_size=1: T = F*H*W, C = 128
        video_flat = self._flatten_video_latent(noisy_image_or_video)
        num_video_tokens = video_flat.shape[1]

        # Compute video positions: [B, 3, T, 2]
        video_positions = self._compute_video_positions(noisy_image_or_video)

        # Compute video timesteps: [B, T, 1]
        video_timesteps = self._compute_timesteps_for_tokens(
            timestep, num_video_tokens, self.video_frame_seqlen
        )

        # Build video modality
        video_modality = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=conditional_dict["video_context"],
            context_mask=conditional_dict.get("attention_mask"),
            enabled=True,
        )

        # Build audio modality if provided
        audio_modality = None
        if noisy_audio is not None:
            num_audio_tokens = noisy_audio.shape[1]

            # Use provided audio timestep or derive from video timestep
            if audio_timestep is None:
                # In bidirectional mode, audio uses same sigma as video.
                # video timestep could be [B] or [B, F_v]. For audio we need [B]
                # or [B, F_a]. If timestep is [B, F_v] (per-frame video), take the
                # first frame's sigma since bidirectional uses uniform sigma anyway.
                if timestep.dim() == 1:
                    audio_timestep = timestep  # [B]
                else:
                    # All video frames have same sigma in bidirectional mode,
                    # take the first frame's value and broadcast to audio frames
                    audio_timestep = timestep[:, 0]  # [B]

            audio_timesteps = self._compute_timesteps_for_tokens(
                audio_timestep, num_audio_tokens, 1  # audio: 1 token per frame
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

        # Forward through model
        perturbations = BatchedPerturbationConfig.empty(batch_size=B)

        # The model returns x0 predictions (X0Model wraps velocity model)
        video_x0, audio_x0 = self.model(
            video=video_modality,
            audio=audio_modality,
            perturbations=perturbations,
        )

        # Unflatten video output: [B, T, C] -> [B, F, C, H, W]
        if video_x0 is not None:
            video_x0 = self._unflatten_video_latent(video_x0, num_video_frames)

        return video_x0, audio_x0

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load state dict, handling potential key mismatches."""
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k] = v
            else:
                new_state_dict[f"model.{k}"] = v

        super().load_state_dict(new_state_dict, strict=strict)


def create_ltx2_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    video_height: int = 512,
    video_width: int = 768,
    registry=None,
) -> LTX2DiffusionWrapper:
    """
    Factory function to create LTX2DiffusionWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype
        video_height: Video height
        video_width: Video width
        registry: Optional shared state registry. Accepted for compatibility
            with callers that coordinate checkpoint caching across wrappers.

    Returns:
        Configured LTX2DiffusionWrapper
    """
    from ltx_pipelines.utils.model_ledger import ModelLedger

    # IMPORTANT: Load to CPU first, then move to target device
    # safetensors doesn't support device indices like "cuda:4"
    # It only accepts "cuda" or "cpu"
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),  # Load to CPU first
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        registry=registry,
    )

    # Get X0Model (wraps velocity model)
    x0_model = ledger.transformer()

    # Move to target device
    x0_model = x0_model.to(device=device, dtype=dtype)

    wrapper = LTX2DiffusionWrapper(
        model=x0_model,
        video_height=video_height,
        video_width=video_width,
    )

    return wrapper

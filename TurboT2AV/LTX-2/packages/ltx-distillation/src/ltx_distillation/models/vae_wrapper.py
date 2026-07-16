"""VAE wrappers for TurboT2AV inference output decoding."""

from typing import Optional
import torch
import torch.nn as nn


class VideoVAEWrapper(nn.Module):
    """
    Wrapper for Video VAE encoder and decoder.

    Used for:
    - Encoding videos to latent space (for visualization)
    - Decoding latents to pixel space (for validation)
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            encoder: VideoEncoder instance (optional)
            decoder: VideoDecoder instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.

        Args:
            video: Pixel video [B, C, F, H, W] in range [-1, 1]

        Returns:
            Latent [B, F', C_latent, H', W']
        """
        if self.encoder is None:
            raise ValueError("Encoder not initialized")

        return self.encoder(video)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pixel space.

        Args:
            latent: Latent [B, F, C, H, W]

        Returns:
            Video [B, C, F_out, H_out, W_out] in range [-1, 1]
        """
        if self.decoder is None:
            raise ValueError("Decoder not initialized")

        # Decoder expects [B, C, F, H, W].
        # TurboT2AV stores video as [B, F, C, H, W] where C=128.
        # Detect this by checking if dim 2 (not dim 1) equals 128.
        if latent.dim() == 5 and latent.shape[2] == 128:
            # Input is [B, F, C, H, W], need to permute to [B, C, F, H, W]
            latent = latent.permute(0, 2, 1, 3, 4)

        return self.decoder(latent)

    @torch.no_grad()
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pixel video for visualization.

        Args:
            latent: Latent [B, F, C, H, W]

        Returns:
            Video frames suitable for logging (normalized to [0, 1])
        """
        video = self.decode(latent)
        # Normalize from [-1, 1] to [0, 1]
        video = (video + 1) / 2
        video = video.clamp(0, 1)
        return video


class AudioVAEWrapper(nn.Module):
    """
    Wrapper for Audio VAE decoder and vocoder.

    Used for:
    - Decoding audio latents to mel spectrogram
    - Converting mel to waveform via vocoder
    """

    def __init__(
        self,
        decoder=None,
        vocoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            decoder: AudioDecoder instance
            vocoder: Vocoder instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.decoder = decoder
        self.vocoder = vocoder
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode audio latent to mel spectrogram.

        TurboT2AV produces audio latents in the transformer's sequence
        format ``[B, T, C*F]`` (3D), but the ``AudioDecoder`` expects the VAE
        spatial format ``[B, C, T, F]`` (4D).  This method handles the
        conversion automatically using the decoder's ``z_channels`` and
        ``mel_bins`` attributes (set during checkpoint loading).

        Args:
            latent: Audio latent, either ``[B, T, C*F]`` (transformer) or
                    ``[B, C, T, F]`` (VAE).

        Returns:
            Mel spectrogram ``[B, out_ch, time, freq]``.
        """
        if self.decoder is None:
            raise ValueError("Decoder not initialized")

        # Reshape 3D transformer latent → 4D VAE latent when necessary.
        # The transformer stores audio as [B, T, C*F] where C=z_channels and
        # F=latent_mel_bins.  The AudioDecoder expects [B, C, T, F].
        # Note: decoder.mel_bins is the *output* spectrogram size (e.g. 64),
        # NOT the latent mel dimension.  The latent mel dim = CF // z_channels.
        if latent.dim() == 3:
            B, T, CF = latent.shape
            z_channels = getattr(self.decoder, "z_channels", None)

            if z_channels is not None:
                latent_mel = CF // z_channels  # e.g. 128 // 8 = 16
                # "b t (c f) -> b c t f"
                latent = latent.reshape(B, T, z_channels, latent_mel).permute(0, 2, 1, 3)
            else:
                raise ValueError(
                    f"Cannot reshape 3D audio latent {latent.shape} to 4D: "
                    "decoder is missing z_channels attribute."
                )

        return self.decoder(latent)

    @torch.no_grad()
    def decode_to_waveform(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode audio latent to waveform.

        Args:
            latent: Audio latent [B, F, C]

        Returns:
            Waveform [B, 1, samples]
        """
        mel = self.decode(latent)

        if self.vocoder is None:
            raise ValueError("Vocoder not initialized")

        # Cast to float32 after vocoder to match the original LTX-2 pipeline's
        # decode_audio() behavior (audio_vae.py:479). The vocoder's 240x upsampling
        # chain amplifies bfloat16 quantization errors into audible high-frequency
        # noise; float32 output prevents this.
        return self.vocoder(mel).float()


def create_vae_wrappers(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry=None,
) -> tuple[VideoVAEWrapper, AudioVAEWrapper]:
    """
    Factory function to create VAE wrappers from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        device: Target device
        dtype: Model dtype
        registry: Optional shared state registry. Accepted for compatibility
            with callers that coordinate checkpoint caching across wrappers.

    Returns:
        Tuple of (VideoVAEWrapper, AudioVAEWrapper)
    """
    from ltx_pipelines.utils.model_ledger import ModelLedger

    # Load to CPU first to avoid safetensors device issues
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        registry=registry,
    )

    video_decoder = ledger.video_decoder()
    audio_decoder = ledger.audio_decoder()
    vocoder = ledger.vocoder()

    # Move to target device
    video_decoder = video_decoder.to(device=device, dtype=dtype)
    audio_decoder = audio_decoder.to(device=device, dtype=dtype)
    vocoder = vocoder.to(device=device, dtype=dtype)

    video_vae = VideoVAEWrapper(
        decoder=video_decoder,
        device=device,
        dtype=dtype,
    )

    audio_vae = AudioVAEWrapper(
        decoder=audio_decoder,
        vocoder=vocoder,
        device=device,
        dtype=dtype,
    )

    return video_vae, audio_vae

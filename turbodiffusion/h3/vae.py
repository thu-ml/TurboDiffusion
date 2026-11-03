"""Public H3 VAEs with native W8A8 video decoding and FP32 keyframe encoding."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import wave

import numpy as np
import torch
import torch.distributed as dist

from .components import load_vae_module


@torch.no_grad()
def apply_video_vae_w8a8(model):
    """Quantize only ViT decoder block projections using the unchanged native ops."""
    from turbodiffusion.ops.core import Int8Linear

    blocks = model.decoder.transformer_blocks
    if not len(blocks):
        raise ValueError("video VAE has no ViT decoder blocks to quantize")
    targets = []
    for block in blocks:
        targets.extend((
            (block.attn, "to_qkv"), (block.attn, "to_out"),
            (block.ff, "w1"), (block.ff, "w2"),
        ))
    # Validate before replacing anything; I/O projections and convolutions stay intact.
    for parent, name in targets:
        layer = getattr(parent, name)
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError(f"VAE {name} must be an unquantized Linear, got {type(layer)}")
        if layer.in_features < 256 or layer.in_features % 128 or layer.out_features % 128:
            raise ValueError(f"unsupported native W8A8 VAE projection: {tuple(layer.weight.shape)}")
        if layer.weight.dtype != torch.bfloat16:
            raise ValueError("convert the video VAE to BF16 before applying native W8A8")
    for parent, name in targets:
        layer = getattr(parent, name)
        quantized = Int8Linear.from_linear(layer, quantize=True)
        # Device-only transfer preserves FP32 scales and BF16 bias.
        setattr(parent, name, quantized.to(device=layer.weight.device).eval())
    return len(targets)


def _stats(path: Path):
    config = json.loads((path / "config.json").read_text())
    if "latents_mean" not in config or "latents_std" not in config:
        raise ValueError(f"missing latent normalization statistics in {path}")
    return tuple(torch.tensor(config[key], dtype=torch.float32)
                 for key in ("latents_mean", "latents_std"))


def normalize_latents(z, mean, std, *, inverse=False):
    shape = [1, -1] + [1] * (z.ndim - 2)
    mean = mean.to(z).view(*shape)
    std = std.to(z).view(*shape)
    return z * std + mean if inverse else (z - mean) / std


def pixels_to_uint8(frames, norm_type="imagenet"):
    norms = {
        "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "simple": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "raw": ((0., 0., 0.), (1., 1., 1.)),
    }
    mean, std = norms[norm_type]
    if frames.ndim == 4:
        frames = frames.unsqueeze(0)
    shape = (1, 3, 1, 1, 1)
    mean = torch.tensor(mean, device=frames.device).view(shape)
    std = torch.tensor(std, device=frames.device).view(shape)
    result = torch.empty(frames.shape, dtype=torch.uint8, device="cpu")
    for start in range(0, frames.shape[2], 32):
        chunk = frames[:, :, start:start + 32].float() * std + mean
        result[:, :, start:start + 32] = (
            chunk.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
        )
    return result


class H3VAEProcessor:
    def __init__(self, cfg):
        self.cfg = cfg["vae"]
        self.video_precision = self.cfg.get("video_decoder_precision", "w8a8")
        if self.video_precision not in ("w8a8", "bf16"):
            raise ValueError("vae.video_decoder_precision must be w8a8 or bf16")
        self.video_path = Path(self.cfg["video_vae_path"])
        self.audio_path = Path(self.cfg["audio_vae_path"])
        self.trust = bool(self.cfg.get("trust_model_code", False))
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.video_module = load_vae_module(self.video_path, "video", trust_model_code=self.trust)
        self.video_mean, self.video_std = _stats(self.video_path)
        self.video_config = json.loads((self.video_path / "config.json").read_text())
        state = self.video_module.get_parallel_state()
        state.clear()
        state.update(
            group_size=self.world, group_rank=self.rank,
            local_process_group=dist.group.WORLD if dist.is_initialized() else None,
            sp_size=self.world, sp_rank=self.rank, sp_enabled=self.world > 1,
            sp_process_group=dist.group.WORLD if dist.is_initialized() else None,
            tp_size=1, tp_rank=0,
        )

    def _configure_tiles(self, model, height, width, *, decode):
        model.tile_size = int(self.cfg.get("tile_size", 256))
        model.tile_overlap_min = int(self.cfg.get("tile_overlap_min", 64))
        if hasattr(model, "decoder_tile_size"):
            model.decoder_tile_size = model.tile_size
            model.decoder_tile_overlap_min = model.tile_overlap_min
        model.decoder_tiling = True
        model.parallel_tiling = bool(self.cfg.get("parallel_tiling", True) and self.world > 1)
        if model.parallel_tiling:
            ny = len(model.split_tiles(height, decode)[0])
            nx = len(model.split_tiles(width, decode)[0])
            model.parallel_tiling = ny * nx >= self.world

    def _load_encoder(self):
        """Materialize only the FP32 encoder tensors, not the 2.4B decoder."""
        from safetensors import safe_open

        config = self.video_config
        source = self.video_path / config["source_path"]
        source_cls = self.video_module._SOURCE_CLASSES[config["source_class_name"]]
        options = {
            name: config["vae_" + name] for name in (
                "clip_length", "token_drop", "encoder_tiling", "decoder_tiling",
                "parallel_tiling", "tile_size", "tile_overlap_min",
                "encoder_parallel", "decoder_parallel", "chunk_dim",
            )
        }
        with torch.device("meta"):
            model, _ = source_cls.from_config(
                source_cls.load_config(str(source)), return_unused_kwargs=True, **options
            )
        for name in ("decoder", "post_quant_conv"):
            if hasattr(model, name):
                delattr(model, name)
        state = {}
        with safe_open(str(source / config["source_safetensors_path"]), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(("encoder.", "quant_conv.")):
                    state[key] = f.get_tensor(key)
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected or any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
            raise RuntimeError(f"incomplete VAE encoder: missing={missing}, unexpected={unexpected}")
        return model.to(self.device).eval()

    @torch.no_grad()
    def encode_image(self, path, height, width):
        from PIL import Image

        model = self._load_encoder()
        self._configure_tiles(model, height, width, decode=False)
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            latent = model.encode_images([pixels], use_fp16_latent=True)[0]
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        if latent.ndim != 5:
            raise ValueError(f"unexpected keyframe latent shape: {tuple(latent.shape)}")
        latent = normalize_latents(latent[:, :, :1].float().cpu(), self.video_mean, self.video_std)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return latent

    @torch.no_grad()
    def decode(self, video_latents, audio_latents):
        video = self.video_module.MiniMaxH3VideoVAE.from_pretrained(str(self.video_path))
        video.to(device=self.device, dtype=torch.bfloat16).eval()
        if self.video_precision == "w8a8":
            count = apply_video_vae_w8a8(video.model)
            if self.rank == 0:
                print(f"[vae] native TurboDiffusion W8A8: {count} decoder projections", flush=True)
        elif self.rank == 0:
            print("[vae] BF16 video decoder", flush=True)
        self._configure_tiles(video.model, video_latents.shape[-2] * 16,
                              video_latents.shape[-1] * 16, decode=True)
        z = normalize_latents(video_latents.to(self.device, torch.bfloat16),
                              self.video_mean, self.video_std, inverse=True)
        frames = video.decode_base(z)
        if isinstance(frames, tuple):
            frames = frames[0]
        frames = (
            pixels_to_uint8(frames, self.video_config.get("pixel_norm_type", "imagenet"))
            if self.rank == 0 else None
        )
        del video, z
        gc.collect()
        torch.cuda.empty_cache()
        audio = None
        if self.rank == 0:
            module = load_vae_module(self.audio_path, "audio", trust_model_code=self.trust)
            vae = module.MiniMaxH3AudioVAE.from_pretrained(str(self.audio_path)).to(self.device).eval()
            mean, std = _stats(self.audio_path)
            z = audio_latents.float().transpose(1, 2).contiguous()
            z = normalize_latents(z, mean, std, inverse=True)
            audio = vae.decode(z.to(self.device))
            if isinstance(audio, tuple):
                audio = audio[0]
            audio = audio.cpu()
            del vae, z
            torch.cuda.empty_cache()
        return frames, audio

    def export(self, frames, audio, output_dir, *, user_hw=None, fps=24):
        output_dir = Path(output_dir)
        frames = frames[0].permute(1, 2, 3, 0).contiguous().numpy()
        _, height, width, _ = frames.shape
        command = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-", "-an",
        ]
        if user_hw and tuple(user_hw) != (height, width):
            command += ["-vf", f"scale={user_hw[1]}:{user_hw[0]}:flags=bicubic"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output_dir / "video.mp4")]
        result = subprocess.run(command, input=frames.tobytes(), capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode()[-1000:])
        if audio is None:
            raise RuntimeError("H3 audio VAE returned no waveform")
        wav = audio.detach().float().cpu()
        if wav.ndim == 3:
            wav = wav[:, 0, :] if wav.shape[1] == 1 else wav[0]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        sample_rate = json.loads((self.audio_path / "config.json").read_text()).get("sample_rate", 32000)
        pcm = (wav.clamp(-1, 1).numpy() * 32767).astype(np.int16).T.copy()
        with wave.open(str(output_dir / "audio.wav"), "wb") as stream:
            stream.setnchannels(wav.shape[0])
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes(pcm.tobytes())
        subprocess.run([
            "ffmpeg", "-y", "-i", str(output_dir / "video.mp4"),
            "-i", str(output_dir / "audio.wav"), "-c:v", "copy", "-c:a", "aac",
            str(output_dir / "i2va.mp4"),
        ], check=True, capture_output=True)

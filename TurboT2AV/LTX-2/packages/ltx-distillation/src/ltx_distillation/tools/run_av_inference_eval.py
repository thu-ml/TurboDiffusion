"""Standalone audio-video inference runner."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
import types
from contextlib import contextmanager, nullcontext
from typing import Any, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader.registry import DummyRegistry, StateDictRegistry
from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
from ltx_distillation.models.ltx_trig_wrapper import create_ltx2_trig_wrapper
from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ltx_distillation.time_utils import rf_to_trig_time


def compute_latent_shapes(
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> Tuple[list, list]:
    """Compute LTX-2 video/audio latent shapes from output dimensions."""
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(
            f"num_frames must be 1 + {vae_temporal_compression}*k, got {num_frames}."
        )

    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression

    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    audio_frames = round(video_duration * audio_latent_fps)

    return (
        [batch_size, latent_frames, latent_channels, latent_h, latent_w],
        [batch_size, audio_frames, latent_channels],
    )


def _load_prompts(prompts_file: str, limit: int | None) -> list[str]:
    import csv
    with open(prompts_file, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        f.seek(0)
        if prompts_file.endswith(".csv") or ("," in first and "video_id" in first):
            prompts = [row.get("prompt", row.get("caption", "")).strip()
                      for row in csv.DictReader(f) if row]
        else:
            prompts = [line.strip() for line in f if line.strip()]
    prompts = [p for p in prompts if p]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def _selected_indices(num_prompts: int, num_shards: int, shard_id: int) -> list[int]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")
    return [i for i in range(num_prompts) if i % num_shards == shard_id]


def _student_sigmas(cfg: Any, device: torch.device, dtype: torch.dtype, force_trig: bool) -> torch.Tensor:
    if force_trig:
        trig_steps = [math.pi / 2, *[float(t) for t in getattr(cfg, "backward_trig_timesteps", [1.5, 1.4, 1.0])], 0.0]
        return torch.tensor(trig_steps, device=device, dtype=dtype)

    full_sigmas = LTX2Scheduler().execute(steps=int(getattr(cfg, "num_inference_steps", 40)))
    denoising_sigmas = []
    for t in getattr(cfg, "denoising_step_list", [1000, 757, 522, 0]):
        target_sigma = float(t) / 1000.0
        idx = (full_sigmas - target_sigma).abs().argmin().item()
        denoising_sigmas.append(full_sigmas[idx])
    return torch.stack(denoising_sigmas).to(device=device, dtype=dtype)


def _load_generator_state(generator: torch.nn.Module, checkpoint_path: str, strict: bool) -> None:
    load_kwargs = {"map_location": "cpu", "mmap": True}
    try:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    except (TypeError, RuntimeError) as exc:
        # Older torch versions or non-file-backed checkpoints may not support
        # mmap. Fall back to the normal CPU load in that case.
        if isinstance(exc, RuntimeError) and "mmap" not in str(exc).lower():
            raise
        load_kwargs.pop("mmap", None)
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    state_dict = checkpoint.get("generator", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    result = generator.load_state_dict(state_dict, strict=strict)
    if result is not None:
        missing, unexpected = result
        if missing:
            print(f"[StudentEval] missing keys: {len(missing)} first={missing[:5]}", flush=True)
        if unexpected:
            print(f"[StudentEval] unexpected keys: {len(unexpected)} first={unexpected[:5]}", flush=True)
    del checkpoint, state_dict
    gc.collect()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _make_registry(cache_state_dicts: bool):
    if cache_state_dicts:
        return StateDictRegistry()
    return DummyRegistry()


def _resolve_init_lock_path(
    output_dir: str,
    requested_path: str | None,
    num_shards: int,
    disabled: bool,
    checkpoint_path: str,
    gemma_path: str | None,
) -> str | None:
    if disabled or _env_flag("AV_EVAL_NO_INIT_LOCK", False):
        return None
    if requested_path:
        return requested_path
    if num_shards <= 1:
        return None
    try:
        key = "\0".join(
            str(os.path.abspath(path))
            for path in (checkpoint_path, gemma_path)
            if path
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        lock_dir = os.environ.get("AV_EVAL_INIT_LOCK_DIR", "/tmp")
        return os.path.join(lock_dir, f"ltx_av_eval_model_init_{digest}.lock")
    except Exception:
        return os.path.join(output_dir, ".av_eval_model_init.lock")


@contextmanager
def _model_init_lock(lock_path: str | None, shard_id: int):
    if lock_path is None:
        with nullcontext():
            yield
        return

    import fcntl

    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        print(f"[AVEval] shard={shard_id} waiting for model-init lock {lock_path}", flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        print(f"[AVEval] shard={shard_id} acquired model-init lock", flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            print(f"[AVEval] shard={shard_id} released model-init lock", flush=True)


def _shape(value: Any) -> tuple[int, ...] | str:
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def _install_shape_debug(model: torch.nn.Module) -> None:
    if os.environ.get("AV_EVAL_DEBUG_SHAPES", "0") != "1":
        return

    original_forward = model.forward

    def debug_forward(self, *args, **kwargs):
        conditional_dict = kwargs.get("conditional_dict", {})
        print(
            "[AVEval][debug] wrapper "
            f"video={_shape(kwargs.get('noisy_image_or_video'))} "
            f"audio={_shape(kwargs.get('noisy_audio'))} "
            f"video_t={_shape(kwargs.get('timestep'))} "
            f"audio_t={_shape(kwargs.get('audio_timestep'))} "
            f"video_ctx={_shape(conditional_dict.get('video_context'))} "
            f"audio_ctx={_shape(conditional_dict.get('audio_context'))} "
            f"mask={_shape(conditional_dict.get('attention_mask'))}",
            flush=True,
        )
        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(debug_forward, model)

    try:
        velocity_model = model.model.velocity_model
        preprocessor = velocity_model.audio_args_preprocessor
        simple_preprocessor = getattr(preprocessor, "simple_preprocessor", preprocessor)
        original_prepare = simple_preprocessor.prepare

        def debug_prepare(modality):
            x = simple_preprocessor.patchify_proj(modality.latent)
            print(
                "[AVEval][debug] audio_prepare "
                f"latent={_shape(modality.latent)} x={_shape(x)} "
                f"timesteps={_shape(modality.timesteps)} positions={_shape(modality.positions)} "
                f"context={_shape(modality.context)} mask={_shape(modality.context_mask)}",
                flush=True,
            )
            del x
            return original_prepare(modality)

        simple_preprocessor.prepare = debug_prepare
    except Exception as exc:
        print(f"[AVEval][debug] install audio preprocessor hook failed: {exc}", flush=True)


def _add_noise(original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    while sigma.dim() < original.dim():
        sigma = sigma.unsqueeze(-1)
    return ((1 - sigma) * original + sigma * noise).to(original.dtype)


def _decode_and_save_sample(
    video_vae,
    audio_vae,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    prompt_idx: int,
    prompt: str,
    output_dir: str,
    video_fps: int,
    audio_sample_rate: int,
) -> None:
    video_dir = os.path.join(output_dir, "video")
    audio_dir = os.path.join(output_dir, "audio")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    video_pixel = video_vae.decode_to_pixel(video_latent)
    audio_waveform = audio_vae.decode_to_waveform(audio_latent)

    vid = video_pixel[0]
    if vid.shape[0] == 3:
        vid = vid.permute(1, 0, 2, 3)
    vid = vid.permute(0, 2, 3, 1)
    vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

    sample_stem = f"sample_{prompt_idx:04d}"
    mp4_path = os.path.join(video_dir, f"{sample_stem}.mp4")
    wav_path = os.path.join(audio_dir, f"{sample_stem}.wav")
    json_path = os.path.join(json_dir, f"{sample_stem}.json")

    from torchvision.io import write_video

    # Keep a separate wav even if the mp4 mux succeeds; JavisBench asserts that
    # sample_XXXX.wav exists in infer_data_dir.
    wav_float = audio_waveform[0].cpu().float()
    try:
        write_video(
            mp4_path,
            vid,
            fps=video_fps,
            audio_array=wav_float,
            audio_fps=audio_sample_rate,
            audio_codec="aac",
        )
    except Exception as exc:
        print(f"[Decode] write_video with audio failed for {sample_stem}: {exc}", flush=True)
        write_video(mp4_path, vid, fps=video_fps)

    _save_wav(wav_path, wav_float, audio_sample_rate)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"index": prompt_idx, "prompt": prompt, "mp4": mp4_path, "wav": wav_path}, f, ensure_ascii=False)

    del video_pixel, audio_waveform
    torch.cuda.empty_cache()


def _save_wav(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    wav = waveform.detach().cpu().float()
    if wav.ndim == 2 and wav.shape[0] <= 8 and wav.shape[1] > wav.shape[0]:
        wav = wav.transpose(0, 1)
    if wav.ndim == 2 and wav.shape[1] == 1:
        wav = wav[:, 0]

    data = wav.numpy()
    data = np.clip(data, -1.0, 1.0)
    pcm16 = (data * 32767.0).astype(np.int16)

    from scipy.io import wavfile

    wavfile.write(path, sample_rate, pcm16)


@torch.no_grad()
def _generate_teacher_sample(
    teacher,
    video_shape: Tuple[int, ...],
    audio_shape: Tuple[int, ...],
    sigmas: torch.Tensor,
    conditional_dict: dict[str, Any],
    unconditional_dict: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    video_cfg: float,
    audio_cfg: float,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = video_shape[0]
    video_frames = video_shape[1]
    audio_frames = audio_shape[1]

    video = torch.randn(video_shape, device=device, dtype=dtype)
    audio = torch.randn(audio_shape, device=device, dtype=dtype)

    if mode == "native_rf":
        teacher_forward = teacher.forward_rf if hasattr(teacher, "forward_rf") else teacher
        schedule = sigmas
    elif mode == "rcm_trig":
        teacher_forward = teacher
        schedule = rf_to_trig_time(sigmas.double()).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported teacher mode: {mode}")

    for i in range(len(schedule) - 1):
        sigma = schedule[i]
        video_sigma = sigma * torch.ones([bsz, video_frames], device=device, dtype=dtype)
        audio_sigma = sigma * torch.ones([bsz, audio_frames], device=device, dtype=dtype)

        video_x0_cond, audio_x0_cond = teacher_forward(
            noisy_image_or_video=video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=audio,
            audio_timestep=audio_sigma,
        )
        video_x0_uncond, audio_x0_uncond = teacher_forward(
            noisy_image_or_video=video,
            conditional_dict=unconditional_dict,
            timestep=video_sigma,
            noisy_audio=audio,
            audio_timestep=audio_sigma,
        )

        video_x0 = video_x0_uncond + video_cfg * (video_x0_cond - video_x0_uncond)
        audio_x0 = audio_x0_uncond + audio_cfg * (audio_x0_cond - audio_x0_uncond)

        sigma_next = schedule[i + 1]
        if sigma_next > 0 and sigma > 0:
            if mode == "native_rf":
                video_velocity = (video.float() - video_x0.float()) / sigma.float()
                audio_velocity = (audio.float() - audio_x0.float()) / sigma.float()
                dt = (sigma_next - sigma).float()
                video = (video.float() + video_velocity * dt).to(dtype)
                audio = (audio.float() + audio_velocity * dt).to(dtype)
            else:
                next_t_video = sigma_next.view(1, 1, 1, 1, 1).to(device=device, dtype=dtype)
                next_t_audio = sigma_next.view(1, 1, 1).to(device=device, dtype=dtype)
                video = (torch.cos(next_t_video) * video_x0 + torch.sin(next_t_video) * torch.randn_like(video)).to(dtype)
                audio = (torch.cos(next_t_audio) * audio_x0 + torch.sin(next_t_audio) * torch.randn_like(audio)).to(dtype)
        else:
            video = video_x0
            audio = audio_x0

    return video, audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone student/teacher AV inference for evaluation")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_kind", choices=["student", "teacher"], required=True)
    parser.add_argument("--student_checkpoint", default=None)
    parser.add_argument("--student_param", choices=["auto", "native_rf", "rcm_trig"], default="auto")
    parser.add_argument("--student_strict", action="store_true", default=False)
    parser.add_argument("--teacher_mode", choices=["native_rf", "rcm_trig"], default="native_rf")
    parser.add_argument("--teacher_steps", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--cache_state_dicts",
        action="store_true",
        default=False,
        help="Keep loaded checkpoint state dicts in CPU RAM during initialization. Disabled by default to reduce RAM.",
    )
    parser.add_argument(
        "--init_lock_path",
        default=None,
        help="Path for the model-initialization file lock. Defaults to a checkpoint-keyed lock in /tmp for sharded runs.",
    )
    parser.add_argument(
        "--no_init_lock",
        action="store_true",
        default=False,
        help="Disable the default model-initialization lock for multi-shard launches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    # Override model paths from environment.
    import os as _os
    for key, env in [
        ("checkpoint_path", "TURBO_CHECKPOINT_PATH"),
        ("gemma_path", "TURBO_GEMMA_PATH"),
    ]:
        if _os.environ.get(env):
            cfg[key] = _os.environ[env]

    if args.model_kind == "student" and not args.student_checkpoint:
        raise ValueError("--student_checkpoint is required for --model_kind student")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if bool(getattr(cfg, "mixed_precision", True)) else torch.float32

    prompts = _load_prompts(args.prompts_file, args.num_prompts)
    indices = _selected_indices(len(prompts), args.num_shards, args.shard_id)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"prompts_shard_{args.shard_id:02d}.txt"), "w", encoding="utf-8") as f:
        for idx in indices:
            f.write(f"{idx}\t{prompts[idx]}\n")

    print(
        f"[AVEval] kind={args.model_kind} prompts={len(prompts)} shard={args.shard_id}/{args.num_shards} "
        f"selected={len(indices)} output={args.output_dir}",
        flush=True,
    )

    cache_state_dicts = bool(args.cache_state_dicts or _env_flag("AV_EVAL_CACHE_STATE_DICTS", False))
    init_lock_path = _resolve_init_lock_path(
        output_dir=args.output_dir,
        requested_path=args.init_lock_path,
        num_shards=args.num_shards,
        disabled=args.no_init_lock,
        checkpoint_path=str(cfg.checkpoint_path),
        gemma_path=str(getattr(cfg, "gemma_path", "")),
    )
    force_trig = False

    with _model_init_lock(init_lock_path, args.shard_id):
        registry = _make_registry(cache_state_dicts)

        if args.model_kind == "teacher":
            force_trig = args.teacher_mode == "rcm_trig"
            wrapper_factory = create_ltx2_trig_wrapper if force_trig else create_ltx2_wrapper
            model = wrapper_factory(
                checkpoint_path=cfg.checkpoint_path,
                gemma_path=cfg.gemma_path,
                device=device,
                dtype=dtype,
                video_height=int(cfg.video_height),
                video_width=int(cfg.video_width),
                registry=registry,
            ).eval()
        else:
            dmd_style = str(getattr(cfg, "dmd_style", "legacy")).lower()
            force_trig = args.student_param == "rcm_trig" or (
                args.student_param == "auto" and dmd_style in {"rcm", "rcm_trig", "trig"}
            )
            wrapper_factory = create_ltx2_trig_wrapper if force_trig else create_ltx2_wrapper
            model = wrapper_factory(
                checkpoint_path=cfg.checkpoint_path,
                gemma_path=cfg.gemma_path,
                device=device,
                dtype=dtype,
                video_height=int(cfg.video_height),
                video_width=int(cfg.video_width),
                registry=registry,
            ).eval()
            _load_generator_state(model, args.student_checkpoint, args.student_strict)
            model.eval()

        _install_shape_debug(model)

        if os.environ.get("AV_EVAL_DEBUG_SHAPES", "0") == "1":
            try:
                velocity_model = model.model.velocity_model
                probe = torch.zeros((1, 126, 128), device=device, dtype=dtype)
                probe_out = velocity_model.audio_patchify_proj(probe)
                print(
                    f"[AVEval] audio_patchify_proj={type(velocity_model.audio_patchify_proj).__name__} "
                    f"probe_in={tuple(probe.shape)} probe_out={tuple(probe_out.shape)}",
                    flush=True,
                )
                del probe, probe_out
            except Exception as exc:
                print(f"[AVEval] audio_patchify_probe failed: {exc}", flush=True)

        text_encoder = create_text_encoder_wrapper(
            checkpoint_path=cfg.checkpoint_path,
            gemma_path=cfg.gemma_path,
            device=device,
            dtype=dtype,
            registry=registry,
        ).eval()
        video_vae, audio_vae = create_vae_wrappers(
            checkpoint_path=cfg.checkpoint_path,
            device=device,
            dtype=dtype,
            registry=registry,
        )
        video_vae.eval()
        audio_vae.eval()

        registry.clear()
        del registry
        gc.collect()

    video_shape, audio_shape = compute_latent_shapes(
        num_frames=int(cfg.num_frames),
        video_height=int(cfg.video_height),
        video_width=int(cfg.video_width),
        batch_size=1,
    )
    print(f"[AVEval] latent_shapes video={video_shape} audio={audio_shape}", flush=True)

    if args.model_kind == "teacher":
        sigmas = LTX2Scheduler().execute(steps=int(args.teacher_steps)).to(device=device, dtype=dtype)
        pipeline = None
    else:
        sigmas = _student_sigmas(cfg, device=device, dtype=dtype, force_trig=force_trig)
        pipeline = BidirectionalAVInferencePipeline(
            generator=model,
            add_noise_fn=_add_noise,
            denoising_sigmas=sigmas,
            use_trigflow=force_trig,
        )

    negative_prompt = str(cfg.negative_prompt)
    start = time.perf_counter()
    for local_pos, prompt_idx in enumerate(indices, start=1):
        sample_stem = f"sample_{prompt_idx:04d}"
        mp4_path = os.path.join(args.output_dir, "video", f"{sample_stem}.mp4")
        wav_path = os.path.join(args.output_dir, "audio", f"{sample_stem}.wav")
        json_path = os.path.join(args.output_dir, "json", f"{sample_stem}.json")
        if (
            not args.overwrite
            and os.path.exists(mp4_path)
            and os.path.exists(wav_path)
            and os.path.exists(json_path)
        ):
            print(f"[AVEval] skip existing index={prompt_idx} ({local_pos}/{len(indices)})", flush=True)
            continue

        prompt = prompts[prompt_idx]
        conditional_dict = text_encoder(text_prompts=[prompt])
        prompt_seed = int(args.seed) + prompt_idx

        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(prompt_seed)
            torch.cuda.manual_seed(prompt_seed)
            gen_start = time.perf_counter()
            if args.model_kind == "teacher":
                unconditional_dict = text_encoder(text_prompts=[negative_prompt])
                video_latent, audio_latent = _generate_teacher_sample(
                    teacher=model,
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    sigmas=sigmas,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    device=device,
                    dtype=dtype,
                    video_cfg=float(getattr(cfg, "teacher_benchmark_video_guidance_scale", 3.0)),
                    audio_cfg=float(getattr(cfg, "teacher_benchmark_audio_guidance_scale", 5.0)),
                    mode=args.teacher_mode,
                )
                del unconditional_dict
            else:
                video_latent, audio_latent = pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                )
            gen_elapsed = time.perf_counter() - gen_start

        _decode_and_save_sample(
            video_vae=video_vae,
            audio_vae=audio_vae,
            video_latent=video_latent,
            audio_latent=audio_latent,
            prompt_idx=prompt_idx,
            prompt=prompt,
            output_dir=args.output_dir,
            video_fps=int(getattr(cfg, "benchmark_video_fps", 24)),
            audio_sample_rate=int(getattr(cfg, "benchmark_audio_sample_rate", 24000)),
        )

        print(
            f"[AVEval] saved index={prompt_idx} ({local_pos}/{len(indices)}) seed={prompt_seed} "
            f"gen={gen_elapsed:.2f}s",
            flush=True,
        )
        del conditional_dict, video_latent, audio_latent
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start
    print(
        f"[AVEval] done kind={args.model_kind} shard={args.shard_id}/{args.num_shards} "
        f"selected={len(indices)} wall={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

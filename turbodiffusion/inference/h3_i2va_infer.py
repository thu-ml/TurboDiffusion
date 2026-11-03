#!/usr/bin/env python3
"""Run the full H3 DMD model with TurboDiffusion W8A8 and optional conservative DBCache.

The bundled turbodiffusion.h3 package implements conditioning, sampling,
offload and decoding. W8A8 uses this repository's existing CUDA operators.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/h3_i2va_w8a8.yaml"
CONSERVATIVE_CACHE = {
    "enabled": True,
    "probe_only": False,
    "Fn_compute_blocks": 8,
    "Bn_compute_blocks": 4,
    "max_warmup_steps": 1,
    "max_continuous_cached_steps": 1,
    "residual_diff_threshold": 0.20,
    "enable_taylorseer": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--model-root", type=Path, help="upstream FL2VA shared text/video/audio encoders")
    ap.add_argument("--dit-model-path", type=Path, help="full BF16 DMD transformer checkpoint directory")
    ap.add_argument("--w8a8-path", type=Path, help="exported H3 TurboDiffusion INT8 safetensors")
    ap.add_argument("--adaln-path", type=Path, help="matching i2va / 5-point AdaLN bake")
    ap.add_argument("--vae-precision", choices=("w8a8", "bf16"),
                    help="video decoder block precision (default: native W8A8)")
    ap.add_argument("--trust-model-code", action="store_true",
                    help="allow the public VAE Python modules in the downloaded FL2VA model")
    ap.add_argument("--profile", choices=("fullopt", "fullopt-dbcache"), default="fullopt",
                    help="default: fullopt; use fullopt-dbcache to enable conservative DBCache")
    ap.add_argument("--image", type=Path, help="first frame; required when encoding")
    ap.add_argument("--prompt", help="H3 prompt; required when encoding")
    ap.add_argument("--prompt-embeds", type=Path, help="reuse an existing encoded prompt")
    ap.add_argument("--visual-cond", type=Path, help="reuse an existing encoded keyframe")
    ap.add_argument("--skip-encode", action="store_true", help="use inputs already in output-dir")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output/h3_i2va")
    ap.add_argument("--seed", type=int, help="override config seed")
    ap.add_argument("--duration-seconds", type=float, help="override config duration")
    ap.add_argument("--resolution", choices=("360p", "480p", "540p", "720p", "768p"))
    ap.add_argument("--short-edge", type=int)
    ap.add_argument("--aspect-ratio", help="width:height; default follows the first image")
    ap.add_argument("--nproc", type=int, default=8, help="DiT / VAE GPU ranks")
    ap.add_argument("--encode-nproc", type=int, default=4, help="Qwen TP / encode GPU ranks")
    ap.add_argument("--master-port", type=int, default=14770)
    ap.add_argument("--dry-run", action="store_true", help="write effective config and print commands")
    return ap.parse_args(argv)


def _config_root(config_path: Path) -> Path:
    if config_path == DEFAULT_CONFIG:
        return ROOT
    return config_path.parent


def effective_config(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"expected a mapping in {config_path}")
    base = _config_root(config_path)

    def absolute(value: str | Path, relative_to: Path = base) -> str:
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else relative_to / path).resolve())

    cfg["model_root"] = absolute(
        args.model_root or cfg["model_root"], Path.cwd() if args.model_root else base
    )
    dit = cfg.setdefault("dit", {})
    vae = cfg.setdefault("vae", {})
    for section, key, override in (
        (dit, "model_path", args.dit_model_path),
        (dit, "turbo_int8_path", args.w8a8_path),
        (dit, "adaln_bake_path", args.adaln_path),
    ):
        section[key] = absolute(
            override or section[key], Path.cwd() if override else base
        )
    if dit.get("int8_path") or vae.get("int8_path") or cfg.get("beacon_params") or vae.get("beacon_params"):
        raise ValueError("H3 uses native TurboDiffusion W8A8; legacy quantization configs are unsupported")
    vae["video_decoder_precision"] = args.vae_precision or vae.get("video_decoder_precision", "w8a8")
    if vae["video_decoder_precision"] not in ("w8a8", "bf16"):
        raise ValueError("vae.video_decoder_precision must be w8a8 or bf16")
    for key, subdir in (("video_vae_path", "video_vae"), ("audio_vae_path", "audio_vae")):
        vae[key] = absolute(vae.get(key) or Path(cfg["model_root"]) / subdir)
    vae["trust_model_code"] = bool(args.trust_model_code)
    if int(dit.get("steps", 0)) != 5 or (float(dit.get("video_shift", 0)),
                                          float(dit.get("audio_shift", 0))) != (12.0, 3.0):
        raise ValueError("the shipped AdaLN bake requires steps=5, video_shift=12, audio_shift=3")
    dit["dbcache"] = (
        dict(CONSERVATIVE_CACHE) if args.profile == "fullopt-dbcache" else {"enabled": False}
    )

    text = cfg.setdefault("text_encoder", {})
    text.setdefault("engine", "tp")
    text.setdefault("tp_size", args.encode_nproc)
    sample = cfg.setdefault("sample", {})
    sample["task"] = "i2va"
    sample["encode_aspect_ratio"] = args.aspect_ratio
    if args.image:
        sample["image"] = str(args.image.resolve())
    for key, value in (
        ("seed", args.seed), ("duration_seconds", args.duration_seconds),
        ("short_edge", args.short_edge), ("resolution", args.resolution),
        ("aspect_ratio", args.aspect_ratio), ("prompt", args.prompt),
    ):
        if value is not None:
            sample[key] = value
    cfg["output_dir"] = str(args.output_dir.resolve())
    return cfg


def validate_inputs(args: argparse.Namespace, cfg: dict) -> None:
    if not args.dry_run and not args.trust_model_code:
        raise ValueError("pass --trust-model-code to load the public H3 VAE implementation")
    for path in (
        Path(cfg["model_root"]),
        Path(cfg["dit"]["model_path"]) / "config.json",
        Path(cfg["dit"]["turbo_int8_path"]),
        Path(cfg["dit"]["adaln_bake_path"]),
        Path(cfg["dit"]["adaln_bake_path"]).with_suffix(".json"),
        Path(cfg["vae"]["video_vae_path"]) / "config.json",
        Path(cfg["vae"]["video_vae_path"]) / "minimax_h3_video_vae.py",
        Path(cfg["vae"]["audio_vae_path"]) / "config.json",
        Path(cfg["vae"]["audio_vae_path"]) / "minimax_h3_audio_vae.py",
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing H3 model artifact: {path}")
    if args.nproc != 8 or args.encode_nproc != 4:
        raise ValueError("this profile is validated for 8 DiT/VAE ranks and 4 encode ranks")
    if args.skip_encode or args.prompt_embeds or args.visual_cond:
        if bool(args.prompt_embeds) != bool(args.visual_cond):
            raise ValueError("--prompt-embeds and --visual-cond must be given together")
        for path in (args.prompt_embeds, args.visual_cond):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        if not args.prompt_embeds:
            for name in ("prompt_embeds.pt", "visual_cond.pt"):
                if not (args.output_dir / name).is_file():
                    raise FileNotFoundError(args.output_dir / name)
    else:
        if not args.image or not args.image.is_file():
            raise ValueError("pass an existing --image for first-frame encoding")
        if not (args.prompt or cfg["sample"].get("prompt")):
            raise ValueError("pass --prompt for first-frame encoding")
    if any((args.output_dir / name).exists() for name in ("latents.pt", "i2va.mp4", "t2va.mp4")):
        raise FileExistsError(f"output already contains a sample: {args.output_dir}")


def stage_commands(args: argparse.Namespace, cfg_path: Path, cfg: dict) -> list[list[str]]:
    def stage(name: str, nproc: int, port: int) -> list[str]:
        return [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}", "--master_addr=127.0.0.1",
            "--nnodes=1", "--node_rank=0", f"--master_port={port}",
            "--module", "turbodiffusion.h3.stages", name, "-i", str(cfg_path),
        ]

    commands = []
    if not (args.skip_encode or args.prompt_embeds):
        commands.append(stage("encode", args.encode_nproc, args.master_port - 1))
    commands.append(stage("sample", args.nproc, args.master_port))
    commands.append(stage("decode", args.nproc, args.master_port + 1))
    return commands


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    cfg = effective_config(args)
    validate_inputs(args, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.output_dir / "run.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if args.prompt_embeds:
        for src, name in (
            (args.prompt_embeds, "prompt_embeds.pt"),
            (args.visual_cond, "visual_cond.pt"),
        ):
            dst = args.output_dir / name
            if src.resolve() != dst:
                shutil.copy2(src, dst)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT), env.get("PYTHONPATH", "")
    )))
    env.setdefault("OMP_NUM_THREADS", "8")
    for command in stage_commands(args, cfg_path, cfg):
        print("[h3]", shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not args.dry_run:
        output = args.output_dir / "i2va.mp4"
        if not output.is_file():
            raise RuntimeError(f"H3 decode did not produce {output}")
        print(f"[h3] ready -> {output}", flush=True)


if __name__ == "__main__":
    main()

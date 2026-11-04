"""Offline AdaLN modulation tables.

``MiniMaxH3AdalnProj`` depends only on the timestep, so a fixed sigma schedule
lets the whole table be precomputed once and the ~24GiB of AdaLN weights dropped
from the shipped checkpoint. The table is exact — the same Linear evaluated at
the same timesteps in the same dtype — so baking is lossless.

A table is bound to ``(task, steps, video_shift, audio_shift)``. Serving a
different schedule needs a different table, which is why the tables ship
alongside the weights rather than inside them.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from turbodiffusion.h3.dit.pack import H3_IMGVID_COND_TIMESTEP

TASKS = ("i2va",)
BAKE_FORMAT = "h3_adaln_bake_v1"
# MiniMaxH3AdalnProj expand_ratio per site (blocks use 6 modulation terms, final 2)
BLOCK_EXPAND = 6
FINAL_EXPAND = 2


def adaln_candidates(
    task: str,
    *,
    t_video: float,
    t_audio: float,
    device: torch.device | str = "cpu",
    imgvid_cond_timestep: float = H3_IMGVID_COND_TIMESTEP,
) -> torch.Tensor:
    """Candidate AdaLN timesteps for one denoise step.

    Single source of truth for both sampling and offline baking; the unique set
    of these values is exactly what the AdaLN table must cover.
    """
    if task in ("i2va", "fl2va"):
        vals = [t_video, t_audio, max(t_video, float(imgvid_cond_timestep))]
    else:
        raise ValueError(f"unknown task {task!r}")
    return torch.tensor(vals, device=device, dtype=torch.float32)


def schedule_timesteps(
    task: str,
    *,
    steps: int,
    video_shift: float,
    audio_shift: float,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Reproduce ``sample_*``'s per-step unique AdaLN timesteps, offline."""
    from turbodiffusion.h3.dit.processor import _shifted_sigma_schedule

    dev = torch.device(device)
    video_sigmas = _shifted_sigma_schedule(steps, video_shift, device=dev)
    audio_sigmas = _shifted_sigma_schedule(steps, audio_shift, device=dev)
    if video_sigmas.numel() != audio_sigmas.numel():
        raise RuntimeError("video/audio sigma schedules disagree in length")
    n_steps = int(video_sigmas.numel()) - 1
    out = []
    for i in range(n_steps):
        cand = adaln_candidates(
            task,
            t_video=1.0 - float(video_sigmas[i]),
            t_audio=1.0 - float(audio_sigmas[i]),
            device=dev,
        )
        out.append(torch.unique(cand, sorted=True))
    return out


def bake_key(timesteps_per_step: list[torch.Tensor]) -> tuple:
    """Schedule identity used to check a table against the running sample."""
    return tuple(round(float(v), 8) for ts in timesteps_per_step for v in ts.tolist())


def save_adaln_bake(
    path: Path | str,
    block_mods: list[list[tuple[torch.Tensor, ...]]],
    final_mods: list[tuple[torch.Tensor, ...]],
    *,
    task: str,
    steps: int,
    video_shift: float,
    audio_shift: float,
    timesteps_per_step: list[torch.Tensor],
    timestep_mlp_dtype: torch.dtype | None = None,
) -> Path:
    """Store one stacked tensor per step; chunks are restored on load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for i, per_block in enumerate(block_mods):
        # each entry is the tuple returned by MiniMaxH3AdalnProj → re-concatenate
        stacked = torch.stack([torch.cat(mods, dim=-1) for mods in per_block], dim=0)
        tensors[f"block.{i}"] = stacked.cpu().contiguous()
    for i, mods in enumerate(final_mods):
        tensors[f"final.{i}"] = torch.cat(mods, dim=-1).cpu().contiguous()

    meta = {
        "format": BAKE_FORMAT,
        "task": task,
        "steps": steps,
        "video_shift": video_shift,
        "audio_shift": audio_shift,
        "n_steps": len(block_mods),
        "n_blocks": len(block_mods[0]) if block_mods else 0,
        "block_expand": BLOCK_EXPAND,
        "final_expand": FINAL_EXPAND,
        "timesteps": [[float(v) for v in ts.tolist()] for ts in timesteps_per_step],
    }
    if timestep_mlp_dtype is not None:
        meta["timestep_mlp_dtype"] = str(timestep_mlp_dtype).removeprefix("torch.")
    save_file(tensors, str(path), metadata={"h3_adaln_bake": json.dumps(meta)})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_adaln_bake(
    path: Path | str,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> tuple[tuple, list[list[tuple[torch.Tensor, ...]]], list[tuple[torch.Tensor, ...]], dict]:
    """Return ``(key, block_mods, final_mods, meta)`` ready to feed the blocks."""
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(f"expected an adaln bake file, got directory {path}")
    meta_path = path.with_suffix(".json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing sidecar metadata {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != BAKE_FORMAT:
        raise ValueError(f"unexpected adaln bake format {meta.get('format')!r}")

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        embedded = (handle.metadata() or {}).get("h3_adaln_bake")
    if embedded is None or json.loads(embedded) != meta:
        raise ValueError(
            "AdaLN embedded metadata does not match its JSON sidecar; "
            "download or regenerate both files together from the same revision"
        )

    raw = load_file(str(path), device=str(device))
    n_steps = int(meta["n_steps"])
    block_expand = int(meta.get("block_expand", BLOCK_EXPAND))
    final_expand = int(meta.get("final_expand", FINAL_EXPAND))

    block_mods: list[list[tuple[torch.Tensor, ...]]] = []
    final_mods: list[tuple[torch.Tensor, ...]] = []
    for i in range(n_steps):
        stacked = raw[f"block.{i}"]
        if dtype is not None:
            stacked = stacked.to(dtype)
        block_mods.append([tuple(stacked[b].chunk(block_expand, dim=-1)) for b in range(stacked.shape[0])])
        fin = raw[f"final.{i}"]
        if dtype is not None:
            fin = fin.to(dtype)
        final_mods.append(tuple(fin.chunk(final_expand, dim=-1)))

    key = tuple(round(float(v), 8) for ts in meta["timesteps"] for v in ts)
    return key, block_mods, final_mods, meta

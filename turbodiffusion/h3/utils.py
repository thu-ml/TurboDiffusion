from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    """Resolve local component paths without relying on a separate checkout."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"expected a configuration mapping in {path}")
    root = Path(cfg["model_root"]).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    cfg["model_root"] = str(root.resolve())
    for section, key, subdir in (
        ("text_encoder", "model_path", "text_encoder"),
        ("dit", "model_path", "transformer"),
        ("vae", "video_vae_path", "video_vae"),
        ("vae", "audio_vae_path", "audio_vae"),
    ):
        node = cfg.setdefault(section, {})
        value = Path(node.get(key) or root / subdir).expanduser()
        node[key] = str((value if value.is_absolute() else path.parent / value).resolve())
    return cfg


def collect_gpu_info(rank: int = 0) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(rank % torch.cuda.device_count())


def barrier() -> None:
    if torch.distributed.is_initialized():
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()

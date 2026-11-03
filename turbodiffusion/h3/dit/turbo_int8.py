"""TurboDiffusion blockwise W8A8 Linear wrappers for the H3 main DiT blocks."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from turbodiffusion.h3.dit.model import MiniMaxH3DiTModel
from turbodiffusion.h3.dit.weight_loader import FP32_PROJECTIONS

FORMAT = "h3_fl2va_turbo_w8a8_128"


def is_mlp_alias_key(key: str) -> bool:
    parts = key.split(".")
    return (
        len(parts) == 5
        and parts[0] == "blocks"
        and parts[1].isdigit()
        and parts[2] == "mlp"
        and parts[3] in ("w1", "w2")
        and parts[4] in ("int8_weight", "scale")
    )


def apply_turbo_int8(model: MiniMaxH3DiTModel, *, quantize: bool) -> int:
    """Wrap main-block MLP and attention projections with the repository's Int8Linear.

    Export quantizes each BF16 Linear on the GPU before returning the INT8
    buffers to CPU. Inference constructs empty wrappers on meta before assign.
    """
    try:
        from turbodiffusion.ops.core import Int8Linear
    except ImportError as exc:
        raise ImportError("install TurboDiffusion with its CUDA ops to use turbo W8A8") from exc

    count = 0
    for block in model.blocks:
        for parent, name, alias in (
            (block.attn, "qkv_proj", None),
            (block.attn, "out_proj", None),
            (block.mlp, "fc1", "w1"),
            (block.mlp, "fc2", "w2"),
        ):
            original = getattr(parent, name)
            if not isinstance(original, torch.nn.Linear):
                raise TypeError(f"{name} is not an unwrapped Linear: {type(original)}")
            replacement = Int8Linear.from_linear(original, quantize=quantize)
            setattr(parent, name, replacement)
            if alias is not None:
                setattr(parent, alias, replacement)
            count += 1
    return count


def load_fl2va_turbo_int8(
    model: MiniMaxH3DiTModel, path: Path | str, *, rank: int = 0
) -> None:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if (f.metadata() or {}).get("format") != FORMAT:
            raise ValueError(f"expected {FORMAT} checkpoint: {path}")
    state = load_file(str(path), device="cpu")
    required_fp32 = [
        name for name in model.state_dict()
        if name == "rope.inv_freq" or name.startswith(FP32_PROJECTIONS)
    ]
    if any(name in state and state[name].dtype != torch.float32 for name in required_fp32):
        raise ValueError(
            "W8A8 checkpoint lost H3 FP32 tensors; re-export from the full model "
            "with turbodiffusion.inference.export_h3_w8a8"
        )
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in incompatible.missing_keys if not is_mlp_alias_key(k)]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"turbo W8A8 checkpoint mismatch: missing={missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    if rank == 0:
        print(f"[dit] loaded turbo W8A8 from {path} ({len(state)} tensors)", flush=True)

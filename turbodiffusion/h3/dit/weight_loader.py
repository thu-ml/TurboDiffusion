from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from turbodiffusion.h3.dit.model import MiniMaxH3DiTModel, _reorder_grouped_qkv_to_qkv


FP32_PROJECTIONS = (
    "video_patch_proj.", "audio_patch_proj.", "time_embedder.",
    "final_layer.video_out.", "final_layer.audio_out.",
)


def materialize_cpu_model(model: MiniMaxH3DiTModel, dtype=torch.bfloat16) -> None:
    """Allocate the BF16 block stack and original FP32 projections on CPU."""
    model.to_empty(device="cpu")
    for name, parameter in model.named_parameters():
        target_dtype = torch.float32 if name.startswith(FP32_PROJECTIONS) else dtype
        parameter.data = torch.empty(parameter.shape, dtype=target_dtype)
    # Buffers retain their declared dtype, particularly FP32 rope.inv_freq.


def _diffusers_key(key: str) -> tuple[str, str | None]:
    """Return runtime parameter and optional Q/K/V part for a Diffusers shard."""
    for source, target in (
        ("audio_proj_in.", "audio_patch_proj."),
        ("audio_proj_out.", "final_layer.audio_out."),
        ("context_embedder.", "condition_proj."),
        ("proj_in.", "video_patch_proj."),
        ("proj_out.", "final_layer.video_out."),
        ("norm_out.linear.", "final_layer.adaln_proj.linear."),
        ("norm_out.norm.", "final_layer.norm."),
        ("time_embedder.linear_1.", "time_embedder.proj_in."),
        ("time_embedder.linear_2.", "time_embedder.proj_out."),
        ("transformer_blocks.", "blocks."),
        ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ):
        if key.startswith(source):
            key = target + key[len(source):]
            break
    key = key.replace(".attn.norm_q.", ".attn.q_norm.")
    key = key.replace(".attn.norm_k.", ".attn.k_norm.")
    key = key.replace(".attn.to_out.0.", ".attn.out_proj.")
    key = key.replace(".ff.net.0.proj.", ".mlp.fc1.")
    key = key.replace(".ff.net.2.", ".mlp.fc2.")
    match = re.match(r"^(.*\.attn)\.to_([qkv])\.(weight)$", key)
    if match:
        return f"{match[1]}.qkv_proj.{match[3]}", match[2]
    return key, None


def load_fl2va_streaming(model: MiniMaxH3DiTModel, model_dir: Path, *, rank: int = 0) -> tuple[int, int]:
    """Assign FL2VA weights one tensor at a time (no full state_dict in RAM).

    Avoids the ~2× peak of ``load_file`` + ``load_state_dict`` which OOMs
    when 8 ranks each hold a 66GB model.
    """
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    diffusers = not index.is_file()
    if diffusers:
        index = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(index)
    with open(index, encoding="utf-8") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    # Invert: shard -> keys
    shards: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        shards.setdefault(shard, []).append(k)

    named = dict(model.named_parameters())
    named_buf = dict(model.named_buffers())
    missing = set(named) | {k for k in named_buf if k == "rope.inv_freq"}
    qkv_parts: dict[str, dict[str, torch.Tensor]] = {}
    # buffers we care about
    loaded = 0
    unexpected = 0

    for shard_name in sorted(shards):
        path = model_dir / shard_name
        if rank == 0:
            print(f"[dit] load shard {shard_name} ...", flush=True)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in shards[shard_name]:
                tensor = f.get_tensor(key)
                target, part = _diffusers_key(key) if diffusers else (key, None)
                if part is not None:
                    pieces = qkv_parts.setdefault(target, {})
                    pieces[part] = tensor
                    if len(pieces) != 3:
                        continue
                    tensor = torch.cat([pieces[name] for name in "qkv"], dim=0)
                    del qkv_parts[target]
                elif diffusers and key.endswith(".ff.net.0.proj.weight"):
                    # Diffusers SwiGLU stores [up, gate]; MiniMaxH3MLP uses [gate, up].
                    up, gate = tensor.chunk(2, dim=0)
                    tensor = torch.cat([gate, up], dim=0)
                elif not diffusers and key.endswith("attn.qkv_proj.weight"):
                    tensor = _reorder_grouped_qkv_to_qkv(
                        tensor,
                        num_heads=model.cfg.num_attention_heads,
                        head_dim=model.cfg.attention_head_dim,
                    )
                if target in named:
                    param = named[target]
                    if param.shape != tensor.shape:
                        raise RuntimeError(f"shape mismatch {target}: {tuple(param.shape)} vs {tuple(tensor.shape)}")
                    if target.startswith(FP32_PROJECTIONS):
                        if tensor.dtype != torch.float32:
                            raise ValueError(f"source {target} must retain its original FP32 precision")
                        param.data = tensor.to(device=param.device)
                    else:
                        param.data.copy_(tensor.to(dtype=param.dtype))
                    missing.discard(target)
                    loaded += 1
                elif target in named_buf:
                    buf = named_buf[target]
                    if target == "rope.inv_freq" and tensor.dtype != torch.float32:
                        raise ValueError("source rope.inv_freq must be FP32; use the original checkpoint")
                    buf.data.copy_(tensor.to(dtype=buf.dtype))
                    missing.discard(target)
                    loaded += 1
                else:
                    unexpected += 1
                del tensor
        # encourage reclaim between shards
        if rank == 0:
            print(f"[dit]   done {shard_name} (loaded so far {loaded})", flush=True)

    if qkv_parts:
        raise RuntimeError(f"incomplete QKV projections: {sorted(qkv_parts)}")
    if diffusers:
        # Diffusers computes RoPE frequencies rather than serializing this buffer.
        rope = named_buf["rope.inv_freq"]
        values = 10000.0 ** (-torch.arange(rope.numel(), dtype=torch.float32) / rope.numel())
        rope.copy_(values)
        missing.discard("rope.inv_freq")
        if missing or unexpected:
            raise RuntimeError(f"Diffusers weights: {len(missing)} missing, {unexpected} unexpected; missing={sorted(missing)[:12]}")
    return loaded, len(missing)

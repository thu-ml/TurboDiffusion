"""Attention adapters for the H3 token layout."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _add_sink_blocks(mask: torch.Tensor, pre: int, post: int, block_q: int, block_k: int) -> None:
    """Preserve dense prefix/suffix query rows and key columns, rounded to blocks."""
    if pre:
        mask[..., : (pre + block_q - 1) // block_q, :] = 1
        mask[..., : (pre + block_k - 1) // block_k] = 1
    if post:
        mask[..., -((post + block_q - 1) // block_q) :, :] = 1
        mask[..., -((post + block_k - 1) // block_k) :] = 1


def attention_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str = "sdpa",
    scale: float | None = None,
    sparge_attn_topk: float = 0.5,
    sparge_attn_pre_sink_length: int = 0,
    sparge_attn_post_sink_length: int = 0,
) -> torch.Tensor:
    """Accept [batch, sequence, heads, dim] and preserve that layout."""
    if q.ndim != 4:
        raise ValueError(f"expected [B,S,H,D], got {tuple(q.shape)}")
    q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    if backend == "sparge_attn":
        topk = float(sparge_attn_topk)
        pre, post = int(sparge_attn_pre_sink_length), int(sparge_attn_post_sink_length)
        if not 0 < topk <= 1:
            raise ValueError("sparge_attn_topk must be in (0, 1]")
        if pre < 0 or post < 0:
            raise ValueError("SpargeAttn sink lengths must be nonnegative")
        try:
            from spas_sage_attn import block_sparse_sage2_attn_cuda
        except ImportError as exc:
            raise ImportError(
                "Install public thu-ml/SpargeAttn (spas_sage_attn); see docs/h3_i2va.md"
            ) from exc
        # The public kernel requires at least 128 query tokens.
        if q.shape[-2] < 128:
            out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=False)
        else:
            from turbodiffusion.h3.pooling import get_block_map

            sm90 = torch.cuda.get_device_capability(q.device) == (9, 0)
            block_q, block_k = (64, 128) if sm90 else (128, 64)
            key_blocks = (k.shape[-2] + block_k - 1) // block_k
            # Rounding 1/K back to an integer count must not produce zero keys.
            min_topk = math.nextafter(1 / key_blocks, 1.0)
            mask, _, _ = get_block_map(
                q, k, max(topk, min_topk), BLKQ=block_q, BLKK=block_k
            )
            _add_sink_blocks(mask, pre, post, block_q, block_k)
            # Disable additional PV-based skipping so the explicit mask governs sparsity.
            out = block_sparse_sage2_attn_cuda(
                q, k, v, mask_id=mask, scale=scale, smooth_k=True,
                pvthreshd=1e6, attention_sink=False, tensor_layout="HND",
            )
    elif backend == "sdpa":
        out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=False)
    else:
        raise ValueError(f"unsupported H3 attention backend: {backend}")
    return out.transpose(1, 2).contiguous()

"""H3-scoped SLA pooling with defined values for masked tail loads."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compress(X, XM, L: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    rows = block * BLOCK_L + tl.arange(0, BLOCK_L)
    cols = tl.arange(0, D)
    values = tl.load(
        X + batch_head * L * D + rows[:, None] * D + cols[None, :],
        mask=rows[:, None] < L, other=0,
    )
    count = tl.minimum(BLOCK_L, L - block * BLOCK_L)
    mean = tl.sum(values, axis=0, dtype=tl.float32) / count
    output = XM + (batch_head * tl.cdiv(L, BLOCK_L) + block) * D + cols
    tl.store(output, mean.to(XM.dtype.element_ty))


def mean_pool(x, block_size):
    batch, heads, length, dim = x.shape
    blocks = triton.cdiv(length, block_size)
    result = torch.empty((batch, heads, blocks, dim), device=x.device, dtype=x.dtype)
    _compress[(blocks, batch * heads)](x.contiguous(), result, length, dim, block_size)
    return result


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    smooth_k = k - k.mean(dim=-2, keepdim=True)
    scores = mean_pool(q, BLKQ) @ mean_pool(smooth_k, BLKK).transpose(-1, -2)
    count = min(scores.shape[-1], max(1, int(topk_ratio * scores.shape[-1])))
    indices = torch.topk(scores, count, dim=-1, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.int8)
    mask.scatter_(-1, indices, 1)
    return mask, indices, count

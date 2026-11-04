# SPDX-License-Identifier: Apache-2.0
"""Process group for the vendored Qwen3-VL encoder.

``encoder.py`` reaches for exactly three things on the group it is handed —
``world_size``, ``rank_in_group`` and ``all_reduce`` — so vLLM's
``GroupCoordinator`` can be replaced by a plain ``torch.distributed`` wrapper and
the encoder runs under a bare ``torchrun`` without pulling in vLLM.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


class EncoderGroup:
    """Tensor-parallel group spanning the first ``world_size`` ranks."""

    def __init__(self, world_size: int, rank: int, device_group=None) -> None:
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = device_group

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            dist.all_reduce(tensor, group=self.device_group)
        return tensor


def build(tp_size: int) -> EncoderGroup:
    """Initialize NCCL if needed and return the group for this rank.

    ``tp_size == 1`` short-circuits to a group that never collectives, so the
    encoder can also be driven from an ordinary single-process script.
    """
    if tp_size == 1:
        return EncoderGroup(world_size=1, rank=0)
    if not dist.is_initialized():
        # device_id or NCCL bootstraps on cuda:0 for every rank, costing a
        # ~500MB context there on top of each rank's own card.
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local))
    rank = dist.get_rank()
    if dist.get_world_size() != tp_size:
        raise ValueError(
            f"launch with --nproc_per_node={tp_size} to match text_encoder.tp_size, "
            f"got world_size={dist.get_world_size()}"
        )
    return EncoderGroup(world_size=tp_size, rank=rank, device_group=dist.group.WORLD)

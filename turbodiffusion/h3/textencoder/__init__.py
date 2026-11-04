# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel Qwen3-VL layer-50 encoder vendored from vllm-omni."""

from .encoder import (
    MINIMAX_H3_QWEN3VL_HIDDEN_DIM,
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
    MiniMaxH3Qwen3VLEncoder,
)
from .group import EncoderGroup, build

__all__ = [
    "MINIMAX_H3_QWEN3VL_HIDDEN_DIM",
    "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER",
    "EncoderGroup",
    "MiniMaxH3Qwen3VLEncoder",
    "build",
]

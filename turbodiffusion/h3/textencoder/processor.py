"""Layer-50 Qwen3-VL text conditioning for FL2VA."""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

from .encoder import MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER, MiniMaxH3Qwen3VLEncoder
from .group import build


@torch.no_grad()
def encode_prompt(cfg: dict, prompt: str) -> torch.Tensor:
    from transformers import AutoProcessor

    te = cfg["text_encoder"]
    if int(te.get("layer_idx", 50)) != MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER:
        raise ValueError("H3 requires Qwen3-VL layer-50 embeddings")
    path = Path(te["model_path"])
    group = build(int(te.get("tp_size", 4)))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    # Official FL2VA separates processor/tokenizer files from the encoder weights.
    processor_path = path if (path / "processor_config.json").exists() else path.parent / "processor"
    processor = AutoProcessor.from_pretrained(str(processor_path))
    model = MiniMaxH3Qwen3VLEncoder(
        str(path), device=device, load_model=True, encoder_group=group
    )
    model.load_to_device()
    model.eval()
    if te.get("apply_chat_template", False):
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=False,
        )
    inputs = processor(
        text=[prompt], return_tensors="pt", padding=True, truncation=True,
        max_length=int(te.get("max_length", 4096)),
    )
    ids = inputs["input_ids"][0]
    if "attention_mask" in inputs:
        ids = ids[:int(inputs["attention_mask"][0].sum())]
    embeds = model.encode_ids(ids).detach().to(dtype=torch.bfloat16).contiguous().cpu()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return embeds

"""Compare eight-rank H3 sampling with a local reference for a long prompt.

Run with torchrun --standalone --nproc_per_node=8 tests/h3_sequence_parallel_check.py.
"""
from unittest.mock import patch

import torch
import torch.distributed as dist

from turbodiffusion.h3 import parallel
from turbodiffusion.h3.dit.config import H3DiTConfig
from turbodiffusion.h3.dit.model import MiniMaxH3DiTModel
from turbodiffusion.h3.dit import processor as processor_module
from turbodiffusion.h3.dit.processor import H3DiTProcessor, _gather_token_preds
from turbodiffusion.h3.dit.weight_loader import FP32_PROJECTIONS


def check_gather(state):
    device = torch.device(f"cuda:{state.local_rank}")
    positions = torch.arange(64, device=device)
    masks = {
        "all_ranks": positions >= 0,
        "rank_zero_empty": positions >= 8,
        "several_empty": (positions // 8) % 3 != 0,
        "only_last_rank": positions >= 56,
        "all_empty": positions < 0,
    }
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        values = torch.arange(256, device=device).reshape(64, 4).to(dtype) / 8
        for name, mask in masks.items():
            local_mask = parallel.shard_sequence(mask)
            local = parallel.shard_sequence(values)[local_mask]
            actual = _gather_token_preds(local, mask, feat=4)
            if actual.dtype != dtype:
                raise AssertionError(f"{name}: prediction dtype changed")
            torch.testing.assert_close(actual, values[mask], rtol=0, atol=0)
    if state.rank == 0:
        print("PASS: SP8 prediction gather, three dtypes, five empty/nonempty layouts", flush=True)


@torch.inference_mode()
def main():
    state = parallel.init_parallel(8, use_quant_comm=False, use_async_comm=True, num_heads=8)
    if state.world_size != 8:
        raise ValueError("requires exactly eight GPU ranks")
    try:
        check_gather(state)
        torch.manual_seed(17)
        arch = H3DiTConfig(
            num_layers=2, token_refiner_num_layers=1, hidden_size=512,
            ffn_hidden_size=1024, num_attention_heads=8, attention_head_dim=64,
            latents_dim=4, audio_latents_dim=4, text_dim=16,
            timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=16,
            adaln_out_features=18 * 512, final_adaln_out_features=2 * 512,
            rope_inv_freq_len=4,
        )
        prompt = torch.randn(513, 16)
        condition = torch.randn(1, arch.video_patch_dim)
        for backend in ("sdpa", "sparge_attn"):
            model = MiniMaxH3DiTModel(arch, attn_backend=backend)
            model.rope.inv_freq.copy_(10000.0 ** (-torch.arange(4).float() / 4))
            model = model.to(f"cuda:{state.local_rank}").eval()
            for name, parameter in model.named_parameters():
                if not name.startswith(FP32_PROJECTIONS):
                    parameter.data = parameter.data.bfloat16()
            processor = H3DiTProcessor({"dit": {"model_path": ".", "attn_backend": backend}})
            processor.model = model
            processor.device = torch.device(f"cuda:{state.local_rank}")
            kwargs = dict(duration_seconds=0.1, steps=5, seed=31, canvas_latent_hw=(2, 2))
            with patch.object(parallel, "_STATE", parallel.ParallelState()):
                expected = processor.sample_i2va(prompt, condition, **kwargs)
            with patch.object(processor_module, "_gather_token_preds", wraps=_gather_token_preds) as gather:
                actual = processor.sample_i2va(prompt, condition, **kwargs)
            if state.rank == 0:
                for call in gather.call_args_list:
                    local = call.args[0]
                    if local.numel() != 0 or local.dtype != torch.float32:
                        raise AssertionError("regression requires empty FP32 video/audio shards on rank zero")
            for reference, output in zip(expected, actual):
                if not torch.isfinite(output).all():
                    raise AssertionError(f"nonfinite {backend} output")
                relative_l2 = (reference.float() - output.float()).norm() / reference.float().norm()
                if relative_l2 > 0.02:
                    raise AssertionError(f"{backend} SP8 mismatch: relative L2={relative_l2.item()}")
            if state.rank == 0:
                print(f"PASS: {backend}, mixed precision, empty rank-zero modalities, "
                      "513 text tokens, partial padding, SP8/local relative L2 < 0.02",
                      flush=True)
        dist.barrier(device_ids=[state.local_rank])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

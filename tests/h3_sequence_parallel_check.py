"""Compare eight-rank H3 sampling with a local reference for a long prompt.

Run with torchrun --standalone --nproc_per_node=8 tests/h3_sequence_parallel_check.py.
"""
from unittest.mock import patch

import torch
import torch.distributed as dist

from turbodiffusion.h3 import parallel
from turbodiffusion.h3.dit.config import H3DiTConfig
from turbodiffusion.h3.dit.model import MiniMaxH3DiTModel
from turbodiffusion.h3.dit.processor import H3DiTProcessor


@torch.inference_mode()
def main():
    state = parallel.init_parallel(8, use_quant_comm=False, use_async_comm=True, num_heads=8)
    if state.world_size != 8:
        raise ValueError("requires exactly eight GPU ranks")
    try:
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
            model = model.to(f"cuda:{state.local_rank}").bfloat16().eval()
            processor = H3DiTProcessor({"dit": {"model_path": ".", "attn_backend": backend}})
            processor.model = model
            processor.device = torch.device(f"cuda:{state.local_rank}")
            kwargs = dict(duration_seconds=0.1, steps=5, seed=31, canvas_latent_hw=(2, 2))
            with patch.object(parallel, "_STATE", parallel.ParallelState()):
                expected = processor.sample_i2va(prompt, condition, **kwargs)
            actual = processor.sample_i2va(prompt, condition, **kwargs)
            for reference, output in zip(expected, actual):
                if not torch.isfinite(output).all():
                    raise AssertionError(f"nonfinite {backend} output")
                relative_l2 = (reference.float() - output.float()).norm() / reference.float().norm()
                if relative_l2 > 0.02:
                    raise AssertionError(f"{backend} SP8 mismatch: relative L2={relative_l2.item()}")
            if state.rank == 0:
                print(f"PASS: {backend}, 513 text tokens, partial padding, SP8/local relative L2 < 0.02",
                      flush=True)
        dist.barrier(device_ids=[state.local_rank])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

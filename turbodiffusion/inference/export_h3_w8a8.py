"""Export full H3 DMD weights using the existing TurboDiffusion W8A8 operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaln-output", type=Path,
                        help="optionally regenerate the four-update AdaLN table with FP32 time embeddings")
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file

    from turbodiffusion.h3.dit.config import H3DiTConfig
    from turbodiffusion.h3.dit.model import MiniMaxH3DiTModel
    from turbodiffusion.h3.dit.turbo_int8 import FORMAT, apply_turbo_int8, is_mlp_alias_key
    from turbodiffusion.h3.dit.weight_loader import load_fl2va_streaming, materialize_cpu_model

    arch = H3DiTConfig.from_json(str(args.model_path / "config.json"))
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(arch, attn_backend="sdpa")
    materialize_cpu_model(model)
    loaded, missing = load_fl2va_streaming(model, args.model_path)
    if missing:
        raise RuntimeError(f"incomplete source checkpoint: {missing} tensors missing")
    if args.adaln_output:
        from turbodiffusion.h3.dit.adaln_bake import save_adaln_bake, schedule_timesteps

        schedule = schedule_timesteps("i2va", steps=5, video_shift=12, audio_shift=3)
        model.time_embedder.cuda()
        blocks, final = model.bake_and_drop_adaln(schedule, device=torch.device("cuda"))
        save_adaln_bake(
            args.adaln_output, blocks, final, task="i2va", steps=5,
            video_shift=12, audio_shift=3, timesteps_per_step=schedule,
            timestep_mlp_dtype=model.time_embedder.proj_in.weight.dtype,
        )
        model.time_embedder.cpu()
        del blocks, final
    projections = apply_turbo_int8(model, quantize=True)
    if projections != 4 * arch.num_layers:
        raise RuntimeError(f"unexpected number of quantized projections: {projections}")
    for block in model.blocks:
        block.adaln_proj = None
    model.final_layer.adaln_proj = None
    state = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in model.state_dict().items() if not is_mlp_alias_key(key)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.output), metadata={"format": FORMAT})
    args.output.with_suffix(".json").write_text(json.dumps({
        "format": FORMAT, "projections": projections, "adaln": "separate",
    }, indent=2))
    print(f"Exported {loaded} source tensors / {projections} W8A8 projections to {args.output}")


if __name__ == "__main__":
    main()

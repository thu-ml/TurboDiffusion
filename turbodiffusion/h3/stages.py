"""Distributed encode, DMD sample and decode stages, all implemented in this package."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .utils import load_yaml


def encode(cfg):
    from PIL import Image, ImageOps

    from .resolution import aspect_string, delivery_size, model_canvas, parse_aspect
    from .textencoder.processor import encode_prompt
    from .vae import H3VAEProcessor

    sample = cfg["sample"]
    output = Path(cfg["output_dir"])
    embeds = encode_prompt(cfg, sample["prompt"])
    if dist.get_rank() == 0:
        torch.save({"prompt_embeds": embeds}, output / "prompt_embeds.pt")
    with Image.open(sample["image"]) as image:
        source_wh = ImageOps.exif_transpose(image).size
    aspect = sample.get("encode_aspect_ratio") or aspect_string(*source_wh)
    ratio = parse_aspect(aspect)
    short = int(sample.get("short_edge", 768))
    resolution = sample.get("resolution", "768p")
    height, width = model_canvas(*ratio, resolution=resolution, short_edge=short)
    user_hw = delivery_size(*ratio, resolution=resolution, short_edge=short)
    vae = H3VAEProcessor(cfg)
    latent = vae.encode_image(sample["image"], height, width)
    _, channels, time, lh, lw = latent.shape
    rows = latent.reshape(1, channels, time, lh // 2, 2, lw // 2, 2)
    rows = rows.permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, channels * 4).contiguous()
    if dist.get_rank() == 0:
        torch.save({
            "visual_cond_rows": rows, "keyframe_anchors": [0],
            "latent_hw": [lh, lw], "user_hw": list(user_hw),
        }, output / "visual_cond.pt")


def sample(cfg):
    from .dit.processor import H3DiTProcessor

    output = Path(cfg["output_dir"])
    prompt = torch.load(output / "prompt_embeds.pt", map_location="cpu", weights_only=True)
    visual = torch.load(output / "visual_cond.pt", map_location="cpu", weights_only=True)
    processor = H3DiTProcessor(cfg)
    processor.setup()
    params, dit = cfg["sample"], cfg["dit"]
    video, audio = processor.sample_i2va(
        prompt["prompt_embeds"], visual["visual_cond_rows"],
        duration_seconds=float(params.get("duration_seconds", 5)),
        short_edge=int(params.get("short_edge", 768)),
        aspect_ratio=params.get("aspect_ratio", "16:9"),
        seed=int(params.get("seed", 0)), steps=int(dit["steps"]),
        video_shift=float(dit["video_shift"]), audio_shift=float(dit["audio_shift"]),
        keyframe_anchors=tuple(visual.get("keyframe_anchors") or (0,)),
        canvas_latent_hw=tuple(visual["latent_hw"]),
    )
    if dist.get_rank() == 0:
        torch.save({"video_latents": video.cpu(), "audio_latents": audio.cpu()}, output / "latents.pt")
    processor.release()


def decode(cfg):
    from .vae import H3VAEProcessor

    output = Path(cfg["output_dir"])
    latent = torch.load(output / "latents.pt", map_location="cpu", weights_only=True)
    visual = torch.load(output / "visual_cond.pt", map_location="cpu", weights_only=True)
    vae = H3VAEProcessor(cfg)
    frames, audio = vae.decode(latent["video_latents"], latent["audio_latents"])
    if dist.get_rank() == 0:
        vae.export(frames, audio, output, user_hw=visual.get("user_hw"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "sample", "decode"))
    parser.add_argument("-i", "--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local))
    try:
        {"encode": encode, "sample": sample, "decode": decode}[args.stage](cfg)
        dist.barrier(device_ids=[local])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

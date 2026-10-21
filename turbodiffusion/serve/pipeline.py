"""Video generation pipeline for TurboDiffusion TUI server."""

import argparse
import math
import os
import sys

# Add inference directory to path for modify_model import
_inference_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference")
if _inference_dir not in sys.path:
    sys.path.insert(0, _inference_dir)

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2 as T

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True


def load_models_t2v(args: argparse.Namespace) -> dict:
    """Load T2V model."""
    log.info(f"Loading DiT model from {args.dit_path}")
    net = create_model(dit_path=args.dit_path, args=args)
    net.cuda().eval()
    torch.cuda.empty_cache()
    log.success("Successfully loaded DiT model.")
    return {"net": net}


def load_models_i2v(args: argparse.Namespace) -> dict:
    """Load I2V models (high noise and low noise)."""
    log.info("Loading DiT models for I2V...")

    log.info(f"Loading high-noise model from {args.high_noise_model_path}")
    high_noise_model = create_model(dit_path=args.high_noise_model_path, args=args)
    high_noise_model.cpu().eval()
    torch.cuda.empty_cache()

    log.info(f"Loading low-noise model from {args.low_noise_model_path}")
    low_noise_model = create_model(dit_path=args.low_noise_model_path, args=args)
    low_noise_model.cpu().eval()
    torch.cuda.empty_cache()

    log.success("Successfully loaded DiT models.")
    return {"high_noise_model": high_noise_model, "low_noise_model": low_noise_model}


def load_models(args: argparse.Namespace) -> dict:
    """Load models based on mode."""
    log.info(f"Loading VAE from {args.vae_path}")
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    log.success("Successfully loaded VAE.")

    if args.mode == "t2v":
        models = load_models_t2v(args)
    else:
        models = load_models_i2v(args)

    models["tokenizer"] = tokenizer
    return models


def generate_t2v(models: dict, args: argparse.Namespace, prompt: str, output_path: str) -> str:
    """Generate video from text prompt (T2V mode)."""
    net = models["net"]
    tokenizer = models["tokenizer"]

    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info(f"Computing embedding for prompt: {prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)
    # Don't clear memory here to avoid re-initializing the model for each prompt
    # TODO: adaptively offload t5_encoder to CPU when memory runs low
    # clear_umt5_memory()

    log.info(f"Generating with prompt: {prompt}")
    condition = {
        "crossattn_emb": repeat(
            text_emb.to(**tensor_kwargs),
            "b l d -> (k b) l d",
            k=args.num_samples
        )
    }

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    # mid_t = [1.3, 1.0, 0.6][: args.num_steps - 1]
    # For better visual quality
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # Sampling steps
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1

    for t_cur, t_next in tqdm(
        list(zip(t_steps[:-1], t_steps[1:])),
        desc="Sampling",
        total=total_steps
    ):
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)

            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )

    samples = x.float()

    log.info("Decoding video...")
    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = [video.float().cpu()]
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image_or_video(
        rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
        output_path,
        fps=16
    )

    return output_path


def generate_i2v(models: dict, args: argparse.Namespace, prompt: str,
                 image_path: str, output_path: str) -> str:
    """Generate video from image and text prompt (I2V mode)."""
    high_noise_model = models["high_noise_model"]
    low_noise_model = models["low_noise_model"]
    tokenizer = models["tokenizer"]

    log.info(f"Computing embedding for prompt: {prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)
    # Don't clear memory here to avoid re-initializing the model for each prompt
    # TODO: adaptively offload t5_encoder to CPU when memory runs low
    # clear_umt5_memory()

    log.info(f"Loading and preprocessing image from: {image_path}")
    input_image = Image.open(image_path).convert("RGB")

    if args.adaptive_resolution:
        log.info("Adaptive resolution mode enabled.")
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        log.info(f"Target area is based on {args.resolution} {args.aspect_ratio} (~{max_resolution_area} pixels).")

        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w
        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)
        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride
        log.info(f"Input image aspect ratio: {image_aspect_ratio:.4f}. Adaptive resolution set to: {w}x{h}")
    else:
        log.info("Fixed resolution mode.")
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        log.info(f"Resolution set to: {w}x{h}")
    F = args.num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    log.info(f"Preprocessing image to {w}x{h}...")
    image_transforms = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = image_transforms(input_image).unsqueeze(0).to(
        device=tensor_kwargs["device"], dtype=torch.float32
    )

    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2),
             torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)],
            dim=2
        )  # -> B, C, T, H, W
        encoded_latents = tokenizer.encode(frames_to_encode)  # -> B, C_lat, T_lat, H_lat, W_lat

        del frames_to_encode
        torch.cuda.empty_cache()

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w,
                      device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)

    log.info(f"Generating with prompt: {prompt}")
    condition = {
        "crossattn_emb": repeat(
            text_emb.to(**tensor_kwargs),
            "b l d -> (k b) l d",
            k=args.num_samples
        ),
        "y_B_C_T_H_W": y
    }

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1
    # Single sync up front while the GPU is still idle; reading the boundary
    # with t_cur.item() inside the loop would serialize the CPU behind the GPU
    # on every sampling step.
    t_cur_values = t_steps[:-1].tolist()
    high_noise_model.cuda()
    net = high_noise_model
    switched = False

    for i, (t_cur, t_next) in enumerate(tqdm(
        list(zip(t_steps[:-1], t_steps[1:])),
        desc="Sampling",
        total=total_steps
    )):
        if t_cur_values[i] < args.boundary and not switched:
            high_noise_model.cpu()
            torch.cuda.empty_cache()
            low_noise_model.cuda()
            net = low_noise_model
            switched = True
            log.info("Switched to low noise model.")

        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)

            if args.ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )

    samples = x.float()
    if switched:
        low_noise_model.cpu()
    else:
        high_noise_model.cpu()
    torch.cuda.empty_cache()

    log.info("Decoding video...")
    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = [video.float().cpu()]
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image_or_video(
        rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
        output_path,
        fps=16
    )

    return output_path

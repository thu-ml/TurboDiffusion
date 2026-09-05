# Modified: 2026-01-30 | Fix OOM via MMAP Loading (Zero-Copy) & Aggressive Cleanup
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import gc
import time
import torch
import ctypes
import os
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np
import logging

from imaginaire.utils.io import save_image_or_video
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

# modify_modelから必要な関数をインポート
from modify_model import tensor_kwargs, select_model, replace_attention, replace_linear_norm

# ロギング設定
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

torch._dynamo.config.suppress_errors = True

# libc for aggressive memory trimming
try:
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

def cleanup_all():
    """強制的にガベージコレクション、VRAM解放、およびOSへのメモリ返却を行う"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if libc:
        try:
            libc.malloc_trim(0)
        except:
            pass

def create_model_gpu(dit_path: str, args: argparse.Namespace) -> torch.nn.Module:
    """
    mmap=True を使用して、CPU RAMを消費せずにモデルをロードする。
    これにより cgroup memory limit (46GB) を回避する。
    """
    log.info(f"Loading DiT (MMAP -> GPU): {dit_path}")
    cleanup_all()
    
    # 1. Init Shell on Meta
    with torch.device("meta"):
        net = select_model(args.model)

    # 2. Patch
    if args.attention_type in ['sla', 'sagesla']:
        net = replace_attention(net, attention_type=args.attention_type, sla_topk=args.sla_topk)
    replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm, quantize=False)

    # 3. Load State Dict with MMAP
    # mmap=Trueにより、ファイル内容をRAMにコピーせず、仮想メモリとしてマッピングする。
    # OSが必要な部分だけをページインし、不要になれば即座に破棄できるため、OOMを防ぐ最強の手段。
    log.info("  Mapping state_dict from disk (mmap)...")
    try:
        # map_location="cpu" + mmap=True が重要
        state_dict = torch.load(dit_path, map_location="cpu", mmap=True)
    except Exception as e:
        log.warning(f"  mmap failed ({e}), falling back to standard load.")
        state_dict = torch.load(dit_path, map_location="cpu")

    # 4. Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_checkpoint_wrapped_module.", "")
        new_state_dict[new_key] = v
    del state_dict # 元の参照を削除
    
    # 5. Load into model
    # assign=True により、モデル内のMetaテンソルをmmapされたCPUテンソルに置き換える
    log.info("  Assigning weights to model...")
    net.load_state_dict(new_state_dict, strict=False, assign=True)
    del new_state_dict
    
    # 6. Move to CUDA
    # ここで初めてVRAMへ転送される。転送済みmmapページはOSが勝手に捨ててくれる。
    log.info("  Moving model to CUDA...")
    net = net.to("cuda")
    
    # 7. Eval
    net.eval()
    return net

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion inference script (Memory Optimized)")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--high_noise_model_path", type=str, required=True)
    parser.add_argument("--low_noise_model_path", type=str, required=True)
    parser.add_argument("--boundary", type=float, default=0.9)
    parser.add_argument("--model", choices=["Wan2.2-A14B"], default="Wan2.2-A14B")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--sigma_max", type=float, default=200)
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--resolution", default="720p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--adaptive_resolution", action="store_true")
    parser.add_argument("--ode", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla")
    parser.add_argument("--sla_topk", type=float, default=0.1)
    parser.add_argument("--quant_linear", action="store_true")
    parser.add_argument("--default_norm", action="store_true")
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.serve:
        log.error("Serve mode is not supported in this memory-optimized script.")
        exit(1)

    if args.prompt is None or args.image_path is None:
        log.error("--prompt and --image_path are required")
        exit(1)

    cleanup_all()

    # 1. Text Encoder (T5)
    log.info(f"Computing embedding for prompt: {args.prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path, 
            prompts=args.prompt
        ).to(**tensor_kwargs)
    
    clear_umt5_memory()
    cleanup_all()
    log.info("Text encoder unloaded.")

    # 2. VAE & Image Preprocessing
    log.info(f"Loading and preprocessing image from: {args.image_path}")
    input_image = Image.open(args.image_path).convert("RGB")
    
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    
    if args.adaptive_resolution:
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w
        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)
        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride
        log.info(f"Adaptive resolution set to: {w}x{h}")
    else:
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        log.info(f"Fixed resolution set to: {w}x{h}")

    F = args.num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    image_transforms = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image_tensor = image_transforms(input_image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)
    
    log.info("Encoding image latents...")
    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)], dim=2
        )
        encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode

    del image_tensor, input_image
    cleanup_all()

    # VAE Offload
    if hasattr(tokenizer, "vae"):
        log.info("Offloading VAE to CPU to save VRAM/RAM...")
        tokenizer.vae.cpu()
    cleanup_all()

    # Latent Setup
    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0
    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)
    
    del msk

    condition = {
        "crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples), 
        "y_B_C_T_H_W": y
    }

    # Noise Schedule
    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)
    init_noise = torch.randn(args.num_samples, *state_shape, dtype=torch.float32, device=tensor_kwargs["device"], generator=generator)
    
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor([math.atan(args.sigma_max), *mid_t, 0], dtype=torch.float64, device=init_noise.device)
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1

    del init_noise

    # 3. Sequential Loading Strategy (High Model)
    model = create_model_gpu(dit_path=args.high_noise_model_path, args=args)
    
    switched = False
    
    with torch.inference_mode():
        for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
            
            # Switch Logic: High -> Low
            if (t_cur.item() < args.boundary) and (not switched):
                log.info("Boundary reached. Switching to Low Noise Model...")
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Release High Model
                del model
                cleanup_all()
                log.info("Waiting for memory release...")
                time.sleep(5)
                
                # Load Low Model
                model = create_model_gpu(dit_path=args.low_noise_model_path, args=args)
                switched = True
            
            v_pred = model(
                x_B_C_T_H_W=x.to(**tensor_kwargs), 
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), 
                **condition
            ).to(torch.float64)
            
            if args.ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape, dtype=torch.float32, device=tensor_kwargs["device"], generator=generator,
                )
            
            del v_pred

    # Lowモデルも削除
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del model
    cleanup_all()
    log.info("Inference done. All DiT models unloaded.")

    # 4. Decode Video
    log.info("Cleaning up before decoding...")
    del condition, y, encoded_latents, text_emb
    cleanup_all()

    if hasattr(tokenizer, "vae"):
        log.info("Moving VAE back to GPU for decoding...")
        tokenizer.vae.to(tensor_kwargs["device"])
    
    log.info("Decoding video...")
    samples = x.float()
    del x
    cleanup_all()
    
    with torch.inference_mode():
        video = tokenizer.decode(samples)
    del samples
    cleanup_all()
    
    to_show = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0
    del video
    cleanup_all()

    # Save
    video_tensor = rearrange(to_show[0], "c t h w -> c t h w")
    save_image_or_video(video_tensor, args.save_path, fps=16)
    
    log.info(f"Saved video to {args.save_path}")
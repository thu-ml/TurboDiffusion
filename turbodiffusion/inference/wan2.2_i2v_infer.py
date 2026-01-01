# Blackwell Bridge
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# -----------------------------------------------------------------------------------------
# TURBODIFFUSION OPTIMIZED INFERENCE SCRIPT (I2V)
#
# Co-developed by: Waverly Edwards & Google Gemini (2025)
#
# Modifications:
#   - Implemented "Tiered Failover System" for robust OOM protection.
#   - Added Intelligent Model Switching (High/Low Noise) with memory optimization.
#   - Integrated Tiled Encoding & Decoding for high-resolution processing.
#   - Added Support for Pre-cached Text Embeddings to skip T5 loading.
#   - Optimized memory management (VAE unload/reload, aggressive GC).
#
# Acknowledgments:
#   - Made possible by the work (cache_t5.py) and creativity of: John D. Pope
#
# Description:
#   cache_t5.py pre-computes text embeddings to allow running inference on GPUs with limited VRAM
#   by removing the need to keep the 11GB T5 encoder loaded in memory.
#
# CREDIT REQUEST:
#   If you utilize, share, or build upon this specific optimized script, please
#   acknowledge Waverly Edwards and Google Gemini in your documentation or credits.
# -----------------------------------------------------------------------------------------

import argparse
import math
import os
import gc 
import time
import sys

# --- 1. Memory Tuning (Must be before torch imports) ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

# Safe import for system memory checks
try:
    import psutil
except ImportError:
    psutil = None

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion inference script for Wan2.2 I2V")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image")
    parser.add_argument("--high_noise_model_path", type=str, required=True, help="Path to high-noise model")
    parser.add_argument("--low_noise_model_path", type=str, required=True, help="Path to low-noise model")
    parser.add_argument("--boundary", type=float, default=0.9, help="Switch boundary")
    parser.add_argument("--model", choices=["Wan2.2-A14B"], default="Wan2.2-A14B")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--sigma_max", type=float, default=200)
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--cached_embedding", type=str, default=None)
    parser.add_argument("--skip_t5", action="store_true")
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
    parser.add_argument("--offload_dit", action="store_true")
    return parser.parse_args()

def print_memory_status(step_name=""):
    """
    Prints a detailed breakdown of GPU memory usage.
    """
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize()
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.memory_reserved() / (1024**3)
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem / (1024**3)
    
    print(f"\n📊 [MEMORY REPORT] {step_name}")
    print(f"   ├── 💾 VRAM In Use:    {allocated_gb:.2f} GB")
    print(f"   ├── 📦 VRAM Reserved:  {reserved_gb:.2f} GB")
    print(f"   ├── 🆓 VRAM Free:      {free_gb:.2f} GB")
    print("-" * 60)

def cleanup_memory(step_info=""):
    """Aggressively clears VRAM."""
    if step_info:
        print(f"🧹 Cleaning memory: {step_info}...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if step_info:
        print_memory_status(f"After Cleanup ({step_info})")

def get_tensor_size_mb(tensor):
    return tensor.element_size() * tensor.nelement() / (1024 * 1024)

def force_cpu_float32(target_obj):
    """
    Recursively forces a model or wrapper object to CPU Float32.
    """
    def recursive_cast_to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().float()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(recursive_cast_to_cpu(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: recursive_cast_to_cpu(v) for k, v in obj.items()}
        return obj

    targets = [target_obj]
    if hasattr(target_obj, "model"):
        targets.append(target_obj.model)

    for obj in targets:
        if isinstance(obj, torch.nn.Module):
            try: obj.cpu().float()
            except: pass
        
        for attr_name in dir(obj):
            if attr_name.startswith("__"): continue
            try:
                val = getattr(obj, attr_name)
                if isinstance(val, torch.nn.Module):
                    val.cpu().float()
                elif isinstance(val, (torch.Tensor, list, tuple)):
                    setattr(obj, attr_name, recursive_cast_to_cpu(val))
            except Exception: pass

        if isinstance(obj, torch.nn.Module):
            try:
                for module in obj.modules():
                    for param in module.parameters(recurse=False):
                        if param is not None:
                            param.data = param.data.cpu().float()
                            if param.grad is not None: param.grad = None
                    for buf in module.buffers(recurse=False):
                        if buf is not None:
                            buf.data = buf.data.cpu().float()
            except: pass
        else:
            for attr_name in dir(obj):
                if attr_name.startswith("__"): continue
                try:
                    val = getattr(obj, attr_name)
                    if isinstance(val, torch.nn.Module):
                        for module in val.modules():
                            for param in module.parameters(recurse=False):
                                if param is not None:
                                    param.data = param.data.cpu().float()
                            for buf in module.buffers(recurse=False):
                                if buf is not None:
                                    buf.data = buf.data.cpu().float()
                except: pass

def tiled_encode_4x(tokenizer, frames, target_dtype):
    B, C, T, H, W = frames.shape
    h_mid = H // 2
    w_mid = W // 2
    print(f"\n🧩 Starting 4-Chunk Tiled Encoding (Input: {W}x{H})")
    latents_list = [[None, None], [None, None]]
    
    try:
        print("   👉 Encoding Chunk 1/4 (Top-Left)...")
        with torch.amp.autocast("cuda", dtype=target_dtype):
            l_tl = tokenizer.encode(frames[:, :, :, :h_mid, :w_mid])
        latents_list[0][0] = l_tl.cpu()
        del l_tl; cleanup_memory("After Chunk 1")
    except Exception as e:
        print(f"❌ Chunk 1 Failed: {e}")
        raise e

    try:
        print("   👉 Encoding Chunk 2/4 (Top-Right)...")
        with torch.amp.autocast("cuda", dtype=target_dtype):
            l_tr = tokenizer.encode(frames[:, :, :, :h_mid, w_mid:])
        latents_list[0][1] = l_tr.cpu()
        del l_tr; cleanup_memory("After Chunk 2")
    except Exception as e:
        print(f"❌ Chunk 2 Failed: {e}")
        raise e
    
    try:
        print("   👉 Encoding Chunk 3/4 (Bottom-Left)...")
        with torch.amp.autocast("cuda", dtype=target_dtype):
            l_bl = tokenizer.encode(frames[:, :, :, h_mid:, :w_mid])
        latents_list[1][0] = l_bl.cpu()
        del l_bl; cleanup_memory("After Chunk 3")
    except Exception as e:
        print(f"❌ Chunk 3 Failed: {e}")
        raise e

    try:
        print("   👉 Encoding Chunk 4/4 (Bottom-Right)...")
        with torch.amp.autocast("cuda", dtype=target_dtype):
            l_br = tokenizer.encode(frames[:, :, :, h_mid:, w_mid:])
        latents_list[1][1] = l_br.cpu()
        del l_br; cleanup_memory("After Chunk 4")
    except Exception as e:
        print(f"❌ Chunk 4 Failed: {e}")
        raise e
    
    print("   🧵 Stitching Latents...")
    row1 = torch.cat([latents_list[0][0], latents_list[0][1]], dim=4)
    row2 = torch.cat([latents_list[1][0], latents_list[1][1]], dim=4)
    full_latents = torch.cat([row1, row2], dim=3)
    return full_latents.to(device=tensor_kwargs["device"], dtype=target_dtype)

def safe_cpu_fallback_encode(tokenizer, frames, target_dtype):
    log.warning("🔄 Switching to CPU for VAE Encode (Slow but reliable)...")
    cleanup_memory("Pre-CPU Encode")
    frames_cpu = frames.cpu().to(dtype=torch.float32)
    force_cpu_float32(tokenizer)
    t0 = time.time()
    with torch.autocast("cpu", enabled=False):
        with torch.autocast("cuda", enabled=False):
            latents = tokenizer.encode(frames_cpu)
    print(f"   ⏱️ CPU Encode took: {time.time() - t0:.2f}s")
    return latents.to(device=tensor_kwargs["device"], dtype=target_dtype)

def tiled_decode_gpu(tokenizer, latents, overlap=12):
    """
    Decodes latents in 4 spatial quadrants with OVERLAP and SIGMOID BLENDING.
    Overlap=12 latents (96 pixels). Safe for 720p.
    Removing Global Color Matching to prevent exposure shifts.
    """
    print(f"\n🧱 Starting Tiled GPU Decode (4 Quadrants, Overlap={overlap}, Blended)...")
    B, C, T, H, W = latents.shape
    scale = tokenizer.spatial_compression_factor
    h_mid = H // 2
    w_mid = W // 2
    
    def decode_tile(tile_latents, name):
        cleanup_memory(f"Tile {name}")
        with torch.no_grad():
            return tokenizer.decode(tile_latents).cpu()

    try:
        # 1. Decode Top-Left and Top-Right
        l_tl = latents[..., :h_mid+overlap, :w_mid+overlap]
        l_tr = latents[..., :h_mid+overlap, w_mid-overlap:]
        v_tl = decode_tile(l_tl, "1/4 (TL)")
        v_tr = decode_tile(l_tr, "2/4 (TR)")
        B_dec, C_dec, T_dec, H_tile, W_tile = v_tl.shape
        
        print(f"   🧵 Blending Top Row (Decoded Frames: {T_dec})...")
        mid_pix = w_mid * scale
        overlap_pix = overlap * scale
        
        # Slices for overlap
        tl_blend_slice = v_tl[..., mid_pix-overlap_pix:] 
        tr_blend_slice = v_tr[..., :2*overlap_pix]
        
        row_top = torch.zeros(B_dec, 3, T_dec, H_tile, W*scale, dtype=v_tl.dtype, device='cpu')
        
        # Place non-overlapping parts (Clamped indices)
        end_left = max(0, mid_pix - overlap_pix)
        start_right = mid_pix + overlap_pix
        
        row_top[..., :end_left] = v_tl[..., :end_left]
        row_top[..., start_right:] = v_tr[..., 2*overlap_pix:]
        
        x = torch.linspace(-6, 6, 2*overlap_pix, device='cpu')
        alpha = torch.sigmoid(x).view(1, 1, 1, 1, -1)
        blended_h = tl_blend_slice * (1 - alpha) + tr_blend_slice * alpha
        
        row_top[..., end_left:start_right] = blended_h
        del v_tl, v_tr, l_tl, l_tr
        
        # 3. Decode Bottom-Left and Bottom-Right
        l_bl = latents[..., h_mid-overlap:, :w_mid+overlap]
        l_br = latents[..., h_mid-overlap:, w_mid-overlap:]
        v_bl = decode_tile(l_bl, "3/4 (BL)")
        v_br = decode_tile(l_br, "4/4 (BR)")
        
        print("   🧵 Blending Bottom Row...")
        bl_blend_slice = v_bl[..., mid_pix-overlap_pix:]
        br_blend_slice = v_br[..., :2*overlap_pix]
        
        row_bot = torch.zeros(B_dec, 3, T_dec, H_tile, W*scale, dtype=v_bl.dtype, device='cpu')
        row_bot[..., :end_left] = v_bl[..., :end_left]
        row_bot[..., start_right:] = v_br[..., 2*overlap_pix:]
        row_bot[..., end_left:start_right] = bl_blend_slice * (1 - alpha) + br_blend_slice * alpha
        del v_bl, v_br, l_bl, l_br
        
        # 5. Blend Top and Bottom Vertically
        print("   🧵 Blending Rows Vertically...")
        h_mid_pix = h_mid * scale
        
        # Slices
        top_blend_slice = row_top[..., h_mid_pix-overlap_pix:, :]
        bot_blend_slice = row_bot[..., :2*overlap_pix, :]
        
        video = torch.zeros(B_dec, 3, T_dec, H*scale, W*scale, dtype=row_top.dtype, device='cpu')
        
        end_top = max(0, h_mid_pix - overlap_pix)
        start_bot = h_mid_pix + overlap_pix
        
        video[..., :end_top, :] = row_top[..., :end_top, :]
        video[..., start_bot:, :] = row_bot[..., 2*overlap_pix:, :]
        
        alpha_v = torch.sigmoid(x).view(1, 1, 1, -1, 1)
        blended_v = top_blend_slice * (1 - alpha_v) + bot_blend_slice * alpha_v
        
        video[..., end_top:start_bot, :] = blended_v
            
    except Exception as e:
        print(f"❌ Tiled GPU Decode Failed: {e}")
        raise e 
    return video.to(latents.device)

def load_dit_model(args, is_high_noise=True, force_offload=False):
    """Helper to load the model, respecting overrides."""
    original_offload = args.offload_dit
    if force_offload:
        args.offload_dit = True
        
    path = args.high_noise_model_path if is_high_noise else args.low_noise_model_path
    log.info(f"Loading {'High' if is_high_noise else 'Low'} Noise DiT (Offload={args.offload_dit})...")
    
    try:
        model = create_model(dit_path=path, args=args).cpu()
    finally:
        args.offload_dit = original_offload
        
    return model

if __name__ == "__main__":
    print_memory_status("Script Start")
    args = parse_arguments()

    if args.serve:
        args.mode = "i2v"
        from serve.tui import main as serve_main
        serve_main(args)
        exit(0)
    
    # --- AUTO-ADJUST FRAME COUNT ---
    if (args.num_frames - 1) % 4 != 0:
        old_f = args.num_frames
        new_f = ((old_f - 1) // 4 + 1) * 4 + 1
        print(f"⚠️  Adjusting --num_frames from {old_f} to {new_f} to satisfy VAE temporal stride (4n+1).")
        args.num_frames = new_f

    # --- AUTO-ENABLE OFFLOAD FOR HIGH FRAMES ---
    if args.num_frames > 90 and not args.offload_dit:
        print(f"⚠️  High frame count ({args.num_frames}) detected. Enabling --offload_dit to prevent OOM.")
        args.offload_dit = True

    # 1. Text Embeddings
    if args.cached_embedding and os.path.exists(args.cached_embedding):
        log.info(f"Loading cached embedding from: {args.cached_embedding}")
        cache_data = torch.load(args.cached_embedding, map_location='cpu')
        text_emb = cache_data['embeddings'][0]['embedding'].to(**tensor_kwargs)
    else:
        log.info(f"Computing embedding...")
        text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(**tensor_kwargs)
        clear_umt5_memory()

    # 2. VAE Encoding
    print("-" * 20 + " VAE SETUP " + "-" * 20)
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    target_dtype = tensor_kwargs.get("dtype", torch.bfloat16)
    input_image = Image.open(args.image_path).convert("RGB")
    
    if args.adaptive_resolution:
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        orig_w, orig_h = input_image.size
        aspect = orig_h / orig_w
        ideal_w = np.sqrt(max_resolution_area / aspect)
        ideal_h = np.sqrt(max_resolution_area * aspect)
        stride = tokenizer.spatial_compression_factor * 2
        h = round(ideal_h / stride) * stride
        w = round(ideal_w / stride) * stride
        log.info(f"Adaptive Res: {w}x{h}")
    else:
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    
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
    
    image_tensor = image_transforms(input_image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=target_dtype)
    frames_to_encode = torch.cat([image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device, dtype=target_dtype)], dim=2)
    
    log.info(f"Encoding {F} frames...")
    
    try:
        free_mem, _ = torch.cuda.mem_get_info()
        if free_mem < 24 * (1024**3): 
            raise torch.OutOfMemoryError("Pre-emptive tiling")
        with torch.amp.autocast("cuda", dtype=target_dtype):
            encoded_latents = tokenizer.encode(frames_to_encode)
    except torch.OutOfMemoryError:
        try:
            cleanup_memory("Switching to Tiled Encode")
            encoded_latents = tiled_encode_4x(tokenizer, frames_to_encode, target_dtype)
        except Exception as e:
            log.warning(f"Tiling failed ({e}). Fallback to CPU.")
            encoded_latents = safe_cpu_fallback_encode(tokenizer, frames_to_encode, target_dtype)

    print(f"✅ VAE Encode Complete.")
    del frames_to_encode
    cleanup_memory("After VAE Encode")

    # Prepare for Diffusion
    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0
    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)
    saved_latent_ch = tokenizer.latent_ch 
    
    del tokenizer
    cleanup_memory("Unloaded VAE Model")

    # 3. Diffusion Sampling
    print("-" * 20 + " DIT LOADING " + "-" * 20)
    
    current_model = load_dit_model(args, is_high_noise=True)
    is_high_noise_active = True
    fallback_triggered = args.offload_dit 

    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples), "y_B_C_T_H_W": y}
    
    generator = torch.Generator(device=tensor_kwargs["device"]).manual_seed(args.seed)
    init_noise = torch.randn(args.num_samples, saved_latent_ch, lat_t, lat_h, lat_w, dtype=torch.float32, device=tensor_kwargs["device"], generator=generator)
    
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor([math.atan(args.sigma_max), *mid_t, 0], dtype=torch.float64, device=init_noise.device)
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    
    # Always ensure CUDA initially
    current_model.cuda()
    
    print("-" * 20 + " SAMPLING START " + "-" * 20)
    print_memory_status("High Noise Model to GPU")
    
    # Sampling Loop
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), total=len(t_steps)-1)):
        if t_cur.item() < args.boundary and is_high_noise_active:
            print(f"\n🔄 Switching DiT Models (Step {i})...")
            current_model.cpu()
            del current_model
            cleanup_memory("Unloaded High Noise")
            
            current_model = load_dit_model(args, is_high_noise=False, force_offload=fallback_triggered)
            current_model.cuda() # Force CUDA
            is_high_noise_active = False
            print_memory_status("Loaded Low Noise to GPU")

        step_success = False
        while not step_success:
            try:
                gc.collect()
                torch.cuda.empty_cache() 
                with torch.no_grad():
                    v_pred = current_model(
                        x_B_C_T_H_W=x.to(**tensor_kwargs), 
                        timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), 
                        **condition
                    ).to(torch.float64)
                step_success = True 
            except torch.OutOfMemoryError:
                if fallback_triggered:
                    log.error("❌ OOM occurred even after reload. Physical Memory Limit Reached.")
                    sys.exit(1)
                
                print(f"\n⚠️  OOM in DiT Sampling Step {i}. Reloading model to clear fragmentation...")
                cleanup_memory("Pre-Reload")
                
                # Unload and Reload to Defrag
                was_high = is_high_noise_active
                current_model.cpu()
                del current_model
                cleanup_memory("Unload for Reload")
                
                fallback_triggered = True
                current_model = load_dit_model(args, is_high_noise=was_high, force_offload=True)
                current_model.cuda() # Move back to GPU
                
                print("♻️  Model Reloaded. Retrying step...")

        if args.ode:
            x = x - (t_cur - t_next) * v_pred
        else:
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(*x.shape, dtype=torch.float32, device=tensor_kwargs["device"], generator=generator)
    
    samples = x.float()
    
    print("-" * 20 + " DECODE SETUP (DEFRAG) " + "-" * 20)
    samples_cpu_backup = samples.cpu()
    del samples
    del x
    current_model.cpu()
    del current_model
    cleanup_memory("FULL WIPE before VAE Load")
    
    log.info("Reloading VAE for decoding...")
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    print_memory_status("Reloaded VAE (Clean Slate)")
    
    samples = samples_cpu_backup.to(device=tensor_kwargs["device"])
    
    with torch.no_grad():
        success = False
        video = None
        
        try:
            log.info("Attempting Standard GPU Decode...")
            video = tokenizer.decode(samples)
            success = True
        except torch.OutOfMemoryError:
            log.warning("⚠️ GPU OOM (Standard). Switching to Tiled GPU Decode...")
            cleanup_memory("Pre-Tile Fallback")
            
            try:
                # 12 Latents overlap = 96 Image pixels
                video = tiled_decode_gpu(tokenizer, samples, overlap=12)
                success = True
            except (torch.OutOfMemoryError, RuntimeError) as e:
                log.warning(f"⚠️ GPU Tiled Decode Failed ({e}). Switching to CPU Decode (Slow)...")
                cleanup_memory("Pre-CPU Fallback")
        
        if not success:
            log.info("Performing Hard Cast of VAE to CPU Float32...")
            samples_cpu = samples.cpu().float()
            force_cpu_float32(tokenizer)
            with torch.autocast("cpu", enabled=False):
                with torch.autocast("cuda", enabled=False):
                     video = tokenizer.decode(samples_cpu)

    to_show = [video.float().cpu()]
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0
    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
    log.success("Done.")

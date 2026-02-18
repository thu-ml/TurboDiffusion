# Blackwell Bridge
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# -----------------------------------------------------------------------------------------
# TURBODIFFUSION OPTIMIZED INFERENCE SCRIPT (T2V)
#
# Co-developed by: Waverly Edwards & Google Gemini (2025)
#
# Modifications:
#   - Implemented "Tiered Failover System" for robust OOM protection (GPU -> Checkpoint -> CPU).
#   - Added Intelligent Hardware Detection (TF32/BF16/FP16 auto-switching).
#   - Integrated Tiled Decoding for high-resolution VAE processing.
#   - Added Support for Pre-cached Text Embeddings to skip T5 loading.
#   - Optimized compilation logic for Quantized models (preventing graph breaks).
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

# --- 1. Memory Tuning ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm
import numpy as np

# --- 2. Hardware Optimization ---
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 'high' allows TF32 but maintains reasonable precision
    torch.set_float32_matmul_precision('high')

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

# Suppress graph break warnings for cleaner output
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion inference script for Wan2.1 T2V")
    parser.add_argument("--dit_path", type=str, required=True, help="Custom path to the DiT model checkpoint")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B"], default="Wan2.1-1.3B", help="Model to use")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=80, help="Initial sigma for rCM")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder")
    parser.add_argument("--cached_embedding", type=str, default=None, help="Path to cached text embeddings (pt file)")
    parser.add_argument("--skip_t5", action="store_true", help="Skip T5 loading (implied if cached_embedding is used)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--resolution", default="480p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4", help="Path to save the generated video")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla", help="Type of attention mechanism")
    parser.add_argument("--sla_topk", type=float, default=0.1, help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true", help="Whether to replace Linear layers with quantized versions")
    parser.add_argument("--default_norm", action="store_true", help="Whether to replace LayerNorm/RMSNorm with faster versions")
    parser.add_argument("--offload_dit", action="store_true", help="Offload DiT to CPU when not in use to save VRAM")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (Inductor) for faster inference")
    return parser.parse_args()

def check_hardware_compatibility():
    if not torch.cuda.is_available(): return
    gpu_name = torch.cuda.get_device_name(0)
    log.info(f"Hardware: {gpu_name}")
    
    current_dtype = tensor_kwargs.get("dtype")
    if current_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        log.warning(f"⚠️  Device does not support BFloat16. Switching to Float16.")
        tensor_kwargs["dtype"] = torch.float16

def print_memory_status(step_name=""):
    if not torch.cuda.is_available(): return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024**3)
    print(f"📊 [MEM] {step_name}: InUse={allocated:.2f}GB, Free={free_gb:.2f}GB")

def cleanup_memory(step_info=""):
    gc.collect()
    torch.cuda.empty_cache()

def load_dit_model(args, force_offload=False):
    orig_offload = args.offload_dit
    if force_offload: args.offload_dit = True
    log.info(f"Loading DiT (Offload={args.offload_dit})...")
    model = create_model(dit_path=args.dit_path, args=args).cpu()
    args.offload_dit = orig_offload
    return model

def tiled_decode_gpu(tokenizer, latents, overlap=12):
    print(f"\n🧱 Starting Tiled GPU Decode (Overlap={overlap})...")
    B, C, T, H, W = latents.shape
    scale = tokenizer.spatial_compression_factor
    h_mid = H // 2
    w_mid = W // 2
    
    def decode_tile(tile_latents):
        cleanup_memory()
        with torch.no_grad(): return tokenizer.decode(tile_latents).cpu()

    # 1. Top Tiles
    l_tl = latents[..., :h_mid+overlap, :w_mid+overlap]
    l_tr = latents[..., :h_mid+overlap, w_mid-overlap:]
    v_tl = decode_tile(l_tl)
    v_tr = decode_tile(l_tr)
    
    B_dec, C_dec, T_dec, H_tile, W_tile = v_tl.shape
    mid_pix = w_mid * scale
    overlap_pix = overlap * scale
    
    row_top = torch.zeros(B_dec, 3, T_dec, H_tile, W*scale, dtype=v_tl.dtype, device='cpu')
    end_left = max(0, mid_pix - overlap_pix)
    start_right = mid_pix + overlap_pix
    
    row_top[..., :end_left] = v_tl[..., :end_left]
    row_top[..., start_right:] = v_tr[..., 2*overlap_pix:]
    
    x_linspace = torch.linspace(-6, 6, 2*overlap_pix, device='cpu')
    alpha = torch.sigmoid(x_linspace).view(1, 1, 1, 1, -1)
    row_top[..., end_left:start_right] = v_tl[..., mid_pix-overlap_pix:] * (1 - alpha) + v_tr[..., :2*overlap_pix] * alpha
    del v_tl, v_tr

    # 2. Bottom Tiles
    l_bl = latents[..., h_mid-overlap:, :w_mid+overlap]
    l_br = latents[..., h_mid-overlap:, w_mid-overlap:]
    v_bl = decode_tile(l_bl)
    v_br = decode_tile(l_br)
    
    row_bot = torch.zeros(B_dec, 3, T_dec, H_tile, W*scale, dtype=v_bl.dtype, device='cpu')
    row_bot[..., :end_left] = v_bl[..., :end_left]
    row_bot[..., start_right:] = v_br[..., 2*overlap_pix:]
    row_bot[..., end_left:start_right] = v_bl[..., mid_pix-overlap_pix:] * (1 - alpha) + v_br[..., :2*overlap_pix] * alpha
    del v_bl, v_br

    # 3. Blend Vertically
    h_mid_pix = h_mid * scale
    video = torch.zeros(B_dec, 3, T_dec, H*scale, W*scale, dtype=row_top.dtype, device='cpu')
    end_top = max(0, h_mid_pix - overlap_pix)
    start_bot = h_mid_pix + overlap_pix
    
    video[..., :end_top, :] = row_top[..., :end_top, :]
    video[..., start_bot:, :] = row_bot[..., 2*overlap_pix:, :]
    
    alpha_v = torch.sigmoid(x_linspace).view(1, 1, 1, -1, 1)
    video[..., end_top:start_bot, :] = row_top[..., h_mid_pix-overlap_pix:, :] * (1 - alpha_v) + row_bot[..., :2*overlap_pix, :] * alpha_v
    
    return video.to(latents.device)

def force_cpu_float32(target_obj):
    for module in target_obj.modules():
        module.cpu().float()

def apply_manual_offload(model, device="cuda"):
    log.info("Applying Tier 3 Offload...")
    block_list_name = None
    max_len = 0
    for name, child in model.named_children():
        if isinstance(child, torch.nn.ModuleList):
            if len(child) > max_len:
                max_len = len(child)
                block_list_name = name
    
    if not block_list_name:
        log.warning("Could not identify Block List! Offloading entire model to CPU.")
        model.to("cpu")
        return

    print(f"   👉 Identified Transformer Blocks: '{block_list_name}' ({max_len} layers)")
    try: model.to(device)
    except RuntimeError: model.to("cpu")
        
    blocks = getattr(model, block_list_name)
    blocks.to("cpu")
    
    def pre_hook(module, args):
        module.to(device)
        return args
    def post_hook(module, args, output):
        module.to("cpu")
        return output
    
    for i, block in enumerate(blocks):
        block.register_forward_pre_hook(pre_hook)
        block.register_forward_hook(post_hook)

if __name__ == "__main__":
    print_memory_status("Script Start")
    
    # --- CREDIT PRINT ---
    log.info("----------------------------------------------------------------")
    log.info("🚀 TurboDiffusion Optimized Inference")
    log.info("   Co-developed by Waverly Edwards & Google Gemini")
    log.info("----------------------------------------------------------------")

    check_hardware_compatibility()
    args = parse_arguments()

    if (args.num_frames - 1) % 4 != 0:
        new_f = ((args.num_frames - 1) // 4 + 1) * 4 + 1
        print(f"⚠️  Adjusting --num_frames to {new_f}")
        args.num_frames = new_f

    if args.num_frames > 90 and not args.offload_dit:
        args.offload_dit = True
    
    # --- CRITICAL FIX: Strictly Disable Compile for Quantized Models ---
    if args.compile and args.quant_linear:
        log.warning("🚫 Quantized Model Detected: FORCE DISABLING `torch.compile` to avoid OOM.")
        log.warning("   (Custom quantized kernels are not compatible with CUDA Graphs)")
        args.compile = False

    # 1. Text Embeddings
    if args.cached_embedding and os.path.exists(args.cached_embedding):
        log.info(f"Loading cache: {args.cached_embedding}")
        c = torch.load(args.cached_embedding, map_location='cpu')
        text_emb = c['embeddings'][0]['embedding'].to(**tensor_kwargs) if isinstance(c, dict) else c.to(**tensor_kwargs)
    else:
        log.info(f"Computing embedding...")
        with torch.no_grad():
            text_emb = get_umt5_embedding(args.text_encoder_path, args.prompt).to(**tensor_kwargs)
        clear_umt5_memory()
    cleanup_memory()

    # 2. VAE Shape Calc & UNLOAD
    log.info("VAE Setup (Temp)...")
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    state_shape = [tokenizer.latent_ch, tokenizer.get_latent_num_frames(args.num_frames), h // tokenizer.spatial_compression_factor, w // tokenizer.spatial_compression_factor]
    del tokenizer
    cleanup_memory("VAE Unloaded")

    # 3. Load DiT
    net = load_dit_model(args)
    
    # 4. Noise & Schedule
    gen = torch.Generator(device=tensor_kwargs["device"]).manual_seed(args.seed)
    cond = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}
    init_noise = torch.randn(args.num_samples, *state_shape, dtype=torch.float32, device=tensor_kwargs["device"], generator=gen)
    
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor([math.atan(args.sigma_max), *mid_t, 0], dtype=torch.float64, device=init_noise.device)
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

    # 5. Fast Sampling Loop
    log.info("🔥 STARTING SAMPLING (INFERENCE MODE) 🔥")
    torch.cuda.empty_cache()
    net.cuda()
    print_memory_status("Tier 1: GPU Ready")
    
    # Compile? (Only if NOT disabled above)
    if args.compile:
        log.info("🚀 Compiling model...")
        try:
            net = torch.compile(net, mode="reduce-overhead")
        except Exception as e:
            log.warning(f"Compile failed: {e}. Running eager.")

    failover = 0
    
    with torch.inference_mode():
        for i, (t_cur, t_next) in enumerate(tqdm(zip(t_steps[:-1], t_steps[1:]), total=len(t_steps)-1)):
            retry = True
            while retry:
                try:
                    t_cur_scalar = t_cur.item()
                    t_next_scalar = t_next.item()
                    
                    v_pred = net(
                        x_B_C_T_H_W=x.to(**tensor_kwargs), 
                        timesteps_B_T=(t_cur * ones * 1000).to(**tensor_kwargs), 
                        **cond
                    ).to(torch.float64)
                    
                    if args.offload_dit and i == len(t_steps)-2 and failover == 0:
                        net.cpu()

                    noise = torch.randn(*x.shape, dtype=torch.float32, device=x.device, generator=gen).to(torch.float64)
                    term1 = x - (v_pred * t_cur_scalar)
                    x = term1 * (1.0 - t_next_scalar) + (noise * t_next_scalar)
                    
                    retry = False
                    
                except torch.OutOfMemoryError:
                    log.warning(f"⚠️ OOM at Step {i}. Recovering...")
                    try: net.cpu() 
                    except: pass
                    del net
                    cleanup_memory()
                    failover += 1
                    
                    if failover == 1:
                        print("♻️ Tier 2: Checkpointing")
                        net = load_dit_model(args, force_offload=True)
                        net.cuda()
                        # Retry compile with safer mode if first attempt was aggressive
                        if args.compile:
                            try: net = torch.compile(net, mode="default")
                            except: pass
                    elif failover == 2:
                        print("♻️ Tier 3: Manual Offload")
                        net = load_dit_model(args, force_offload=True)
                        apply_manual_offload(net)
                    else:
                        sys.exit("❌ Critical OOM.")

    samples = x.float()

    # 6. Decode
    if 'net' in locals():
        try: net.cpu() 
        except: pass
        del net
    cleanup_memory("Pre-VAE")
    
    log.info("Decoding...")
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    with torch.no_grad():
        try:
            video = tokenizer.decode(samples)
        except torch.OutOfMemoryError:
            log.warning("Falling back to Tiled Decode...")
            video = tiled_decode_gpu(tokenizer, samples)

    to_show = [video.float().cpu()]
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0
    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
    log.success(f"Saved: {args.save_path}")

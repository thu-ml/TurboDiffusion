import os
import sys
import subprocess
import gradio as gr
import glob
import random
import time
import select
import torch
from datetime import datetime

# --- 1. System Setup ---
PROJECT_ROOT = "/home/wedwards/Documents/Programs/TurboDiffusion"
os.chdir(PROJECT_ROOT)
os.system('clear' if os.name == 'posix' else 'cls')

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

T2V_SCRIPT = "turbodiffusion/inference/wan2.1_t2v_infer.py"
I2V_SCRIPT = "turbodiffusion/inference/wan2.2_i2v_infer.py"
CACHE_SCRIPT = "turbodiffusion/inference/cache_t5.py"

def get_gpu_status_original():
    """System-level GPU check."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding='utf-8'
        ).strip().split(',')
        return f"🖥️ {res[0]} | ⚡ VRAM: {res[1]}MB / {res[2]}MB"
    except:
        return "🖥️ GPU Monitor Active"


def get_gpu_status():
    """
    Check GPU status using PyTorch. 
    Returns system-wide VRAM usage without relying on nvidia-smi CLI.
    """
    try:
        # 1. Check for CUDA (NVIDIA) or ROCm (AMD)
        if torch.cuda.is_available():
            # mem_get_info returns (free_bytes, total_bytes)
            free_mem, total_mem = torch.cuda.mem_get_info()
            
            used_mem = total_mem - free_mem
            
            # Convert to MB for display
            total_mb = int(total_mem / 1024**2)
            used_mb = int(used_mem / 1024**2)
            name = torch.cuda.get_device_name(0)
            
            return f"🖥️ {name} | ⚡ VRAM: {used_mb}MB / {total_mb}MB"

        # 2. Check for Apple Silicon (MPS)
        # Note: Apple uses Unified Memory, so 'VRAM' is shared with System RAM.
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "🖥️ Apple Silicon (MPS) | ⚡ Unified Memory Active"

        # 3. Fallback to CPU
        else:
            return "🖥️ Running on CPU"

    except ImportError:
        return "🖥️ GPU Monitor: PyTorch not installed"
    except Exception as e:
        return f"🖥️ GPU Monitor Error: {str(e)}"

def save_debug_metadata(video_path, script_rel, cmd_list, cache_cmd_list=None):
    """
    Saves a fully executable reproduction script with env vars.
    """
    meta_path = video_path.replace(".mp4", "_metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Copy and paste the lines below to reproduce this video exactly:\n\n")
        
        # Environment Variables
        f.write("export PYTHONPATH=turbodiffusion\n")
        f.write("export PYTORCH_ALLOC_CONF=expandable_segments:True\n")
        f.write("export TOKENIZERS_PARALLELISM=false\n\n")
        
        # Optional Cache Step
        if cache_cmd_list:
            f.write("# --- Step 1: Pre-Cache Embeddings ---\n")
            f.write(f"python {CACHE_SCRIPT} \\\n")
            c_args = cache_cmd_list[2:]
            for i, arg in enumerate(c_args):
                if arg.startswith("--"):
                    val = f'"{c_args[i+1]}"' if i+1 < len(c_args) and not c_args[i+1].startswith("--") else ""
                    f.write(f"    {arg} {val} \\\n")
            f.write("\n# --- Step 2: Run Inference ---\n")

        # Main Inference Command
        f.write(f"python {script_rel} \\\n")
        args_only = cmd_list[2:]
        for i, arg in enumerate(args_only):
            if arg.startswith("--"):
                val = f'"{args_only[i+1]}"' if i+1 < len(args_only) and not args_only[i+1].startswith("--") else ""
                f.write(f"    {arg} {val} \\\n")

def sync_path(scale):
    fname = "TurboWan2.1-T2V-1.3B-480P-quant.pth" if "1.3B" in scale else "TurboWan2.1-T2V-14B-720P-quant.pth"
    return os.path.join(CHECKPOINT_DIR, fname)

# --- 2. Unified Generation Logic (With Safety Checks) ---

def run_gen(mode, prompt, model, dit_path, i2v_high, i2v_low, image, res, ratio, steps, seed, quant, attn, top_k, frames, sigma, norm, adapt, ode, use_cache, cache_path, pr=gr.Progress()):
    # --- PRE-FLIGHT SAFETY CHECK ---
    error_msg = ""
    if mode == "T2V":
        if "quant" in dit_path.lower() and not quant:
            error_msg = "❌ CONFIG ERROR: Quantized model selected but '8-bit' disabled."
        if attn == "original" and ("turbo" in dit_path.lower() or "quant" in dit_path.lower()):
            error_msg = "❌ COMPATIBILITY ERROR: 'Original' attention with Turbo/Quantized checkpoint."
    else:
        if ("quant" in i2v_high.lower() or "quant" in i2v_low.lower()) and not quant:
            error_msg = "❌ CONFIG ERROR: Quantized I2V model selected but '8-bit' disabled."
        if attn == "original" and (("turbo" in i2v_high.lower() or "quant" in i2v_high.lower()) or ("turbo" in i2v_low.lower() or "quant" in i2v_low.lower())):
            error_msg = "❌ COMPATIBILITY ERROR: 'Original' attention with Turbo/Quantized checkpoints."

    if error_msg:
        yield None, None, "❌ Config Error", "🛑 Aborted", error_msg
        return
    # -------------------------------

    actual_seed = random.randint(1, 1000000) if seed <= 0 else int(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    
    full_log = f"🚀 Starting Job: {timestamp}\n"
    pr(0, desc="🚀 Starting...")

    # --- FRAME SANITIZATION (4n+1 RULE) ---
    # Wan2.1 VAE requires frames to be (4n + 1). If not, we sanitize.
    target_frames = int(frames)
    valid_frames = ((target_frames - 1) // 4) * 4 + 1
    
    # If the user input (e.g., 32) became smaller (29) or changed, we log it.
    # Note: We enforce a minimum of 1 frame just in case.
    valid_frames = max(1, valid_frames)

    if valid_frames != target_frames:
        warning_msg = f"⚠️ AUTO-ADJUST: Frame count {target_frames} is incompatible with VAE (requires 4n+1).\n"
        warning_msg += f"   Adjusted {target_frames} -> {valid_frames} frames to prevent kernel crash.\n"
        full_log += warning_msg
        print(warning_msg) # Print to console as well
    
    # Use valid_frames for the rest of the logic
    frames = valid_frames
    # --------------------------------------

    # --- AUTO-CACHE STEP ---
    cache_cmd_list = None
    if use_cache:
        pr(0, desc="💾 Auto-Caching T5 Embeddings...")
        cache_script_full = os.path.join(PROJECT_ROOT, CACHE_SCRIPT)
        encoder_path = os.path.join(CHECKPOINT_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
        
        cache_cmd = [
            sys.executable,
            cache_script_full,
            "--prompt", prompt,
            "--output", cache_path,
            "--text_encoder_path", encoder_path
        ]
        cache_cmd_list = cache_cmd 
        
        full_log += f"\n[System] Running Cache Script: {' '.join(cache_cmd)}\n"
        yield None, None, f"Seed: {actual_seed}", "💾 Caching...", full_log
        
        cache_process = subprocess.Popen(cache_cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        while True:
            if cache_process.poll() is not None:
                rest = cache_process.stdout.read()
                if rest: full_log += rest
                break
            line = cache_process.stdout.readline()
            if line:
                full_log += line
                yield None, None, f"Seed: {actual_seed}", "💾 Caching...", full_log
            time.sleep(0.02)

        if cache_process.returncode != 0:
            full_log += "\n❌ CACHE FAILED. Aborting generation."
            yield None, None, "❌ Cache Failed", "🛑 Aborted", full_log
            return
            
        full_log += "\n✅ Cache Complete. Starting Inference...\n"
    # -----------------------------------------
    
    # --- SETUP VIDEO GENERATION ---
    if mode == "T2V":
        save_path = os.path.join(OUTPUT_DIR, f"t2v_{timestamp}.mp4")
        script_rel = T2V_SCRIPT
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, T2V_SCRIPT), "--model", model, "--dit_path", dit_path, "--prompt", prompt, "--resolution", res, "--aspect_ratio", ratio, "--num_steps", str(steps), "--seed", str(actual_seed), "--attention_type", attn, "--sla_topk", str(top_k), "--num_samples", "1", "--num_frames", str(frames), "--sigma_max", str(sigma)]
    else:
        save_path = os.path.join(OUTPUT_DIR, f"i2v_{timestamp}.mp4")
        script_rel = I2V_SCRIPT
        # Note: Added frames to I2V command in previous step, maintained here.
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, I2V_SCRIPT), "--prompt", prompt, "--image_path", image, "--high_noise_model_path", i2v_high, "--low_noise_model_path", i2v_low, "--resolution", res, "--aspect_ratio", ratio, "--num_steps", str(steps), "--seed", str(actual_seed), "--attention_type", attn, "--sla_topk", str(top_k), "--num_frames", str(frames)]
        if adapt: cmd.append("--adaptive_resolution")
        if ode: cmd.append("--ode")

    if quant: cmd.append("--quant_linear")
    if norm: cmd.append("--default_norm")

    if use_cache:
        cmd.extend(["--cached_embedding", cache_path, "--skip_t5"])

    cmd.extend(["--save_path", save_path])

    # Call the restored metadata saver
    save_debug_metadata(save_path, script_rel, cmd, cache_cmd_list)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "turbodiffusion")
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"

    full_log += f"\n[System] Running Inference: {' '.join(cmd)}\n"
    process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    last_ui_update = 0
    
    while True:
        if process.poll() is not None:
            rest = process.stdout.read()
            if rest: full_log += rest
            break

        reads = [process.stdout.fileno()]
        ret = select.select(reads, [], [], 0.1)

        if ret[0]:
            line = process.stdout.readline()
            full_log += line
            
            if "Loading DiT" in line: pr(0.1, desc="⚡ Loading weights...")
            if "Encoding" in line: pr(0.05, desc="🖼️ VAE Encoding...")
            if "Switching to CPU" in line: pr(0.1, desc="⚠️ CPU Fallback...")
            if "Sampling:" in line:
                try:
                    pct = int(line.split('%')[0].split('|')[-1].strip())
                    pr(0.2 + (pct/100 * 0.7), desc=f"🎬 Sampling: {pct}%")
                except: pass
            if "decoding" in line.lower(): pr(0.95, desc="🎥 Decoding VAE...")

        current_time = time.time()
        if current_time - last_ui_update > 0.25:
            last_ui_update = current_time
            elapsed = f"{int(current_time - start_time)}s"
            yield None, None, f"Seed: {actual_seed}", f"⏱️ Time: {elapsed}", full_log

    history = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
    total_time = f"{int(time.time() - start_time)}s"
    
    yield save_path, history, f"✅ Done | Seed: {actual_seed}", f"🏁 Finished in {total_time}", full_log

# --- 3. UI Layout ---
with gr.Blocks(title="TurboDiffusion Studio") as demo:
    with gr.Row():
        gr.HTML("<h2 style='margin: 10px 0;'>⚡ TurboDiffusion Studio</h2>")
        with gr.Column(scale=1):
            gpu_display = gr.Markdown(get_gpu_status())
    
    gr.Timer(2).tick(get_gpu_status, outputs=gpu_display)

    with gr.Tabs():
        with gr.Tab("Text-to-Video"):
            with gr.Row():
                with gr.Column(scale=4):
                    t2v_p = gr.Textbox(label="Prompt", lines=3, value="A stylish woman walks down a Tokyo street...")
                    with gr.Row():
                        t2v_m = gr.Radio(["Wan2.1-1.3B", "Wan2.1-14B"], label="Model", value="Wan2.1-1.3B")
                        t2v_res = gr.Dropdown(["480p", "720p"], label="Resolution", value="480p")
                        t2v_ratio = gr.Dropdown(["16:9", "4:3", "1:1", "9:16"], label="Aspect Ratio", value="16:9")
                    t2v_dit = gr.Textbox(label="DiT Path", value=sync_path("Wan2.1-1.3B"), interactive=False)
                    t2v_btn = gr.Button("Generate Video", variant="primary")
                with gr.Column(scale=3):
                    t2v_out = gr.Video(label="Result", height=320)
                    with gr.Row():
                        t2v_stat = gr.Textbox(label="Status", interactive=False, scale=2)
                        t2v_time = gr.Textbox(label="Timer", value="⏱️ Ready", interactive=False, scale=1)

        with gr.Tab("Image-to-Video"):
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                        i2v_img = gr.Image(label="Source", type="filepath", height=200)
                        i2v_p = gr.Textbox(label="Motion Prompt", lines=7)
                    with gr.Row():
                        i2v_res = gr.Dropdown(["480p", "720p"], label="Resolution", value="720p")
                        i2v_ratio = gr.Dropdown(["16:9", "4:3", "1:1", "9:16"], label="Aspect Ratio", value="16:9")
                    with gr.Row():
                        i2v_adapt = gr.Checkbox(label="Adaptive Resolution", value=True)
                        i2v_ode = gr.Checkbox(label="Use ODE", value=False)
                    with gr.Accordion("I2V Path Overrides", open=False):
                        i2v_high = gr.Textbox(label="High-Noise", value=os.path.join(CHECKPOINT_DIR, "TurboWan2.2-I2V-A14B-high-720P-quant.pth"))
                        i2v_low = gr.Textbox(label="Low-Noise", value=os.path.join(CHECKPOINT_DIR, "TurboWan2.2-I2V-A14B-low-720P-quant.pth"))
                    i2v_btn = gr.Button("Animate Image", variant="primary")
                with gr.Column(scale=3):
                    i2v_out = gr.Video(label="Result", height=320)
                    with gr.Row():
                        i2v_stat_2 = gr.Textbox(label="Status", interactive=False, scale=2)
                        i2v_time_2 = gr.Textbox(label="Timer", value="⏱️ Ready", interactive=False, scale=1)

    console_out = gr.Textbox(label="Live CLI Console Output", lines=8, max_lines=8, interactive=False)

    with gr.Accordion("⚙️ Precision & Advanced Settings", open=False):
        with gr.Row():
            quant_opt = gr.Checkbox(label="Enable --quant_linear (8-bit)", value=True)
            steps_opt = gr.Slider(1, 4, value=4, step=1, label="Steps")
            seed_opt = gr.Number(label="Seed (0=Random)", value=0, precision=0)
        with gr.Row():
            top_k_opt = gr.Slider(0.01, 0.5, value=0.15, step=0.01, label="SLA Top-K")
            attn_opt = gr.Radio(["sagesla", "sla", "original"], label="Attention", value="sagesla")
            sigma_opt = gr.Number(label="Sigma Max", value=80)
            norm_opt = gr.Checkbox(label="Original Norms", value=False)
            frames_opt = gr.Slider(1, 120, value=77, step=4, label="Frames (Steps of 4)")
        with gr.Row(variant="panel"):
            # --- T5 CACHE UI ---
            use_cache_opt = gr.Checkbox(label="Use Cached T5 Embeddings (Auto-Run)", value=True)
            cache_path_opt = gr.Textbox(label="Cache File Path", value="cached_t5_embeddings.pt", scale=2)
            # -------------------

    history_gal = gr.Gallery(value=sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), reverse=True), columns=6, height="auto")

    # --- 4. Logic Bindings ---
    t2v_m.change(fn=sync_path, inputs=t2v_m, outputs=t2v_dit)
    
    t2v_args = [gr.State("T2V"), t2v_p, t2v_m, t2v_dit, gr.State(""), gr.State(""), gr.State(""), t2v_res, t2v_ratio, steps_opt, seed_opt, quant_opt, attn_opt, top_k_opt, frames_opt, sigma_opt, norm_opt, gr.State(False), gr.State(False), use_cache_opt, cache_path_opt]
    t2v_btn.click(run_gen, t2v_args, [t2v_out, history_gal, t2v_stat, t2v_time, console_out], show_progress="hidden")

    i2v_args = [i2v_img, i2v_p, gr.State("Wan2.2-A14B"), gr.State(""), i2v_high, i2v_low, i2v_img, i2v_res, i2v_ratio, steps_opt, seed_opt, quant_opt, attn_opt, top_k_opt, frames_opt, gr.State(200), norm_opt, i2v_adapt, i2v_ode, use_cache_opt, cache_path_opt]
    i2v_btn.click(run_gen, i2v_args, [i2v_out, history_gal, i2v_stat_2, i2v_time_2, console_out], show_progress="hidden")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(), allowed_paths=[OUTPUT_DIR])
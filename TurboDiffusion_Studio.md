
## 🚀 Scripts & Inference

This repository contains optimized inference engines for the Wan2.1 and Wan2.2 models, specifically tuned for high-resolution output and robust memory management on consumer hardware.

### 🎥 Inference Engines

| Script | Function | Key Features |
| --- | --- | --- |
| **`wan2.2_i2v_infer.py`** | **Image-to-Video** | **Tiered Failover System**: Automatic recovery from OOM errors.<br>

<br> **Intelligent Model Switching**: Transitions between High and Low Noise models based on step boundaries.<br>

<br> **Tiled Processing**: Uses 4-chunk tiled encoding/decoding for 720p+ stability. |
| **`wan2.1_t2v_infer.py`** | **Text-to-Video** | **Hardware Auto-Detection**: Automatically selects TF32, BF16, or FP16 based on GPU capabilities.<br>

<br> **Quantization Safety**: Force-disables `torch.compile` for quantized models to prevent graph-break OOMs.<br>

<br> **3-Tier Recovery**: Escalates from GPU ➔ Checkpointing ➔ Manual CPU Offloading if memory is exceeded. |

### 🛠️ Utilities

* **`cache_t5.py`**
* **Purpose**: Pre-computes and saves T5 text embeddings to disk.
* **VRAM Benefit**: Eliminates the need to load the **11GB T5 encoder** during the main inference run, allowing 14B models to fit on GPUs with lower VRAM.
* **Usage**: Run this first to generate a `.pt` file, then pass it to the inference scripts using the `--cached_embedding` flag.


---

## 🚀 Getting Started with TurboDiffusion

To run the large 14B models on consumer GPUs, it is recommended to use the **T5 Caching** workflow. This offloads the 11GB text encoder from VRAM, leaving more space for the DiT model and high-resolution video decoding.

### **Step 1: Environment Setup**

Ensure your project structure is organized as follows:

* **Root**: `/your/path/to/TurboDiffusion`
* **Checkpoints**: Place your `.pth` models in the `checkpoints/` directory.
* **Output**: Generated videos and metadata will be saved to `output/`.

### **Step 2: The Two Ways to Cache T5**

#### **Option A: Manual Pre-Caching (Recommended for Batching)**

If you have a list of prompts you want to use frequently, use the standalone utility:

```bash
python turbodiffusion/inference/cache_t5.py --prompt "Your descriptive prompt here" --output cached_t5_embeddings.pt

```

This saves the processed text into a small `.pt` file, allowing the inference scripts to "skip" the heavy T5 model entirely.

#### **Option B: Automatic Caching via Web UI**

For a more streamlined experience, use the **TurboDiffusion Studio**:

1. Launch the UI: `python turbo_diffusion_t5_cache_optimize_v6.py`.
2. Open the **Precision & Advanced Settings** accordion.
3. Check **Use Cached T5 Embeddings (Auto-Run)**.
4. When you click generate, the UI will automatically run the caching script first, clear the T5 model from memory, and then start the video generation.

### **Step 3: Running Inference**

Once your UI is launched and caching is configured:

1. **Select Mode**: Choose between **Text-to-Video** (Wan2.1) or **Image-to-Video** (Wan2.2).
2. **Apply Quantization**: For 24GB VRAM GPUs (like the RTX 3090/4090/5090), ensure **Enable --quant_linear (8-bit)** is checked to avoid OOM errors.
3. **Monitor Hardware**: Watch the **Live GPU Monitor** at the top of the UI to track real-time VRAM usage during the sampling process.
4. **Retrieve Results**: Your video and its reproduction metadata (containing the exact CLI command used) will appear in the `output/` gallery.


---

## 🖥️ TurboDiffusion Studio (Web UI)

The `turbo_diffusion_t5_cache_optimize_v6.py` script provides a high-performance, unified **Gradio-based Web interface** for both Text-to-Video and Image-to-Video generation. It serves as a centralized "Studio" dashboard that automates complex environment setups and memory optimizations.

### **Key Features**

| Feature | Description |
| --- | --- |
| **Unified Interface** | Toggle between **Text-to-Video (Wan2.1)** and **Image-to-Video (Wan2.2)** workflows within a single dashboard. |
| **Real-time GPU Monitor** | Native PyTorch-based VRAM monitoring that displays current memory usage and hardware status directly in the UI. |
| **Auto-Cache T5 Integration** | Automatically runs the `cache_t5.py` utility before inference to offload the 11GB text encoder, significantly reducing peak VRAM usage. |
| **Frame Sanitization** | Automatically enforces the **4n + 1 rule** required by the Wan VAE to prevent kernel crashes during decoding. |
| **Reproduction Metadata** | Every generated video automatically saves a matching `_metadata.txt` file containing the exact CLI command and environment variables needed to reproduce the result. |
| **Live Console Output** | Pipes real-time CLI logs and progress bars directly into a "Live Console" window in the web browser. |

### **Advanced Controls**

The UI exposes granular controls for technical users:

* **Precision & Quantization:** Toggle 8-bit `--quant_linear` mode for low-VRAM operation.
* **Attention Tuning:** Switch between `sagesla`, `sla`, and `original` attention mechanisms.
* **Adaptive I2V:** Enable adaptive resolution and ODE solvers for Image-to-Video workflows.
* **Integrated Gallery:** Browse and view your output history directly within the `output/` directory.

---

## 🛠️ Usage

To launch the studio:

```bash
python turbo_diffusion_t5_cache_optimize_v6.py

```

> **Note:** The script defaults to `/your/path/to/TurboDiffusion`as the project root. Ensure your local paths are configured accordingly in the **System Setup** section of the code.


---

## 💳 Credits & Acknowledgments

If you utilize, share, or build upon these optimized scripts, please include the following acknowledgments:

* **Optimization & Development**: Co-developed by **Waverly Edwards** and **Google Gemini**.
* **T5 Caching Logic**: Original concept and utility implementation by **John D. Pope**.
* **Base Framework**: Built upon the NVIDIA Imaginaire and Wan-Video research.

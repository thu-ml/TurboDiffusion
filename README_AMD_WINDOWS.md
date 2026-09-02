# TurboDiffusion - AMD ROCm on Windows Setup Guide

This guide explains how to build and run TurboDiffusion on Windows with AMD GPUs using ROCm.

> **Note:** These steps should also work on Linux with minor modifications (use bash commands instead of PowerShell, `source venv/bin/activate` instead of `.\venv\Scripts\Activate.ps1`, and skip the Visual Studio environment setup). However, Linux support has not been tested yet and may have issues.

## Supported Hardware

TurboDiffusion on Windows has been tested with RDNA3/RDNA3.5 GPUs (gfx1100, gfx1101, gfx1102, gfx1151).

## Prerequisites

- Windows 10/11
- Python 3.11, 3.12, or 3.13
- Visual Studio 2022 with C++ build tools
- AMD Adrenaline driver (latest recommended)

## Installation

### 1. Install ROCm and PyTorch from TheRock

Follow the instructions at [ROCm/TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) to install ROCm and PyTorch wheels for your GPU architecture.

#### Create a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Install PyTorch (includes ROCm SDK as dependency)

For **gfx1151** (AMD Strix Halo iGPU):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision
```

For **gfx110X** (RX 7900 XTX, RX 7800 XT, RX 7700S, Radeon 780M):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision
```

For **gfx120X** (RX 9060, RX 9070):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre torch torchaudio torchvision
```

#### Initialize ROCm SDK

```powershell
rocm-sdk init
```

#### Install Triton with AMD Windows Support

```powershell
pip install triton-windows
```

### 2. Set Environment Variables

Open a PowerShell terminal and run:

```powershell
# Activate Visual Studio environment
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Set ROCm paths using rocm-sdk
$ROCM_ROOT = (rocm-sdk path --root).Trim()
$ROCM_BIN = (rocm-sdk path --bin).Trim()
$env:ROCM_HOME = $ROCM_ROOT
$env:PATH = "$ROCM_ROOT\lib\llvm\bin;$ROCM_BIN;$env:PATH"

# Set compiler and build settings
$env:CC = "clang-cl"
$env:CXX = "clang-cl"
$env:DISTUTILS_USE_SDK = "1"

# Enable experimental features
$env:FLASH_ATTENTION_TRITON_AMD_ENABLE = "TRUE"
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"

# Set PYTHONPATH for TurboDiffusion
$env:PYTHONPATH = "turbodiffusion"
```

### 3. Build and Install TurboDiffusion

```powershell
cd <path_to_turbodiffusion>
pip install --no-build-isolation -e .
```

### 4. Install SpargeAttn (Optional, for sparse attention)

If you want to use sparse attention with TurboDiffusion, clone the AMD Windows fork:

```powershell
git clone --branch jam/amd_windows https://github.com/jammm/SpargeAttn.git
cd SpargeAttn
pip install --no-build-isolation -v .
```

## Running Inference

### Text-to-Video with Wan2.1

```powershell
# Make sure environment variables are set (see step 2)

python turbodiffusion/inference/wan2.1_t2v_infer.py `
    --model Wan2.1-1.3B `
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth `
    --resolution 480p `
    --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." `
    --num_samples 1 `
    --num_steps 4 `
    --quant_linear `
    --attention_type sagesla `
    --sla_topk 0.1
```

### Available Attention Types

- `sdpa` - PyTorch Scaled Dot Product Attention
- `sagesla` - SageAttention with Sparse Linear Attention (requires SpargeAttn)

## Environment Variable Summary

| Variable | Value | Description |
|----------|-------|-------------|
| `CC` | `clang-cl` | C compiler |
| `CXX` | `clang-cl` | C++ compiler |
| `DISTUTILS_USE_SDK` | `1` | Use SDK for distutils |
| `ROCM_HOME` | `<rocm-sdk path --root>` | ROCm SDK root path |
| `PATH` | Include LLVM and ROCm bin | Required for hipcc, clang, lld-link |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | Enable Triton Flash Attention on AMD |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | Enable experimental aotriton kernels |
| `PYTHONPATH` | `turbodiffusion` | Include turbodiffusion module |

## Known Issues

1. **Triton compiler warnings** - You may see `clang-cl: warning: unknown argument ignored` warnings during first run. These are harmless.

2. **First run is slow** - Triton and MIOpen kernels are compiled on first use and cached. Subsequent runs will be faster.

3. **No FP8 support on RDNA3** - RDNA3 GPUs don't support FP8, so FP16/BF16 kernels are used.

## Troubleshooting

### "LoadLibrary failed" or "cannot find amdhip64.dll"

Make sure you ran `rocm-sdk init` after installing the ROCm SDK packages.

### "LINK : fatal error LNK1104: cannot open file 'python312.lib'"

Ensure Visual Studio environment is activated before building:
```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
```

### "PermissionError" when compiling Triton kernels

This is a known Windows issue with temp file handling. Make sure you're using the latest `triton-windows` package (`pip install --upgrade triton-windows`).

### "flash_attn is not installed" warning

This warning is expected. Flash Attention is not available on AMD GPUs, but Triton-based attention is used instead when `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` is set.


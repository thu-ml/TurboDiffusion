#!/bin/bash
# TurboDiffusion Installation Script
# For RTX 5090 (Blackwell) with CUDA 13.0

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TurboDiffusion Installation Script"
echo "=============================================="
echo ""

# =============================================================================
# Check for Miniconda
# =============================================================================
check_conda() {
    if command -v conda &> /dev/null; then
        echo "✅ Conda found: $(conda --version)"
        return 0
    fi

    # Check common install locations
    for conda_path in ~/miniconda3/bin/conda ~/anaconda3/bin/conda /opt/conda/bin/conda; do
        if [ -f "$conda_path" ]; then
            echo "✅ Found conda at: $conda_path"
            eval "$($conda_path shell.bash hook)"
            return 0
        fi
    done

    return 1
}

install_miniconda() {
    echo ""
    echo "❌ Conda/Miniconda not found!"
    echo ""
    echo "Please install Miniconda first:"
    echo ""
    echo "  # Download Miniconda (Linux x86_64)"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "  # Install (follow prompts)"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "  # Restart shell or run:"
    echo "  source ~/.bashrc"
    echo ""
    echo "  # Then re-run this script"
    echo ""

    read -p "Would you like to download and install Miniconda now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading Miniconda..."
        wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

        echo "Installing Miniconda to ~/miniconda3..."
        bash /tmp/miniconda.sh -b -p ~/miniconda3

        echo "Initializing conda..."
        ~/miniconda3/bin/conda init bash
        eval "$(~/miniconda3/bin/conda shell.bash hook)"

        rm /tmp/miniconda.sh
        echo "✅ Miniconda installed!"
        return 0
    else
        exit 1
    fi
}

if ! check_conda; then
    install_miniconda
fi

# Source conda for current shell
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

# =============================================================================
# Check for CUDA
# =============================================================================
echo ""
echo "Checking CUDA..."

if ! command -v nvcc &> /dev/null; then
    echo "⚠️  nvcc not found in PATH"
    # Check common locations
    for cuda_path in /usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda; do
        if [ -f "$cuda_path/bin/nvcc" ]; then
            echo "   Found CUDA at: $cuda_path"
            export PATH="$cuda_path/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
            break
        fi
    done
fi

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    echo "✅ CUDA version: $CUDA_VERSION"
else
    echo "❌ CUDA not found. Please install CUDA 13.0 for RTX 5090 support."
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "✅ GPU: $GPU_NAME ($GPU_MEMORY)"
fi

# =============================================================================
# Create/Activate Conda Environment
# =============================================================================
ENV_NAME="turbodiffusion"

echo ""
echo "Setting up conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "   Environment '$ENV_NAME' already exists"
    read -p "   Recreate environment? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing environment..."
        conda env remove -n $ENV_NAME -y
        echo "   Creating fresh environment..."
        conda create -n $ENV_NAME python=3.12 -y
    fi
else
    echo "   Creating new environment with Python 3.12..."
    conda create -n $ENV_NAME python=3.12 -y
fi

echo "   Activating environment..."
conda activate $ENV_NAME

echo "✅ Python: $(python --version)"

# =============================================================================
# Install PyTorch with CUDA 13.0 (Nightly for Blackwell support)
# =============================================================================
echo ""
echo "Installing PyTorch with CUDA 13.0 support..."
echo "   (Nightly build required for RTX 5090/Blackwell)"

pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# Verify PyTorch installation
python -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda}')" || {
    echo "❌ PyTorch installation failed"
    exit 1
}

# =============================================================================
# Install Dependencies
# =============================================================================
echo ""
echo "Installing dependencies..."

pip install psutil

# =============================================================================
# Initialize Git Submodules (CUTLASS)
# =============================================================================
echo ""
echo "Initializing git submodules (CUTLASS)..."

if [ -d ".git" ]; then
    git submodule update --init --recursive
    echo "✅ Submodules initialized"
else
    echo "⚠️  Not a git repository, checking if CUTLASS exists..."
    if [ ! -f "turbodiffusion/ops/cutlass/include/cutlass/cutlass.h" ]; then
        echo "❌ CUTLASS not found. Please clone with: git clone --recursive <repo>"
        exit 1
    fi
fi

# Verify CUTLASS headers
if [ ! -f "turbodiffusion/ops/cutlass/include/cutlass/cutlass.h" ]; then
    echo "❌ CUTLASS headers not found after submodule init"
    exit 1
fi
echo "✅ CUTLASS headers verified"

# =============================================================================
# Build and Install TurboDiffusion
# =============================================================================
echo ""
echo "Building TurboDiffusion..."
echo "   Compiling CUDA kernels for: sm_80, sm_89, sm_90, sm_120a (Blackwell)"
echo "   This may take several minutes..."
echo ""

# Clean previous builds if requested
if [ "$1" == "--clean" ]; then
    echo "Cleaning previous builds..."
    rm -rf build/ dist/ *.egg-info/
    find . -name "*.so" -path "*/turbodiffusion/*" -delete 2>/dev/null || true
fi

pip install -e . --no-build-isolation 2>&1 | tee build.log

# =============================================================================
# Create Module Symlinks (for inference scripts)
# =============================================================================
echo ""
echo "Creating module symlinks..."

# The inference scripts import from top-level (e.g., 'from imaginaire.utils.io')
# but modules are inside turbodiffusion/. Create symlinks at repo root.
cd "$SCRIPT_DIR"

for module in imaginaire rcm ops SLA; do
    if [ -d "turbodiffusion/$module" ]; then
        if [ ! -L "$module" ]; then
            ln -sf "turbodiffusion/$module" "$module"
            echo "   Created symlink: $module -> turbodiffusion/$module"
        else
            echo "   Symlink exists: $module"
        fi
    fi
done

# Verify symlinks work
python -c "
import sys
sys.path.insert(0, '.')
from imaginaire.utils.io import save_image_or_video
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from ops import FastLayerNorm, FastRMSNorm, Int8Linear
from SLA import SparseLinearAttention, SageSparseLinearAttention
print('✅ All module imports working')
" || echo "⚠️  Some imports failed - check symlinks"

# =============================================================================
# Install SpargeAttn (Sparse Attention for efficiency)
# =============================================================================
echo ""
echo "Installing SpargeAttn..."

# Get GPU compute capability
GPU_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "8.0")
echo "   Detected GPU compute capability: $GPU_ARCH"

# Clone, patch for Blackwell (sm_120) if needed, and install
SPARGE_TMP="/tmp/SpargeAttn_build_$$"
rm -rf "$SPARGE_TMP"
git clone --depth 1 https://github.com/thu-ml/SpargeAttn.git "$SPARGE_TMP"

# Add sm_120 (Blackwell) support if not already present
if grep -q '"12.0"' "$SPARGE_TMP/setup.py"; then
    echo "   SpargeAttn already supports sm_120"
else
    echo "   Patching SpargeAttn for Blackwell (sm_120) support..."
    sed -i 's/SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0"}/SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0", "12.0"}/' "$SPARGE_TMP/setup.py"
fi

cd "$SPARGE_TMP"
TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install -e . --no-build-isolation
cd "$SCRIPT_DIR"
rm -rf "$SPARGE_TMP"

# =============================================================================
# Verify Installation
# =============================================================================
echo ""
echo "Verifying installation..."

python -c "
import torch
import turbo_diffusion_ops
print('✅ turbo_diffusion_ops loaded')
print('   Available ops:', [x for x in dir(turbo_diffusion_ops) if not x.startswith('_')])

try:
    import spas_sage_attn
    print('✅ SpargeAttn (spas_sage_attn) loaded')
except ImportError:
    print('⚠️  SpargeAttn not available (optional)')

print()
print('GPU Info:')
if torch.cuda.is_available():
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'   Compute Capability: {torch.cuda.get_device_capability(0)}')
"

echo ""
echo "=============================================="
echo "✅ Installation complete!"
echo "=============================================="
echo ""
echo "Usage:"
echo "  conda activate $ENV_NAME"
echo "  python -c 'import turbodiffusion'"
echo ""
echo "To run the TUI server:"
echo "  python -m turbodiffusion.tui_serve"
echo ""

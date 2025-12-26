#!/bin/bash
# ComfyUI with TurboDiffusion startup script
#
# Usage:
#   ./comfyui-turbo.sh              # Start ComfyUI
#   ./comfyui-turbo.sh --stop       # Stop ComfyUI
#   ./comfyui-turbo.sh --status     # Check status
#
# Setup for boot:
#   crontab -e
#   @reboot /path/to/comfyui-turbo.sh

# ============================================================================
# Configuration - Edit these paths for your system
# ============================================================================
CONDA_PATH="$HOME/miniconda3"
CONDA_ENV="turbodiffusion"
COMFYUI_PATH="/media/2TB/ComfyUI"
CUDA_PATH="/usr/local/cuda-13.0"
LOG_FILE="/tmp/comfyui_turbo.log"
PORT=8188

# ============================================================================
# Functions
# ============================================================================

start_comfyui() {
    # Check if already running
    if pgrep -f "python.*main.py.*$PORT" > /dev/null; then
        echo "ComfyUI already running on port $PORT"
        exit 1
    fi

    # Source conda
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        source "$CONDA_PATH/etc/profile.d/conda.sh"
    else
        echo "Error: Conda not found at $CONDA_PATH"
        exit 1
    fi

    # Activate environment
    conda activate "$CONDA_ENV"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate conda environment '$CONDA_ENV'"
        exit 1
    fi

    # Set CUDA path
    if [ -d "$CUDA_PATH" ]; then
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    fi

    # Change to ComfyUI directory
    if [ ! -d "$COMFYUI_PATH" ]; then
        echo "Error: ComfyUI not found at $COMFYUI_PATH"
        exit 1
    fi
    cd "$COMFYUI_PATH"

    # Start ComfyUI with nohup
    echo "============================================" >> "$LOG_FILE"
    echo "Starting ComfyUI at $(date)" >> "$LOG_FILE"
    echo "Environment: $CONDA_ENV" >> "$LOG_FILE"
    echo "============================================" >> "$LOG_FILE"

    nohup python main.py --listen 0.0.0.0 --port $PORT >> "$LOG_FILE" 2>&1 &

    PID=$!
    echo "ComfyUI started with PID: $PID"
    echo "Port: $PORT"
    echo "Log file: $LOG_FILE"
}

stop_comfyui() {
    if pkill -f "python.*main.py.*$PORT"; then
        echo "ComfyUI stopped"
    else
        echo "ComfyUI not running"
    fi
}

check_status() {
    if pgrep -f "python.*main.py.*$PORT" > /dev/null; then
        echo "ComfyUI is running on port $PORT"
        # Try to get GPU info
        curl -s "http://127.0.0.1:$PORT/system_stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'  GPU: {d[\"devices\"][0][\"name\"]}')
    print(f'  VRAM: {d[\"devices\"][0][\"vram_total\"]/1024**3:.1f} GB')
except:
    pass
" 2>/dev/null
    else
        echo "ComfyUI is not running"
    fi
}

# ============================================================================
# Main
# ============================================================================

case "${1:-start}" in
    --stop|-s)
        stop_comfyui
        ;;
    --status|-t)
        check_status
        ;;
    --help|-h)
        echo "Usage: $0 [--stop|--status|--help]"
        echo "  (default)  Start ComfyUI"
        echo "  --stop     Stop ComfyUI"
        echo "  --status   Check if running"
        ;;
    *)
        start_comfyui
        ;;
esac

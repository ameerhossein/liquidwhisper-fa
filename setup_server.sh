#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Setting up Persian ASR Environment on GPU Server"
echo "=========================================================="

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: ${PYTHON_VERSION}"

# 2. Create virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Upgrade pip, wheel, setuptools
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with CUDA 12 support
echo "Installing PyTorch with CUDA support..."
pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Install project dependencies
echo "Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

# 5b. Check telephony codecs (Iran MCI/Irancell/TCI phone simulation)
echo "Checking ffmpeg telephony codecs..."
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg found: $(ffmpeg -version 2>/dev/null | head -n 1)"
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libopencore_amrnb; then
        echo "AMR-NB encoder (mobile) available."
    else
        echo "WARNING: ffmpeg lacks libopencore_amrnb; mobile sim falls back to lite approximation."
        echo "  On Ubuntu: sudo apt-get install -y ffmpeg libopencore-amrnb0"
    fi
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q pcm_alaw; then
        echo "G.711 A-law encoder (PSTN) available."
    else
        echo "WARNING: ffmpeg lacks pcm_alaw; PSTN sim falls back to lite approximation."
    fi
else
    echo "WARNING: ffmpeg not found; phone simulation uses lite scipy-only fallback."
    echo "  Install with: sudo apt-get update && sudo apt-get install -y ffmpeg"
    echo "  (training still works; true AMR-NB/G.711 requires ffmpeg on the L40S server)"
fi

# 6. Verify GPU and Environment
echo "=========================================================="
echo " Verifying GPU Acceleration & BF16 Support"
echo "=========================================================="
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
    print('Device Capability:', torch.cuda.get_device_capability(0))
    print('Native BF16 Support:', torch.cuda.is_bf16_supported())
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Total VRAM: {vram_gb:.2f} GB')
else:
    print('WARNING: No CUDA GPU detected!')
"

echo "=========================================================="
echo " Setup complete! Activate the environment with:"
echo "   source .venv/bin/activate"
echo "=========================================================="

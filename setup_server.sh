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

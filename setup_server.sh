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

# 4b. Install torchcodec pinned to match torch 2.6 (cu124).
# torch/CUDA (datasets>=3/4 requires it to decode the "audio" column).
# Unpinned installs pull 0.17 which needs torch>=2.11 / CUDA 13
# (OSError: libnvrtc.so.13) while cu124 caps torch at 2.6.
# Pairing: torchcodec 0.5.x <-> torch 2.6. Bump together for cu128+.
echo "Installing torchcodec 0.5.0 to match torch 2.6 (cu124)..."
pip uninstall -y torchcodec 2>/dev/null || true
pip install "torchcodec==0.5.0" --index-url https://download.pytorch.org/whl/cu124
python3 -c "import torchcodec; print('torchcodec OK:', torchcodec.__version__)" || {
    echo "WARNING: torchcodec import failed; audio decoding will fail."
    echo "  Retry with: pip install 'torchcodec==0.5.0' --index-url https://download.pytorch.org/whl/cu124"
}

# 5. Install project dependencies
echo "Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

# 5b. Check telephony codecs (Iran MCI/Irancell/TCI phone simulation)
echo "Checking ffmpeg telephony codecs..."
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg found: $(ffmpeg -version 2>/dev/null | head -n 1)"
    if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libopencore_amrnb; then
        echo "ffmpeg lacks libopencore_amrnb; installing true static ffmpeg..."
        if command -v apt-get &> /dev/null; then
            (sudo apt-get update -qq && sudo apt-get install -y -qq libopencore-amrnb0 libopencore-amrwb0 ffmpeg xz-utils curl 2>&1 | tail -n 3) || \
            (apt-get update -qq && apt-get install -y -qq libopencore-amrnb0 libopencore-amrwb0 ffmpeg xz-utils curl 2>&1 | tail -n 3) || true
        fi
    fi
    # Ubuntu apt ffmpeg is often built without AMR-NB; static johnvansickle
    # builds ship libopencore_amrnb + pcm_alaw. Install to /usr/local/bin
    # so it shadows /usr/bin/ffmpeg on PATH.
    if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libopencore_amrnb; then
        TMPD=$(mktemp -d)
        if (curl -fsSL -o "$TMPD/ffmpeg.tar.xz" https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz || \
            wget -qO "$TMPD/ffmpeg.tar.xz" https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz); then
            tar -xf "$TMPD/ffmpeg.tar.xz" -C "$TMPD"
            STATIC_FFMPEG=$(echo "$TMPD"/ffmpeg-*-amd64-static/ffmpeg | head -n 1)
            if [ -f "$STATIC_FFMPEG" ]; then
                cp -f "$STATIC_FFMPEG" /usr/local/bin/ffmpeg
                chmod +x /usr/local/bin/ffmpeg
                hash -r 2>/dev/null || true
                echo "static ffmpeg installed: $(/usr/local/bin/ffmpeg -version 2>/dev/null | head -n 1)"
            fi
        else
            echo "WARNING: static ffmpeg download failed; keeping system ffmpeg."
        fi
        rm -rf "$TMPD"
    fi
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libopencore_amrnb; then
        echo "AMR-NB encoder (mobile) available."
    else
        echo "WARNING: ffmpeg still lacks libopencore_amrnb; mobile sim falls back to lite approximation."
        echo "  Manual fix: curl -fsSL -o /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        echo "    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp && cp -f /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg"
        echo "  Training still works; src/telephone.py uses alaw_quantize fallback."
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

#!/bin/bash

set -euo pipefail

# Detect OS and Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected OS: $OS"
echo "Detected Arch: $ARCH"

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo ""
    echo "To install uv, run one of the following:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  brew install uv"
    echo "  pip install uv"
    echo ""
    echo "For more options, visit: https://docs.astral.sh/uv/installation/"
    exit 1
fi

UV_EXTRAS=()

install_ffmpeg_macos() {
    if command -v ffmpeg &> /dev/null; then
        return
    fi
    if ! command -v brew &> /dev/null; then
        echo "Homebrew is required to install ffmpeg on macOS."
        exit 1
    fi
    echo "Installing ffmpeg via Homebrew..."
    brew install ffmpeg
}

install_ffmpeg_ubuntu() {
    if command -v ffmpeg &> /dev/null; then
        return
    fi
    if ! command -v apt-get &> /dev/null; then
        echo "Ubuntu setup expects apt-get for ffmpeg installation."
        exit 1
    fi
    echo "Installing ffmpeg via apt-get..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
}

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "Configuring Apple Silicon environment with MLX-Audio."
    install_ffmpeg_macos
    UV_EXTRAS+=("mlx")
elif [[ "$OS" == "Linux" ]]; then
    echo "Configuring Ubuntu/Linux environment."
    install_ffmpeg_ubuntu
    UV_EXTRAS+=("cuda")
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Adding cuda-compat libraries."
        UV_EXTRAS+=("cuda-compat")
        HAS_CUDA=true
        # flash-attn skipped — difficult to build on Blackwell/aarch64
    else
        echo "No NVIDIA GPU detected. CUDA endpoint path is still in development."
    fi
else
    echo "Unsupported platform: $OS $ARCH"
    exit 1
fi

HAS_CUDA=${HAS_CUDA:-false}

uv venv
SYNC_ARGS=()
for extra in "${UV_EXTRAS[@]}"; do
    SYNC_ARGS+=(--extra "$extra")
done
uv sync "${SYNC_ARGS[@]}"
if [[ "$HAS_CUDA" == true ]]; then
    # PyPI default torch on aarch64 is CPU-only; overwrite with CUDA torch from nightly index
    echo "Installing PyTorch with CUDA support..."
    uv pip install --reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

cat <<EOF

Setup complete.

----------------------------------------------------------------
Run: ./run.sh                             # Starts server on port 4003 (prod)
     ./run.sh --dev                       # Starts with development env
     uv run kortexa-tts [--dev|--prod]    # Starts server manually
----------------------------------------------------------------

Notes:
- macOS Apple Silicon is the primary supported runtime today and uses MLX-Audio.
- Linux/CUDA setup is scaffolded, but the OpenAI-style endpoint path there is still in development.
- ffmpeg is required for MP3/AAC/Opus output.
EOF

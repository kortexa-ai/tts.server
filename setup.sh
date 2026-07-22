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
    if command -v ffmpeg &> /dev/null && command -v sox &> /dev/null; then
        return
    fi
    if ! command -v brew &> /dev/null; then
        echo "Homebrew is required to install ffmpeg and sox on macOS."
        exit 1
    fi
    if ! command -v ffmpeg &> /dev/null; then
        echo "Installing ffmpeg via Homebrew..."
        brew install ffmpeg
    fi
    if ! command -v sox &> /dev/null; then
        echo "Installing sox via Homebrew..."
        brew install sox
    fi
}

install_ffmpeg_ubuntu() {
    local PACKAGES=()
    if ! command -v ffmpeg &> /dev/null; then
        PACKAGES+=(ffmpeg)
    fi
    if ! command -v sox &> /dev/null; then
        PACKAGES+=(sox libsox-fmt-all)
    fi
    if [[ ${#PACKAGES[@]} -eq 0 ]]; then
        return
    fi
    if ! command -v apt-get &> /dev/null; then
        echo "Ubuntu setup expects apt-get for ffmpeg and sox installation."
        exit 1
    fi
    echo "Installing missing audio tools via apt-get: ${PACKAGES[*]}"
    sudo apt-get update
    sudo apt-get install -y "${PACKAGES[@]}"
}

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "Configuring Apple Silicon environment with MLX-Audio."
    install_ffmpeg_macos
    UV_EXTRAS+=("mlx")
elif [[ "$OS" == "Linux" ]]; then
    echo "Configuring Ubuntu/Linux environment."
    install_ffmpeg_ubuntu
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected."
        UV_EXTRAS+=("cuda")
        HAS_CUDA=true
        # flash-attn skipped — difficult to build on Blackwell/aarch64
    else
        echo "No NVIDIA GPU detected. The Linux backend requires an NVIDIA GPU."
        exit 1
    fi
else
    echo "Unsupported platform: $OS $ARCH"
    exit 1
fi

HAS_CUDA=${HAS_CUDA:-false}

uv venv --clear
SYNC_ARGS=()
for extra in "${UV_EXTRAS[@]}"; do
    SYNC_ARGS+=(--extra "$extra")
done
uv sync --locked "${SYNC_ARGS[@]}"
if [[ "$HAS_CUDA" == true ]]; then
    echo "Verifying the locked PyTorch CUDA environment..."
    .venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch installed, but CUDA is not available")

print(f"PyTorch {torch.__version__} / CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
fi

cat <<EOF

Setup complete.

----------------------------------------------------------------
Run: ./run.sh                             # Starts server on port 4003 (prod)
     ./run.sh --dev                       # Starts with development env
     uv run kortexa-tts [--dev|--prod]    # Starts server manually
----------------------------------------------------------------

Notes:
- macOS Apple Silicon uses MLX-Audio backend.
- Linux/CUDA uses qwen-tts backend (streaming falls back to single-chunk delivery).
- ffmpeg is required for MP3/AAC/Opus output.
EOF

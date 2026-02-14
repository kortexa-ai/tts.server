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

# Platform-specific setup
if [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected on Linux - will install with flash-attn support"
        export UV_EXTRA="flash"
    else
        echo "No NVIDIA GPU detected on Linux - installing CPU version"
    fi
else
    echo "macOS detected - installing Metal/CPU version (no CUDA, no flash-attn)"
    # sox is needed by qwen-tts on macOS
    if ! command -v sox &> /dev/null; then
        echo "Installing sox via Homebrew..."
        brew install sox
    fi
fi

uv venv
if [[ -n "${UV_EXTRA:-}" ]]; then
    uv sync --extra "$UV_EXTRA"
else
    uv sync
fi

cat <<EOF

Note: Qwen3-TTS model weights will be downloaded automatically on first use.
To pre-download, run: uv run python -c "from qwen_tts import Qwen3TTSModel; Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign')"

Setup complete!
----------------------------------------------------------------
Run: ./run.sh                             # Starts server on port 4003 (prod)
     ./run.sh --dev                       # Starts with development env
     uv run kortexa-tts [--dev|--prod]    # Starts server manually
----------------------------------------------------------------
EOF

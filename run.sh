#!/bin/bash

set -euo pipefail

# Ensure we're in the repo root
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Ensure uv exists early for clearer failures under systemd
if ! command -v uv >/dev/null 2>&1; then
  echo "[tts.server] UV_MISSING" >&2
  exit 2
fi

# Detect OS for logging
OS="$(uname -s)"
echo "Running on OS: $OS"

# Defaults (allow override via environment)
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-4003}
TTS_ENV=${TTS_ENV:-production}
TTS_MODEL_ID=${TTS_MODEL_ID:-qwen3-tts-customvoice-1.7b}
# Platform-aware default model repo
if [[ "$OS" == "Darwin" ]]; then
    TTS_MODEL_REPO=${TTS_MODEL_REPO:-mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16}
else
    TTS_MODEL_REPO=${TTS_MODEL_REPO:-Qwen/Qwen3-TTS-12Hz-1.7B}
fi

PASS_MODE_FLAG=""
RELOAD_FLAG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      TTS_ENV=development
      PASS_MODE_FLAG="--dev"
      RELOAD_FLAG="--reload"
      shift
      ;;
    --prod)
      TTS_ENV=production
      PASS_MODE_FLAG="--prod"
      shift
      ;;
    --model-id)
      TTS_MODEL_ID="$2"
      shift 2
      ;;
    --model-repo)
      TTS_MODEL_REPO="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS="$EXTRA_ARGS $1"
      shift
      ;;
  esac
done

echo "Starting Kortexa TTS server ($TTS_ENV)..."
echo "Model ID: $TTS_MODEL_ID"
echo "Model repo: $TTS_MODEL_REPO"

# Use .venv/bin directly to avoid uv run re-resolving and downgrading CUDA torch to CPU
exec .venv/bin/kortexa-tts --host "$HOST" --port "$PORT" --model-id "$TTS_MODEL_ID" --model-repo "$TTS_MODEL_REPO" $PASS_MODE_FLAG $RELOAD_FLAG $EXTRA_ARGS

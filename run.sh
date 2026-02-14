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
MODEL=${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}

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
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS="$EXTRA_ARGS $1"
      shift
      ;;
  esac
done

echo "Starting Kortexa TTS server ($TTS_ENV)..."
echo "Model: $MODEL"

# We don't quote EXTRA_ARGS to allow word splitting of multiple arguments
uv run kortexa-tts --host "$HOST" --port "$PORT" --model "$MODEL" $PASS_MODE_FLAG $RELOAD_FLAG $EXTRA_ARGS

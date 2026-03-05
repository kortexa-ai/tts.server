#!/usr/bin/env bash
# Start the Voice Designer server and React client
set -e
cd "$(dirname "$0")"

trap 'kill 0' EXIT

uv run python scripts/voice_designer.py &
cd client && npm run dev &

wait

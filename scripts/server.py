#!/usr/bin/env python3
"""
Kortexa TTS Server

FastAPI server providing REST API for text-to-speech with Qwen3-TTS VoiceDesign.
"""

import os
import sys
import copy
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kortexa.tts.dotenv_helper import resolve_env_file, load_env_file
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG
import uvicorn


def main():
    """Main entry point for the Kortexa TTS server."""

    parser = argparse.ArgumentParser(description="Kortexa TTS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=4003, help="Port to listen to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--root-path", default="", help="ASGI root path (e.g. /tts behind proxy)")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", help="TTS model to use")
    parser.add_argument(
        "--max-concurrent-inference",
        type=int,
        default=None,
        help="Max concurrent inferences (default: 1)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dev", action="store_true", help="Use development env resolution")
    group.add_argument("--prod", action="store_true", help="Use production env resolution")
    args = parser.parse_args()

    # Resolve and load env before honoring PORT
    mode = "development" if args.dev else "production"
    env_file = resolve_env_file(mode)
    if env_file and os.path.isfile(env_file):
        load_env_file(env_file)

    # If PORT is set in env and user didn't explicitly pass --port, honor env
    if "PORT" in os.environ and "--port" not in sys.argv:
        try:
            args.port = int(os.environ["PORT"])
        except Exception:
            pass

    print("kortexa.ai TTS")
    print(f"Environment: {mode}")
    if env_file:
        print(f"Env file: {env_file}")
    else:
        print("Env file: (none)")
    print(f"Using model: {args.model}")
    if args.max_concurrent_inference is not None:
        os.environ["TTS_MAX_CONCURRENT_INFERENCE"] = str(max(1, args.max_concurrent_inference))
    max_concurrency = os.environ.get("TTS_MAX_CONCURRENT_INFERENCE", "1")
    print(f"Max concurrent inference: {max_concurrency}")
    url = f"http://{args.host}:{args.port}{args.root_path}"
    print(f"Url: {url}")
    print("Press Ctrl+C to stop the server\n")

    # Extend uvicorn logging config
    log_config = copy.deepcopy(UVICORN_LOGGING_CONFIG)
    loggers = log_config.setdefault("loggers", {})
    # Only attach handler to the root namespace; child loggers propagate up
    loggers["kortexa"] = {"handlers": ["default"], "level": "INFO", "propagate": False}

    # Set model in environment for the app factory to pick up
    os.environ["TTS_MODEL"] = args.model

    uvicorn.run(
        "kortexa.tts.server:app",
        host=args.host,
        port=args.port,
        root_path=args.root_path,
        log_level="info",
        reload=args.reload,
        log_config=log_config,
        factory=False,
    )


if __name__ == "__main__":
    main()

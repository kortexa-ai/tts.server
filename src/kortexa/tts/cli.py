from __future__ import annotations

import argparse
import copy
import os
import sys

import uvicorn
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

from .dotenv_helper import load_env_file, resolve_env_file
from .service import DEFAULT_MODEL_ID, DEFAULT_MODEL_REPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Kortexa TTS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=4003, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--root-path",
        default="",
        help="ASGI root path when mounted behind a proxy",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Public model id exposed on /v1/models",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Underlying model repo to load",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dev", action="store_true", help="Use development env resolution")
    group.add_argument("--prod", action="store_true", help="Use production env resolution")
    args = parser.parse_args()

    mode = "development" if args.dev else "production"
    env_file = resolve_env_file(mode)
    if env_file and os.path.isfile(env_file):
        load_env_file(env_file)

    if "HOST" in os.environ and "--host" not in sys.argv:
        args.host = os.environ["HOST"]
    if "PORT" in os.environ and "--port" not in sys.argv:
        try:
            args.port = int(os.environ["PORT"])
        except ValueError:
            pass
    if "TTS_MODEL_ID" in os.environ and "--model-id" not in sys.argv:
        args.model_id = os.environ["TTS_MODEL_ID"]
    if "TTS_MODEL_REPO" in os.environ and "--model-repo" not in sys.argv:
        args.model_repo = os.environ["TTS_MODEL_REPO"]

    print("kortexa.ai TTS")
    print(f"Environment: {mode}")
    print(f"Env file: {env_file or '(none)'}")
    print(f"Public model id: {args.model_id}")
    print(f"Model repo: {args.model_repo}")
    print(f"Url: http://{args.host}:{args.port}{args.root_path}")
    print("Press Ctrl+C to stop the server\n")

    log_config = copy.deepcopy(UVICORN_LOGGING_CONFIG)
    loggers = log_config.setdefault("loggers", {})
    loggers["kortexa"] = {"handlers": ["default"], "level": "INFO", "propagate": False}

    os.environ["TTS_MODEL_ID"] = args.model_id
    os.environ["TTS_MODEL_REPO"] = args.model_repo

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

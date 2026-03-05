from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .service import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REPO,
    STREAMING_RESPONSE_FORMAT,
    TTSService,
)

logger = logging.getLogger("kortexa.tts")


class VoiceReference(BaseModel):
    id: str


class SpeechRequest(BaseModel):
    model: str = Field(..., description="Model id from GET /v1/models")
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str | VoiceReference
    instructions: str | None = Field(default=None, max_length=4096)
    response_format: Literal["mp3", "wav", "flac", "pcm", "aac", "opus"] | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Literal["audio", "sse"] | None = None


def error_payload(message: str, error_type: str) -> dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def create_app(
    root_path: str | None = None,
    model_id: str | None = None,
    model_repo: str | None = None,
) -> FastAPI:
    if model_id is None:
        model_id = os.environ.get("TTS_MODEL_ID", DEFAULT_MODEL_ID)
    if model_repo is None:
        model_repo = os.environ.get("TTS_MODEL_REPO", DEFAULT_MODEL_REPO)

    for name in ("kortexa", "kortexa.tts", "kortexa.tts.service"):
        logging.getLogger(name).setLevel(logging.INFO)

    tts_service = TTSService(model_id=model_id, model_repo=model_repo)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading TTS service (model=%s repo=%s)", model_id, model_repo)
        tts_service.load_model()
        app.state.tts_service = tts_service
        yield
        tts_service.unload_model()

    app = FastAPI(
        title="Kortexa TTS Server",
        description="OpenAI-compatible text-to-speech API backed by MLX-Audio.",
        root_path=root_path or "",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        error_type = "server_error" if exc.status_code >= 500 else "invalid_request_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(detail, error_type),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError):
        message = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        return JSONResponse(
            status_code=400,
            content=error_payload(message, "invalid_request_error"),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content=error_payload(str(exc), "invalid_request_error"),
        )

    @app.exception_handler(RuntimeError)
    async def runtime_exception_handler(_request: Request, exc: RuntimeError):
        return JSONResponse(
            status_code=503,
            content=error_payload(str(exc), "service_unavailable"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=error_payload(str(exc), "server_error"),
        )

    @app.get("/", response_class=JSONResponse)
    async def index():
        return {
            "name": app.title,
            "version": "2.0.0",
            "endpoints": [
                "GET /health",
                "GET /v1/models",
                "GET /v1/voices",
                "POST /v1/voices/reload",
                "POST /v1/audio/speech",
            ],
        }

    @app.get("/health", response_class=JSONResponse)
    async def health():
        svc: TTSService = app.state.tts_service
        return svc.health()

    @app.get("/v1/models", response_class=JSONResponse)
    async def list_models():
        svc: TTSService = app.state.tts_service
        return {"object": "list", "data": svc.list_models()}

    @app.get("/v1/voices", response_class=JSONResponse)
    async def list_voices():
        svc: TTSService = app.state.tts_service
        svc.ensure_ready()
        return {
            "object": "list",
            "data": svc.list_voices(),
            "default_voice": svc.default_voice.id if svc.default_voice else None,
        }

    @app.post("/v1/voices/reload", response_class=JSONResponse)
    async def reload_voices():
        svc: TTSService = app.state.tts_service
        svc.ensure_ready()
        svc.reload_custom_voices()
        return {
            "status": "ok",
            "voice_count": len(svc.supported_voices),
            "custom_count": sum(1 for v in svc.supported_voices if v.is_custom),
        }

    @app.post("/v1/audio/speech")
    async def create_speech(payload: SpeechRequest):
        svc: TTSService = app.state.tts_service
        text = payload.input.strip()
        if not text:
            raise HTTPException(status_code=400, detail="`input` cannot be blank")

        svc.ensure_model(payload.model)
        voice = svc.resolve_voice(payload.voice)

        response_format = payload.response_format
        if response_format is None:
            response_format = (
                STREAMING_RESPONSE_FORMAT if payload.stream_format else "mp3"
            )

        if payload.stream_format and response_format != STREAMING_RESPONSE_FORMAT:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Streaming currently supports "
                    f"`response_format=\"{STREAMING_RESPONSE_FORMAT}\"` only."
                ),
            )

        if payload.stream_format == "audio":
            return StreamingResponse(
                svc.stream_audio_bytes(
                    text=text,
                    voice=voice,
                    instructions=payload.instructions or "",
                    speed=payload.speed,
                    response_format=response_format,
                ),
                media_type=svc.media_type_for_format(response_format),
                headers={
                    "Content-Disposition": f'attachment; filename="speech.{response_format}"',
                    "x-model-id": svc.model_id,
                    "x-voice-id": voice.id,
                    "x-sample-rate": str(svc.sample_rate),
                },
            )

        if payload.stream_format == "sse":
            return StreamingResponse(
                svc.stream_sse(
                    text=text,
                    voice=voice,
                    instructions=payload.instructions or "",
                    speed=payload.speed,
                    response_format=response_format,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "x-model-id": svc.model_id,
                    "x-voice-id": voice.id,
                    "x-sample-rate": str(svc.sample_rate),
                },
            )

        audio, sample_rate = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: svc.synthesize(
                text=text,
                voice=voice,
                instructions=payload.instructions or "",
                speed=payload.speed,
            ),
        )
        body = svc.encode_audio(audio, response_format)
        return Response(
            content=body,
            media_type=svc.media_type_for_format(response_format),
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format}"',
                "x-model-id": svc.model_id,
                "x-voice-id": voice.id,
                "x-sample-rate": str(sample_rate),
            },
        )

    return app


app = create_app()

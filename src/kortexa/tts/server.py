from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable, Iterator, Literal

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

_STREAM_COMPLETE = object()
_STREAM_QUEUE_SIZE = 2
_STREAM_QUEUE_POLL_SECONDS = 0.1


def _next_stream_item(iterator: Iterator[bytes | str]) -> bytes | str | object:
    return next(iterator, _STREAM_COMPLETE)


async def _disconnect_safe_stream(
    content: Iterable[bytes | str],
    inference_semaphore: asyncio.Semaphore,
    background_tasks: set[asyncio.Task[None]],
) -> AsyncIterator[bytes | str]:
    """Keep model iteration owned until a disconnected response can close it.

    This prevents the synchronous generator from retaining the inference lock.
    """
    queue: asyncio.Queue[bytes | str | Exception | object] = asyncio.Queue(
        maxsize=_STREAM_QUEUE_SIZE,
    )
    abandoned = asyncio.Event()

    async def offer(item: bytes | str | Exception | object) -> bool:
        while not abandoned.is_set():
            try:
                await asyncio.wait_for(
                    queue.put(item),
                    timeout=_STREAM_QUEUE_POLL_SECONDS,
                )
                return True
            except TimeoutError:
                continue
        return False

    async def produce() -> None:
        iterator = iter(content)
        try:
            async with inference_semaphore:
                while not abandoned.is_set():
                    item = await asyncio.to_thread(_next_stream_item, iterator)
                    if item is _STREAM_COMPLETE:
                        break
                    if not await offer(item):
                        break
        except Exception as exc:
            if not abandoned.is_set():
                await offer(exc)
            else:
                logger.warning(
                    "TTS inference failed while cleaning up a disconnected stream",
                    exc_info=True,
                )
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                await asyncio.to_thread(close)
            await offer(_STREAM_COMPLETE)

    producer = asyncio.create_task(produce(), name="tts-stream-producer")
    background_tasks.add(producer)
    producer.add_done_callback(background_tasks.discard)

    try:
        while True:
            item = await queue.get()
            if item is _STREAM_COMPLETE:
                return
            if isinstance(item, Exception):
                raise item
            if not isinstance(item, (bytes, str)):
                raise TypeError("Unexpected TTS stream item")
            yield item
    finally:
        # StreamingResponse can cancel iteration while a synchronous model call
        # is still running in a worker thread. The producer remains alive long
        # enough to regain ownership, close the iterator, and release
        # inference.
        abandoned.set()


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


def error_payload(message: str, error_type: str) -> dict[str, Any]:
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
    # Async semaphore gates access to inference so requests queue in the
    # event loop instead of blocking thread-pool workers on the inference lock.
    inference_semaphore = asyncio.Semaphore(1)
    inference_tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Loading TTS service (model=%s repo=%s)", model_id, model_repo)
        tts_service.load_model()
        app.state.tts_service = tts_service
        app.state.inference_semaphore = inference_semaphore
        app.state.inference_tasks = inference_tasks
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
    async def http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        error_type = (
            "server_error" if exc.status_code >= 500 else "invalid_request_error"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(detail, error_type),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        return JSONResponse(
            status_code=400,
            content=error_payload(message, "invalid_request_error"),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_payload(str(exc), "invalid_request_error"),
        )

    @app.exception_handler(RuntimeError)
    async def runtime_exception_handler(
        _request: Request, exc: RuntimeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=error_payload(str(exc), "service_unavailable"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=error_payload(str(exc), "server_error"),
        )

    @app.get("/", response_class=JSONResponse)
    async def index() -> dict[str, Any]:
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
    async def health() -> dict[str, Any]:
        svc: TTSService = app.state.tts_service
        return svc.health()

    @app.get("/v1/models", response_class=JSONResponse)
    async def list_models() -> dict[str, Any]:
        svc: TTSService = app.state.tts_service
        return {"object": "list", "data": svc.list_models()}

    @app.get("/v1/voices", response_class=JSONResponse)
    async def list_voices() -> dict[str, Any]:
        svc: TTSService = app.state.tts_service
        svc.ensure_ready()
        return {
            "object": "list",
            "data": svc.list_voices(),
            "default_voice": svc.default_voice.id if svc.default_voice else None,
        }

    @app.post("/v1/voices/reload", response_class=JSONResponse)
    async def reload_voices() -> dict[str, Any]:
        svc: TTSService = app.state.tts_service
        svc.ensure_ready()
        svc.reload_custom_voices()
        return {
            "status": "ok",
            "voice_count": len(svc.supported_voices),
            "custom_count": sum(1 for v in svc.supported_voices if v.is_custom),
        }

    @app.post("/v1/audio/speech")
    async def create_speech(payload: SpeechRequest) -> Response:
        svc: TTSService = app.state.tts_service
        text = payload.input.strip()
        if not text:
            raise HTTPException(status_code=400, detail="`input` cannot be blank")

        svc.ensure_model(payload.model)
        voice_input = (
            payload.voice.model_dump()
            if isinstance(payload.voice, VoiceReference)
            else payload.voice
        )
        voice = svc.resolve_voice(voice_input)

        response_format: str = payload.response_format or (
            STREAMING_RESPONSE_FORMAT if payload.stream_format else "mp3"
        )

        if payload.stream_format and response_format != STREAMING_RESPONSE_FORMAT:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Streaming currently supports "
                    f'`response_format="{STREAMING_RESPONSE_FORMAT}"` only.'
                ),
            )

        # Raw PCM has no container to finalise, so chunking it is always safe:
        # a client that reads the whole body still gets the whole body, and one
        # that reads incrementally gets audio while it is still being made.
        # Streaming it only on an explicit `stream_format` meant OpenAI SDK
        # clients — which send `response_format="pcm"` and nothing else — sat
        # through the entire synthesis before their first byte. Measured
        # locally: ~1000ms to first chunk that way, ~25ms this way.
        wants_audio_stream = payload.stream_format == "audio" or (
            payload.stream_format is None
            and response_format == STREAMING_RESPONSE_FORMAT
        )

        if wants_audio_stream:
            return StreamingResponse(
                _disconnect_safe_stream(
                    svc.stream_audio_bytes(
                        text=text,
                        voice=voice,
                        instructions=payload.instructions or "",
                        speed=payload.speed,
                        response_format=response_format,
                    ),
                    app.state.inference_semaphore,
                    app.state.inference_tasks,
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
                _disconnect_safe_stream(
                    svc.stream_sse(
                        text=text,
                        voice=voice,
                        instructions=payload.instructions or "",
                        speed=payload.speed,
                        response_format=response_format,
                    ),
                    app.state.inference_semaphore,
                    app.state.inference_tasks,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "x-model-id": svc.model_id,
                    "x-voice-id": voice.id,
                    "x-sample-rate": str(svc.sample_rate),
                },
            )

        async with app.state.inference_semaphore:
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

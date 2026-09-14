from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import aclosing, asynccontextmanager
from threading import Event
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

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
    background_tasks: set[asyncio.Task[Any]],
) -> AsyncIterator[bytes | str]:
    """Keep model iteration owned until a disconnected response can close it.

    This prevents the synchronous generator from retaining the inference lock.
    """
    queue: asyncio.Queue[bytes | str | Exception | object] = asyncio.Queue(
        maxsize=_STREAM_QUEUE_SIZE,
    )
    abandoned = threading.Event()
    loop = asyncio.get_running_loop()

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

    def produce_sync() -> None:
        # A model generator can hold thread-local inference contexts across
        # yields. Iterate and close it on one worker, rather than dispatching
        # each next() to a potentially different thread-pool worker.
        def send(item: bytes | str | Exception | object) -> bool:
            return asyncio.run_coroutine_threadsafe(offer(item), loop).result()

        try:
            iterator = iter(content)
            try:
                while not abandoned.is_set():
                    item = _next_stream_item(iterator)
                    if item is _STREAM_COMPLETE:
                        break
                    if not send(item):
                        break
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        except Exception as exc:
            if not abandoned.is_set():
                send(exc)
            else:
                logger.warning(
                    "TTS inference failed while cleaning up a disconnected stream",
                    exc_info=True,
                )
        finally:
            send(_STREAM_COMPLETE)

    async def produce() -> None:
        async with inference_semaphore:
            if not abandoned.is_set():
                await asyncio.to_thread(produce_sync)

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


async def _disconnect_safe_inference(
    request: Request,
    synthesize: Callable[[], tuple[Any, int]],
    inference_semaphore: asyncio.Semaphore,
    background_tasks: set[asyncio.Task[Any]],
) -> tuple[Any, int]:
    """Cancel queued work, but keep running model work owned after its caller leaves."""
    started = False
    abandoned = Event()

    async def watch_disconnect() -> None:
        # FastAPI has consumed the request body before the endpoint calls us.
        while (await request.receive())["type"] != "http.disconnect":
            pass
        abandoned.set()

    def synthesize_if_current() -> tuple[Any, int]:
        # Admission can precede worker availability. Check again on the worker
        # so a request abandoned in the executor queue never starts inference.
        if abandoned.is_set():
            raise asyncio.CancelledError
        return synthesize()

    async def produce() -> tuple[Any, int]:
        nonlocal started
        async with inference_semaphore:
            if abandoned.is_set():
                raise asyncio.CancelledError
            # No suspension between admission and marking ownership. Only queued
            # tasks may be cancelled: cancelling to_thread does not stop inference.
            started = True
            return await asyncio.to_thread(synthesize_if_current)

    def settled(task: asyncio.Task[tuple[Any, int]]) -> None:
        background_tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None and abandoned.is_set():
                logger.warning(
                    "TTS inference failed after its request ended", exc_info=error
                )

    disconnected = asyncio.create_task(watch_disconnect(), name="tts-disconnect")
    producer = asyncio.create_task(produce(), name="tts-inference")
    background_tasks.add(producer)
    producer.add_done_callback(settled)
    try:
        done, _ = await asyncio.wait(
            (producer, disconnected),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnected in done:
            await disconnected
            raise HTTPException(status_code=499, detail="Client disconnected")
        return producer.result()
    finally:
        abandoned.set()
        if not started:
            producer.cancel()
        disconnected.cancel()
        await asyncio.gather(disconnected, return_exceptions=True)


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


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    model: str | None = None
    voice: str | VoiceReference | None = None
    instructions: str | None = Field(default=None, max_length=4096)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


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
    inference_tasks: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Loading TTS service (model=%s repo=%s)", model_id, model_repo)
        tts_service.load_model()
        app.state.tts_service = tts_service
        app.state.inference_semaphore = inference_semaphore
        app.state.inference_tasks = inference_tasks
        try:
            yield
        finally:
            # Disconnected handlers can leave service-owned inference finishing
            # off-loop. The model must outlive those workers and stream iterators.
            while inference_tasks:
                draining = tuple(inference_tasks)
                await asyncio.gather(
                    *(asyncio.shield(task) for task in draining),
                    return_exceptions=True,
                )
                # Gathering already-completed tasks need not yield to their
                # queued discard callbacks. Remove the drained snapshot here
                # so shutdown cannot spin and starve those callbacks forever.
                inference_tasks.difference_update(draining)
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
                "POST /generate",
                "WS /ws",
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

    async def _create_speech(payload: SpeechRequest, request: Request) -> Response:
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

        audio, sample_rate = await _disconnect_safe_inference(
            request,
            lambda: svc.synthesize(
                text=text,
                voice=voice,
                instructions=payload.instructions or "",
                speed=payload.speed,
            ),
            app.state.inference_semaphore,
            app.state.inference_tasks,
        )
        # Whole-file codecs can wait for ffmpeg or compress a long waveform.
        # Keep that CPU work off the HTTP loop so PCM streams keep flowing.
        body = await asyncio.to_thread(svc.encode_audio, audio, response_format)
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

    @app.post("/v1/audio/speech")
    async def create_speech(payload: SpeechRequest, request: Request) -> Response:
        return await _create_speech(payload, request)

    def facade_payload(payload: GenerateRequest, *, streaming: bool) -> SpeechRequest:
        svc: TTSService = app.state.tts_service
        svc.ensure_ready()
        default_voice = svc.default_voice
        selected_voice = payload.voice
        if selected_voice is None:
            if default_voice is None:
                raise HTTPException(status_code=503, detail="No TTS voice is available")
            selected_voice = default_voice.id
        return SpeechRequest(
            model=payload.model or svc.model_id,
            input=payload.prompt,
            voice=selected_voice,
            instructions=payload.instructions,
            speed=payload.speed,
            response_format="pcm" if streaming else "mp3",
            stream_format="audio" if streaming else None,
        )

    @app.post("/generate")
    async def generate(payload: GenerateRequest, request: Request) -> Response:
        """Simple media facade: prompt in, MP3 out."""
        return await _create_speech(facade_payload(payload, streaming=False), request)

    @app.websocket("/ws")
    async def websocket_voice(websocket: WebSocket) -> None:
        """Stream PCM16 chunks for JSON ``generate`` requests."""
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                if not isinstance(message, dict):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Expected a JSON object",
                        }
                    )
                    continue
                if message.get("type", "generate") != "generate":
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Expected a generate message",
                        }
                    )
                    continue
                try:
                    payload = GenerateRequest.model_validate(message)
                    request = facade_payload(payload, streaming=True)
                    svc: TTSService = app.state.tts_service
                    svc.ensure_model(request.model)
                    if not request.input.strip():
                        raise ValueError("`prompt` cannot be blank")
                    voice_input = (
                        request.voice.model_dump()
                        if isinstance(request.voice, VoiceReference)
                        else request.voice
                    )
                    voice = svc.resolve_voice(voice_input)
                    await websocket.send_json(
                        {
                            "type": "start",
                            "format": "pcm_s16le",
                            "sample_rate": svc.sample_rate,
                            "channels": 1,
                            "model": svc.model_id,
                            "voice": voice.id,
                        }
                    )
                    async with aclosing(
                        _disconnect_safe_stream(
                            svc.stream_audio_bytes(
                                text=request.input.strip(),
                                voice=voice,
                                instructions=request.instructions or "",
                                speed=request.speed,
                                response_format="pcm",
                            ),
                            app.state.inference_semaphore,
                            app.state.inference_tasks,
                        )
                    ) as chunks:
                        async for chunk in chunks:
                            if isinstance(chunk, str):
                                chunk = chunk.encode("utf-8")
                            await websocket.send_bytes(chunk)
                    await websocket.send_json({"type": "done"})
                except (
                    HTTPException,
                    RequestValidationError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            return

    return app


app = create_app()

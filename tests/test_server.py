import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect

import kortexa.tts.server as server


class FakeService:
    model_id = "tts-test"
    sample_rate = 24000
    default_voice = SimpleNamespace(id="mira")
    supported_voices = [default_voice]

    def __init__(self, **kwargs):
        self.inference_thread = None
        self.encoding_thread = None

    def load_model(self):
        pass

    def unload_model(self):
        pass

    def ensure_ready(self):
        pass

    def ensure_model(self, model):
        if model != self.model_id:
            raise ValueError("Unsupported model")

    def resolve_voice(self, voice):
        return self.default_voice

    def synthesize(self, **kwargs):
        self.inference_thread = threading.get_ident()
        return b"samples", self.sample_rate

    def encode_audio(self, audio, response_format):
        self.encoding_thread = threading.get_ident()
        return b"encoded"

    def media_type_for_format(self, response_format):
        return "audio/mpeg"


def endpoint(app, path):
    return next(route.endpoint for route in app.routes if route.path == path)


def connected_request():
    async def receive():
        await asyncio.Future()

    return SimpleNamespace(receive=receive)


@pytest.mark.asyncio
async def test_shutdown_removes_completed_tasks_without_waiting_for_callbacks(
    monkeypatch,
):
    monkeypatch.setattr(server, "TTSService", FakeService)
    app = server.create_app()
    async with app.router.lifespan_context(app):
        completed = asyncio.create_task(asyncio.sleep(0))
        await completed
        # Model a completed producer whose discard callback is still queued.
        app.state.inference_tasks.add(completed)
    assert not app.state.inference_tasks


@pytest.mark.asyncio
async def test_buffered_encoding_runs_off_event_loop(monkeypatch):
    monkeypatch.setattr(server, "TTSService", FakeService)
    app = server.create_app()
    async with app.router.lifespan_context(app):
        response = await endpoint(app, "/generate")(
            server.GenerateRequest(prompt="hello"), connected_request()
        )
        assert response.body == b"encoded"
        svc = app.state.tts_service
        assert svc.inference_thread != threading.get_ident()
        assert svc.encoding_thread != threading.get_ident()


@pytest.mark.asyncio
async def test_cancelled_buffered_request_keeps_inference_owned(monkeypatch):
    started = threading.Event()
    finish = threading.Event()

    class BlockingService(FakeService):
        def synthesize(self, **kwargs):
            started.set()
            assert finish.wait(2)
            return super().synthesize(**kwargs)

    monkeypatch.setattr(server, "TTSService", BlockingService)
    app = server.create_app()
    async with app.router.lifespan_context(app):
        task = asyncio.create_task(
            endpoint(app, "/generate")(
                server.GenerateRequest(prompt="hello"), connected_request()
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert app.state.inference_semaphore.locked()
            assert app.state.inference_tasks
        finally:
            finish.set()
            await asyncio.wait_for(
                asyncio.gather(*app.state.inference_tasks, return_exceptions=True), 2
            )
        assert not app.state.inference_semaphore.locked()


@pytest.mark.asyncio
async def test_cancelled_queued_request_does_not_start_inference(monkeypatch):
    monkeypatch.setattr(server, "TTSService", FakeService)
    app = server.create_app()
    async with app.router.lifespan_context(app):
        await app.state.inference_semaphore.acquire()
        task = asyncio.create_task(
            endpoint(app, "/generate")(
                server.GenerateRequest(prompt="hello"), connected_request()
            )
        )
        try:
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            app.state.inference_semaphore.release()
        if app.state.inference_tasks:
            await asyncio.wait_for(
                asyncio.gather(*app.state.inference_tasks, return_exceptions=True), 2
            )
        assert app.state.tts_service.inference_thread is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        {"prompt": "   "},
        {"prompt": "hello", "model": "wrong-model"},
    ],
)
async def test_websocket_rejects_invalid_generation_before_start(monkeypatch, message):
    class Socket:
        sent = []

        async def accept(self):
            pass

        async def receive_json(self):
            if self.sent:
                raise WebSocketDisconnect()
            return message

        async def send_json(self, value):
            self.sent.append(value)

    monkeypatch.setattr(server, "TTSService", FakeService)
    app = server.create_app()
    async with app.router.lifespan_context(app):
        socket = Socket()
        await endpoint(app, "/ws")(socket)
        assert len(socket.sent) == 1
        assert socket.sent[0]["type"] == "error"

"""Real request cancellation must not leave abandoned inference ahead of live PCM."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import socket
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol

from kortexa.tts.server import create_app
from kortexa.tts.service import TTSService


@asynccontextmanager
async def runtime(monkeypatch, *, inference_error=False):
    loop = asyncio.get_running_loop()
    state = SimpleNamespace(
        started=asyncio.Event(),
        finished=asyncio.Event(),
        release=threading.Event(),
        admissions=asyncio.Queue(),
        order=[],
        calls=[],
        encoded=[],
        closed={},
    )

    def load(service):
        service.model = object()
        service.backend = "qwen-tts"
        service._set_supported_voices(["aiden"])

    def unload(service):
        state.order.append("unload")
        service.model = None

    def infer(service, *, text, voice, instructions):
        state.calls.append(text)
        if text == "running":
            loop.call_soon_threadsafe(state.started.set)
            try:
                assert state.release.wait(
                    timeout=10
                ), "Fixture did not release inference"
                state.order.append("inference finished")
                assert service.model is not None, "Model was unloaded during inference"
                if inference_error:
                    raise RuntimeError("synthetic inference failure")
            finally:
                loop.call_soon_threadsafe(state.finished.set)
        return np.zeros(2400, dtype=np.float32)

    def encode(service, audio, response_format):
        state.encoded.append(response_format)
        return b"\x01\x00" * len(audio)

    class AdmissionSemaphore(asyncio.Semaphore):
        async def acquire(self):
            state.admissions.put_nowait(None)
            return await super().acquire()

    monkeypatch.setattr(TTSService, "load_model", load)
    monkeypatch.setattr(TTSService, "unload_model", unload)
    monkeypatch.setattr(TTSService, "_synthesize_cuda", infer)
    monkeypatch.setattr(TTSService, "encode_audio", encode)
    state.app = create_app()
    async with state.app.router.lifespan_context(state.app):
        state.service = state.app.state.tts_service
        state.app.state.inference_semaphore = AdmissionSemaphore(1)
        yield state


def payload(state, text, endpoint="/v1/audio/speech", response_format="wav"):
    if endpoint == "/generate":
        return {"prompt": text}
    return {
        "model": state.service.model_id,
        "voice": "aiden",
        "input": text,
        "response_format": response_format,
    }


async def admitted(state):
    await asyncio.wait_for(state.admissions.get(), 3)


async def cancel(request):
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request


async def settle(state, requests):
    state.release.set()
    await asyncio.wait_for(asyncio.gather(*requests, return_exceptions=True), 3)
    if state.started.is_set():
        await asyncio.wait_for(state.finished.wait(), 3)
    if state.app.state.inference_tasks:
        await asyncio.wait_for(
            asyncio.gather(*state.app.state.inference_tasks, return_exceptions=True),
            3,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/audio/speech", "/generate"])
async def test_cancelled_handlers_keep_inference_owned_and_skip_queued_work(
    monkeypatch, endpoint
):
    async with runtime(monkeypatch) as state:
        transport = httpx.ASGITransport(app=state.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            requests = []
            try:
                first = asyncio.create_task(
                    client.post(endpoint, json=payload(state, "running", endpoint))
                )
                requests.append(first)
                await admitted(state)
                await asyncio.wait_for(state.started.wait(), 3)
                await cancel(first)
                owned_after_cancel = state.app.state.inference_semaphore.locked()
                second = asyncio.create_task(
                    client.post(endpoint, json=payload(state, "abandoned", endpoint))
                )
                requests.append(second)
                await admitted(state)
                await cancel(second)
                pcm = asyncio.create_task(
                    client.post(
                        "/v1/audio/speech",
                        json=payload(state, "live", response_format="pcm"),
                    )
                )
                requests.append(pcm)
                await admitted(state)
                state.release.set()
                response = await asyncio.wait_for(pcm, 3)
                assert (
                    response.status_code == 200
                    and response.content == b"\x01\x00" * 2400
                )
                assert (
                    owned_after_cancel
                ), "Cancelled await released admission before its worker finished"
                assert state.calls == ["running", "live"]
                assert state.encoded == ["pcm"]
            finally:
                await settle(state, requests)


@asynccontextmanager
async def loopback_server(state):
    class DisconnectObserver(H11Protocol):
        def connection_lost(self, exc):
            super().connection_lost(exc)
            name = dict((self.scope or {}).get("headers", [])).get(b"x-fixture")
            if name in state.closed:
                state.closed[name].set()

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    sock.setblocking(False)
    server = uvicorn.Server(
        uvicorn.Config(
            state.app,
            http=DisconnectObserver,
            lifespan="off",
            ws="none",
            log_level="critical",
            access_log=False,
        )
    )
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 3)
        sock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/audio/speech", "/generate"])
async def test_real_http_disconnect_skips_abandoned_jobs_before_live_pcm(
    monkeypatch, endpoint
):
    async with runtime(monkeypatch) as state, loopback_server(state) as base_url:
        async with httpx.AsyncClient(base_url=base_url, timeout=5) as client:
            requests = []
            try:
                for text in ("running", "abandoned"):
                    name = text.encode()
                    state.closed[name] = asyncio.Event()
                    request = asyncio.create_task(
                        client.post(
                            endpoint,
                            headers={"x-fixture": text},
                            json=payload(state, text, endpoint),
                        )
                    )
                    requests.append(request)
                    await admitted(state)
                    if text == "running":
                        await asyncio.wait_for(state.started.wait(), 3)
                    await cancel(request)
                    # Observe the real TCP close, without consuming ASGI receive on the app's behalf.
                    await asyncio.wait_for(state.closed[name].wait(), 3)
                pcm = asyncio.create_task(
                    client.post(
                        "/v1/audio/speech",
                        json=payload(state, "live", response_format="pcm"),
                    )
                )
                requests.append(pcm)
                await admitted(state)
                health = await asyncio.wait_for(client.get("/health"), 3)
                assert health.status_code == 200 and health.json()["ready"]
                state.release.set()
                response = await asyncio.wait_for(pcm, 3)
                assert (
                    response.status_code == 200
                    and response.content == b"\x01\x00" * 2400
                )
                assert state.calls == ["running", "live"]
                assert state.encoded == ["pcm"]
            finally:
                await settle(state, requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("inference_error", [False, True])
async def test_shutdown_drains_cancelled_inference_before_unloading(
    monkeypatch, inference_error
):
    context = runtime(monkeypatch, inference_error=inference_error)
    state = await context.__aenter__()
    closing = None
    request = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=state.app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(
                client.post("/generate", json={"prompt": "running"})
            )
            await asyncio.wait_for(state.started.wait(), 3)
            await cancel(request)
            closing_entered = asyncio.Event()

            async def close():
                closing_entered.set()
                await context.__aexit__(None, None, None)

            closing = asyncio.create_task(close())
            await asyncio.wait_for(closing_entered.wait(), 3)
            state.release.set()
            await asyncio.wait_for(closing, 3)
            assert state.order == ["inference finished", "unload"]
            assert state.app.state.inference_tasks == set()
    finally:
        state.release.set()
        if state.started.is_set():
            await asyncio.wait_for(state.finished.wait(), 3)
        if request is not None:
            await asyncio.gather(request, return_exceptions=True)
        if closing is None:
            await context.__aexit__(None, None, None)
        else:
            await asyncio.gather(closing, return_exceptions=True)


@pytest.mark.asyncio
async def test_abandoned_inference_does_not_start_when_the_executor_is_busy(
    monkeypatch,
):
    loop = asyncio.get_running_loop()
    release = threading.Event()
    submitted = asyncio.Event()
    run_in_executor = loop.run_in_executor
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = run_in_executor(executor, release.wait, 10)

        def submit(pool, function, *args):
            future = run_in_executor(
                executor if pool is None else pool, function, *args
            )
            if pool is None:
                submitted.set()
            return future

        monkeypatch.setattr(loop, "run_in_executor", submit)
        async with runtime(monkeypatch) as state:
            requests = []
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=state.app), base_url="http://test"
                ) as client:
                    request = asyncio.create_task(
                        client.post("/generate", json={"prompt": "abandoned"})
                    )
                    requests.append(request)
                    await asyncio.wait_for(submitted.wait(), 3)
                    await cancel(request)
                    release.set()
                    await blocker
                    await settle(state, requests)
                    assert (
                        state.calls == []
                    ), "Cancelled job started inference after waiting for a worker"
                    assert state.encoded == []
            finally:
                release.set()
                await blocker
                await settle(state, requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected_status",
    [(ValueError("bad inference"), 400), (RuntimeError("model failed"), 503)],
)
async def test_connected_inference_errors_keep_the_http_contract(
    monkeypatch, error, expected_status
):
    async with runtime(monkeypatch) as state:

        def fail(**kwargs):
            raise error

        monkeypatch.setattr(state.service, "synthesize", fail)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=state.app), base_url="http://test"
        ) as client:
            response = await client.post("/generate", json={"prompt": "test"})
            assert response.status_code == expected_status
            assert response.json()["error"]["message"] == str(error)
        assert state.app.state.inference_tasks == set()

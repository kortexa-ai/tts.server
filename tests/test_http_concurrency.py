"""Real HTTP routes stay responsive while a whole-file encoder is busy."""

import asyncio
import threading

import httpx
import numpy as np
import pytest

from kortexa.tts.server import create_app
from kortexa.tts.service import TTSService


@pytest.fixture
def app(monkeypatch):
    # Keep real request validation, routing, metadata and error middleware.
    # No test loads a GPU model; only inference is replaced with fixed PCM.
    app = create_app()
    service = TTSService()
    service.model = object()
    service.backend = "qwen-tts"
    service._set_supported_voices(["aiden"])
    monkeypatch.setattr(service, "synthesize", lambda **kwargs: (np.zeros(2400), 24000))
    monkeypatch.setattr(
        service, "stream_audio_bytes", lambda **kwargs: iter([b"\x01\x00" * 480])
    )
    app.state.tts_service = service
    app.state.inference_semaphore = asyncio.Semaphore(1)
    app.state.inference_tasks = set()
    return app


def speech_payload(app, response_format="mp3"):
    return {
        "model": app.state.tts_service.model_id,
        "voice": "aiden",
        "input": "Test speech.",
        "response_format": response_format,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/audio/speech", "/generate"])
async def test_busy_encoder_does_not_block_health_or_pcm(app, monkeypatch, endpoint):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    released, finished = threading.Event(), threading.Event()
    encoder_threads = []

    def encode(audio, response_format):
        encoder_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        try:
            # Finite fail-safe lets the regression fail rather than hang the suite.
            released.wait(timeout=2)
            return b"encoded-file"
        finally:
            finished.set()

    monkeypatch.setattr(app.state.tts_service, "encode_audio", encode)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = (
            {"prompt": "Test speech."}
            if endpoint == "/generate"
            else speech_payload(app)
        )
        encoding = asyncio.create_task(client.post(endpoint, json=payload))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert not finished.is_set(), "The encoder blocked the HTTP event loop"
            health = await asyncio.wait_for(client.get("/health"), 1)
            pcm = await asyncio.wait_for(
                client.post("/v1/audio/speech", json=speech_payload(app, "pcm")), 1
            )
            assert health.status_code == 200 and health.json()["ready"]
            assert pcm.status_code == 200 and pcm.content == b"\x01\x00" * 480
            assert not finished.is_set(), "Concurrent endpoints waited for encoding"
            assert len(encoder_threads) == 1
            assert encoder_threads[0] != threading.get_ident()
        finally:
            released.set()
            response = await asyncio.wait_for(encoding, 3)
            if app.state.inference_tasks:
                await asyncio.gather(*app.state.inference_tasks)
        assert response.status_code == 200
        assert response.content == b"encoded-file"
        assert response.headers["x-sample-rate"] == "24000"
        assert response.headers["x-voice-id"] == "aiden"


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", ["mp3", "wav", "flac", "aac", "opus"])
async def test_encoding_keeps_format_and_metadata(app, monkeypatch, response_format):
    def encode(audio, format):
        assert format == response_format
        np.testing.assert_array_equal(audio, np.zeros(2400))
        return b"encoded-file"

    monkeypatch.setattr(app.state.tts_service, "encode_audio", encode)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/audio/speech", json=speech_payload(app, response_format)
        )
    assert response.status_code == 200 and response.content == b"encoded-file"
    assert response.headers["x-model-id"] == app.state.tts_service.model_id
    assert response.headers["content-disposition"].endswith(
        f'"speech.{response_format}"'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,status",
    [(ValueError("bad encoding"), 400), (RuntimeError("encoder unavailable"), 503)],
)
async def test_encoding_errors_keep_the_http_contract(
    app, monkeypatch, failure, status
):
    def encode(*args):
        raise failure

    monkeypatch.setattr(app.state.tts_service, "encode_audio", encode)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/audio/speech", json=speech_payload(app))
    assert response.status_code == status
    assert response.json()["error"]["message"] == str(failure)


@pytest.mark.asyncio
async def test_real_mp3_encoder_still_returns_decodable_audio(app):
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is a documented runtime prerequisite")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/audio/speech", json=speech_payload(app))
    assert response.status_code == 200
    decoded = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "24000",
            "pipe:1",
        ],
        input=response.content,
        capture_output=True,
        check=True,
        timeout=5,
    )
    # Lossy-container padding is allowed, empty or malformed audio is not.
    assert len(decoded.stdout) >= 2400 * 2


@pytest.mark.asyncio
async def test_cancelled_encoding_does_not_stop_pcm_or_abandon_its_input(
    app, monkeypatch
):
    loop = asyncio.get_running_loop()
    entered, finished = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    errors = []

    def encode(audio, format):
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(timeout=3)
            np.testing.assert_array_equal(audio, np.zeros(2400))
            return b"discarded-file"
        except BaseException as exc:
            errors.append(exc)
            raise
        finally:
            loop.call_soon_threadsafe(finished.set)

    monkeypatch.setattr(app.state.tts_service, "encode_audio", encode)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(client.post("/generate", json={"prompt": "Test"}))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert not finished.is_set()
            pcm = await asyncio.wait_for(
                client.post("/v1/audio/speech", json=speech_payload(app, "pcm")), 1
            )
            assert pcm.status_code == 200 and pcm.content == b"\x01\x00" * 480
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), 2)
            await asyncio.gather(request, return_exceptions=True)
            if app.state.inference_tasks:
                await asyncio.gather(*app.state.inference_tasks)
    assert not errors

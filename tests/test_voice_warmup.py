import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kortexa.tts import service


class FakeModel:
    def __init__(self, outcomes=None, graph_error=False):
        self.outcomes = outcomes or {}
        self.graph_error = graph_error
        self.events = []
        # Retain SDK generators: closing must not depend on garbage collection.
        self.streams = []

    def warmup(self):
        self.events.append("graphs")
        if self.graph_error:
            raise RuntimeError("synthetic graph failure")

    def get_supported_speakers(self):
        return ["Builtin"]

    def get_supported_languages(self):
        return ["auto", "english"]

    def generate_voice_clone_streaming(self, **kwargs):
        assert kwargs["chunk_size"] == 8
        assert kwargs["language"] == "auto"
        assert kwargs["xvec_only"] is True
        assert kwargs["text"].strip()
        voice = Path(kwargs["ref_audio"]).stem.lower()

        def generate():
            self.events.append(f"start:{voice}")
            try:
                outcome = self.outcomes.get(voice, "audio")
                if outcome == "error":
                    raise RuntimeError("synthetic voice failure")
                if outcome == "exhausted":
                    return
                chunk = np.ones(8, dtype=np.float32)
                if outcome == "empty":
                    chunk = chunk[:0]
                yield (chunk, 24_000) if outcome == "tuple" else chunk
                pytest.fail("Warmup must consume at most one chunk per voice")
            finally:
                self.events.append(f"close:{voice}")

        stream = generate()
        self.streams.append(stream)
        return stream


def load_fake(monkeypatch, tmp_path, model, names, *, cuda=False, cleanup_error=False):
    for name in names:
        (tmp_path / f"{name}.wav").touch()
    monkeypatch.setattr(service, "VOICES_DIR", tmp_path)

    def empty_cache():
        model.events.append("empty_cache")
        if cleanup_error:
            raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda, empty_cache=empty_cache)
        ),
    )

    def from_pretrained(repo, *, device):
        assert repo == service.DEFAULT_MODEL_REPO_CUDA
        assert device == ("cuda:0" if cuda else "cpu")
        return model

    monkeypatch.setitem(
        sys.modules,
        "faster_qwen3_tts",
        SimpleNamespace(
            FasterQwen3TTS=SimpleNamespace(from_pretrained=from_pretrained)
        ),
    )
    tts = service.TTSService(model_repo=service.DEFAULT_MODEL_REPO_CUDA)
    assert tts._load_model_cuda_faster()
    assert tts.ready
    assert tts.model is model
    assert tts.sample_rate == 24_000
    assert tts.supported_languages == ["auto", "english"]
    return tts


def test_loader_warms_custom_voices_serially_and_closes_after_first_chunk(
    monkeypatch, tmp_path
):
    model = FakeModel({"second": "tuple"})
    tts = load_fake(monkeypatch, tmp_path, model, ["Second", "First"])
    assert model.events == [
        "graphs",
        "start:first",
        "close:first",
        "start:second",
        "close:second",
    ]
    assert [voice.id for voice in tts.supported_voices] == [
        "builtin",
        "first",
        "second",
    ]
    assert tts.default_voice.id == "builtin"


@pytest.mark.parametrize("outcome", ["error", "empty", "exhausted"])
def test_failed_voice_warmup_does_not_disable_other_voices(
    monkeypatch, tmp_path, caplog, outcome
):
    model = FakeModel({"first": outcome})
    with caplog.at_level(logging.WARNING, logger="kortexa.tts.service"):
        tts = load_fake(monkeypatch, tmp_path, model, ["First", "Second"])
    assert model.events == [
        "graphs",
        "start:first",
        "close:first",
        "start:second",
        "close:second",
    ]
    assert [voice.id for voice in tts.supported_voices] == [
        "builtin",
        "first",
        "second",
    ]
    assert "first" in caplog.text
    assert "warmup failed" in caplog.text


def test_loader_without_custom_voices_does_not_attempt_cloning(monkeypatch, tmp_path):
    model = FakeModel()
    load_fake(monkeypatch, tmp_path, model, [])
    assert model.events == ["graphs"]
    assert model.streams == []


@pytest.mark.parametrize("outcome", ["audio", "error"])
def test_cuda_releases_unused_memory_once_after_all_voice_streams_close(
    monkeypatch, tmp_path, outcome
):
    model = FakeModel({"first": outcome})
    load_fake(monkeypatch, tmp_path, model, ["First", "Second"], cuda=True)
    assert model.events == [
        "graphs",
        "start:first",
        "close:first",
        "start:second",
        "close:second",
        "empty_cache",
    ]


def test_cleanup_failure_preserves_the_loaded_model(monkeypatch, tmp_path, caplog):
    model = FakeModel()
    load_fake(monkeypatch, tmp_path, model, ["First"], cuda=True, cleanup_error=True)
    assert model.events == ["graphs", "start:first", "close:first", "empty_cache"]
    assert "Could not release unused warmup CUDA memory" in caplog.text


def test_voice_warmup_still_runs_after_graph_warmup_failure(monkeypatch, tmp_path):
    model = FakeModel(graph_error=True)
    load_fake(monkeypatch, tmp_path, model, ["First"])
    assert model.events == ["graphs", "start:first", "close:first"]


def test_reload_does_not_run_startup_inference(monkeypatch, tmp_path):
    model = FakeModel()
    tts = load_fake(monkeypatch, tmp_path, model, ["First"])
    before = model.events.copy()
    (tmp_path / "Second.wav").touch()
    tts.reload_custom_voices()
    assert model.events == before
    assert [voice.id for voice in tts.supported_voices] == [
        "builtin",
        "first",
        "second",
    ]


def test_normal_stream_still_yields_every_sdk_chunk_and_closes():
    first = np.ones(8, dtype=np.float32)
    second = np.full(8, 0.5, dtype=np.float32)
    closed = []

    def generate():
        try:
            yield first
            yield second, 24_000
        finally:
            closed.append(True)

    sdk_stream = generate()
    tts = service.TTSService()
    tts.model = SimpleNamespace(
        generate_voice_clone_streaming=lambda **kwargs: sdk_stream
    )
    voice = service.VoiceInfo(id="test", name="Test", wav_path="synthetic.wav")
    chunks = list(tts._stream_faster(text="Synthetic speech.", voice=voice))
    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], first)
    np.testing.assert_array_equal(chunks[1], second)
    assert closed == [True]

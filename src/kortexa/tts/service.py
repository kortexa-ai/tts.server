from __future__ import annotations

import base64
import importlib.util
import io
import json
import logging
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

logger = logging.getLogger("kortexa.tts.service")

DEFAULT_MODEL_ID = "qwen3-tts-customvoice-1.7b"
DEFAULT_MODEL_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
DEFAULT_VOICE_ID = "aiden"
SUPPORTED_RESPONSE_FORMATS = ("mp3", "wav", "flac", "pcm", "aac", "opus")
STREAMING_RESPONSE_FORMAT = "pcm"


@dataclass
class VoiceInfo:
    id: str
    name: str


class TTSService:
    """Small MLX-Audio wrapper for the public HTTP API."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_repo: str = DEFAULT_MODEL_REPO,
    ):
        self.model_id = model_id
        self.model_repo = model_repo
        self.backend = "mlx-audio"
        self.sample_rate = 24_000

        self.model = None
        self.mx = None
        self.audio_write = None
        self.load_error: Optional[str] = None

        self.supported_languages: list[str] = []
        self.supported_voices: list[VoiceInfo] = []
        self._voice_lookup: dict[str, VoiceInfo] = {}
        self._inference_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.load_error is None

    @property
    def default_voice(self) -> Optional[VoiceInfo]:
        if not self.supported_voices:
            return None
        preferred = self._voice_lookup.get(DEFAULT_VOICE_ID)
        if preferred is not None:
            return preferred
        return self.supported_voices[0]

    def load_model(self) -> None:
        if self.model is not None:
            return

        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.load_error = (
                "The OpenAI-compatible MLX backend currently supports macOS Apple Silicon "
                "only. Linux/CUDA endpoint parity is still in development."
            )
            logger.warning(self.load_error)
            return

        if importlib.util.find_spec("mlx_audio.tts.models.qwen3_tts") is None:
            self.load_error = (
                "Installed mlx-audio build does not include qwen3_tts. "
                "Install the GitHub version with "
                "`uv pip install --upgrade git+https://github.com/Blaizzy/mlx-audio.git`."
            )
            logger.error(self.load_error)
            return

        try:
            import mlx.core as mx
            from mlx_audio.audio_io import write
            from mlx_audio.tts.utils import load_model

            logger.info("Loading MLX model repo: %s", self.model_repo)
            self.mx = mx
            self.audio_write = write
            self.model = load_model(self.model_repo)
            self.sample_rate = int(getattr(self.model, "sample_rate", self.sample_rate))
            self.supported_languages = list(
                getattr(self.model, "get_supported_languages", lambda: [])() or []
            )
            speakers = list(
                getattr(self.model, "get_supported_speakers", lambda: [])() or []
            )
            self._set_supported_voices(speakers)
            self.load_error = None
            self.mx.clear_cache()
            logger.info(
                "MLX model loaded (voices=%d, languages=%d)",
                len(self.supported_voices),
                len(self.supported_languages),
            )
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)
            logger.exception("Failed to load MLX model")

    def unload_model(self) -> None:
        if self.model is not None:
            logger.info("Unloading MLX model...")
            self.model = None
        self.supported_voices = []
        self.supported_languages = []
        self._voice_lookup = {}
        if self.mx is not None:
            self.mx.clear_cache()

    def _set_supported_voices(self, speakers: list[str]) -> None:
        voices: list[VoiceInfo] = []
        lookup: dict[str, VoiceInfo] = {}
        for speaker in speakers:
            voice = VoiceInfo(id=speaker.strip().lower(), name=speaker.strip())
            voices.append(voice)
            lookup[voice.id] = voice
        self.supported_voices = voices
        self._voice_lookup = lookup

    def ensure_ready(self) -> None:
        if self.ready:
            return
        raise RuntimeError(self.load_error or "TTS model is not ready")

    def ensure_model(self, requested_model: str) -> None:
        if requested_model != self.model_id:
            raise ValueError(
                f"Unsupported model '{requested_model}'. Use '{self.model_id}'."
            )

    def resolve_voice(self, voice: str | dict[str, Any]) -> VoiceInfo:
        self.ensure_ready()

        if hasattr(voice, "id"):
            voice_id = str(getattr(voice, "id", "")).strip().lower()
        elif isinstance(voice, dict):
            voice_id = str(voice.get("id", "")).strip().lower()
        else:
            voice_id = str(voice).strip().lower()

        if not voice_id:
            raise ValueError("`voice` is required")
        if voice_id not in self._voice_lookup:
            available = [item.id for item in self.supported_voices]
            raise ValueError(f"Unknown voice '{voice_id}'. Available voices: {available}")
        return self._voice_lookup[voice_id]

    def health(self) -> dict[str, Any]:
        default_voice = self.default_voice.id if self.default_voice else None
        return {
            "status": "ok" if self.ready else "error",
            "ready": self.ready,
            "backend": self.backend,
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "model": {
                "id": self.model_id,
                "repo": self.model_repo,
            },
            "sample_rate": self.sample_rate,
            "voice_count": len(self.supported_voices),
            "default_voice": default_voice,
            "load_error": self.load_error,
        }

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": self.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "kortexa",
                "metadata": {
                    "backend": self.backend,
                    "repo": self.model_repo,
                },
            }
        ]

    def list_voices(self) -> list[dict[str, Any]]:
        default_voice = self.default_voice.id if self.default_voice else None
        return [
            {
                "id": voice.id,
                "object": "voice",
                "name": voice.name,
                "model": self.model_id,
                "default": voice.id == default_voice,
                "languages": self.supported_languages,
            }
            for voice in self.supported_voices
        ]

    def _collect_audio(self, results: list[Any]) -> np.ndarray:
        chunks = [self._to_numpy(result.audio) for result in results]
        if not chunks:
            return np.array([], dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

    def _to_numpy(self, audio: Any) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32, copy=False)
        if hasattr(audio, "__array__"):
            return np.asarray(audio, dtype=np.float32)
        if hasattr(audio, "tolist"):
            return np.asarray(audio.tolist(), dtype=np.float32)
        return np.array(audio, dtype=np.float32)

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        if speed == 1.0 or audio.size == 0:
            return audio.astype(np.float32, copy=False)

        old_length = int(audio.shape[0])
        new_length = max(1, int(round(old_length / speed)))
        old_positions = np.arange(old_length, dtype=np.float32)
        new_positions = np.linspace(0, old_length - 1, new_length, dtype=np.float32)
        resampled = np.interp(new_positions, old_positions, audio).astype(np.float32)
        return np.clip(resampled, -1.0, 1.0)

    def synthesize(
        self,
        *,
        text: str,
        voice: VoiceInfo,
        instructions: str,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        self.ensure_ready()
        with self._inference_lock:
            results = list(
                self.model.generate_custom_voice(
                    text=text,
                    speaker=voice.name,
                    language="auto",
                    instruct=instructions or None,
                    stream=False,
                )
            )
        audio = self._apply_speed(self._collect_audio(results), speed)
        return audio, self.sample_rate

    def stream_audio(
        self,
        *,
        text: str,
        voice: VoiceInfo,
        instructions: str,
        speed: float,
        streaming_interval: float = 1.0,
    ) -> Iterator[np.ndarray]:
        self.ensure_ready()
        with self._inference_lock:
            for result in self.model.generate_custom_voice(
                text=text,
                speaker=voice.name,
                language="auto",
                instruct=instructions or None,
                stream=True,
                streaming_interval=streaming_interval,
            ):
                yield self._apply_speed(self._to_numpy(result.audio), speed)

    def encode_audio(self, audio: np.ndarray, response_format: str) -> bytes:
        response_format = response_format.lower()
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise ValueError(
                f"Unsupported response_format '{response_format}'. "
                f"Supported formats: {list(SUPPORTED_RESPONSE_FORMATS)}"
            )

        if response_format in {"aac", "opus"}:
            return self._encode_ffmpeg(audio, response_format)

        if self.audio_write is None:
            raise RuntimeError("Audio encoder is not ready")

        buffer = io.BytesIO()
        self.audio_write(buffer, audio, self.sample_rate, format=response_format)
        return buffer.getvalue()

    def stream_audio_bytes(
        self,
        *,
        text: str,
        voice: VoiceInfo,
        instructions: str,
        speed: float,
        response_format: str,
        streaming_interval: float = 1.0,
    ) -> Iterator[bytes]:
        if response_format != STREAMING_RESPONSE_FORMAT:
            raise ValueError(
                "Streaming currently supports `response_format=\"pcm\"` only."
            )

        for chunk in self.stream_audio(
            text=text,
            voice=voice,
            instructions=instructions,
            speed=speed,
            streaming_interval=streaming_interval,
        ):
            yield self.encode_audio(chunk, response_format)

    def stream_sse(
        self,
        *,
        text: str,
        voice: VoiceInfo,
        instructions: str,
        speed: float,
        response_format: str,
        streaming_interval: float = 1.0,
    ) -> Iterator[str]:
        if response_format != STREAMING_RESPONSE_FORMAT:
            raise ValueError(
                "SSE streaming currently supports `response_format=\"pcm\"` only."
            )

        started = time.perf_counter()
        for index, chunk in enumerate(
            self.stream_audio(
                text=text,
                voice=voice,
                instructions=instructions,
                speed=speed,
                streaming_interval=streaming_interval,
            )
        ):
            encoded = self.encode_audio(chunk, response_format)
            payload = {
                "type": "audio.chunk",
                "index": index,
                "audio": base64.b64encode(encoded).decode("ascii"),
                "format": response_format,
                "sample_rate": self.sample_rate,
                "voice": voice.id,
            }
            yield f"event: audio.chunk\ndata: {json.dumps(payload)}\n\n"

        done = {
            "type": "audio.done",
            "format": response_format,
            "sample_rate": self.sample_rate,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
        yield f"event: audio.done\ndata: {json.dumps(done)}\n\n"

    def media_type_for_format(self, response_format: str) -> str:
        response_format = response_format.lower()
        if response_format == "mp3":
            return "audio/mpeg"
        if response_format == "wav":
            return "audio/wav"
        if response_format == "flac":
            return "audio/flac"
        if response_format == "aac":
            return "audio/aac"
        if response_format == "opus":
            return "audio/ogg"
        return "audio/pcm"

    def _encode_ffmpeg(self, audio: np.ndarray, response_format: str) -> bytes:
        ffmpeg = shutil_which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"ffmpeg is required for {response_format} output. Install ffmpeg first."
            )

        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2").tobytes()
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
        ]

        if response_format == "aac":
            cmd.extend(["-c:a", "aac", "-f", "adts", "pipe:1"])
        elif response_format == "opus":
            cmd.extend(["-c:a", "libopus", "-f", "ogg", "pipe:1"])
        else:
            raise ValueError(f"Unsupported ffmpeg response format: {response_format}")

        completed = subprocess.run(cmd, input=pcm, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="ignore").strip())
        return completed.stdout


def shutil_which(binary: str) -> Optional[str]:
    import shutil

    return shutil.which(binary)

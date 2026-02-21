from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi import Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .service import TTSService

logger = logging.getLogger("kortexa.tts")


def create_app(
    root_path: Optional[str] = None,
    model_name: Optional[str] = None,
    max_concurrent_inference: Optional[int] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    if model_name is None:
        model_name = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    if max_concurrent_inference is None:
        try:
            max_concurrent_inference = int(
                os.environ.get("TTS_MAX_CONCURRENT_INFERENCE", "1")
            )
        except ValueError:
            max_concurrent_inference = 1

    # Set up logging
    for name in ("kortexa", "kortexa.tts", "kortexa.tts.service"):
        logging.getLogger(name).setLevel(logging.INFO)

    # Create service
    tts_service = TTSService(
        model_name=model_name,
        max_concurrent_inference=max_concurrent_inference,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: load model
        print(f"Loading TTS model: {model_name}")
        tts_service.load_model()
        app.state.tts_service = tts_service
        print("TTS server started")
        yield
        # Shutdown: clean up
        print("Shutting down, unloading model...")
        tts_service.unload_model()
        print("Model unloaded")

    app = FastAPI(
        title="Kortexa TTS Server",
        description="Text-to-speech with Qwen3-TTS VoiceDesign",
        root_path=root_path or "",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=JSONResponse)
    async def index():
        return {
            "name": app.title,
            "version": "1.0",
            "endpoints": ["GET /health", "POST /tts"],
        }

    @app.get("/health", response_class=JSONResponse)
    async def health():
        svc: TTSService = app.state.tts_service
        return {
            "status": "ok",
            "model_id": svc.model_name,
            "device": svc.device,
            "sample_rate": svc.sample_rate,
        }

    @app.post("/tts")
    async def tts(
        text: str = Body(..., embed=True),
        # voice.server compat: accepted but unused (VoiceDesign doesn't use speaker IDs)
        speaker_id: int = Body(0),
        # VoiceDesign params
        instruct: str = Body("", description="Natural-language voice description"),
        language: str = Body("Auto", description="Language: English, Chinese, Auto, etc."),
        seed: Optional[int] = Body(None, description="Random seed for reproducible voice generation"),
        format: str = Query("pcm16", pattern="^(wav|pcm16)$"),
        chunk_ms: int = Query(200, ge=20, le=1000),
    ):
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text must be non-empty")

        cleaned_text = text.strip()
        svc: TTSService = app.state.tts_service

        # Set seed for reproducible voice generation across calls
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if cleaned_text.lower() == "__tone__":
            # Synthesize a 1kHz sine tone for 2 seconds to aid debugging
            sr = svc.sample_rate
            duration_s = 2.0
            t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
            audio = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        else:
            audio, sr = await svc.synthesize(
                cleaned_text,
                instruct=instruct,
                language=language,
            )

        if len(audio):
            min_val = float(np.min(audio))
            max_val = float(np.max(audio))
        else:
            min_val = max_val = 0.0
        print(f"[tts] generated {len(audio)} samples at {sr} Hz (min={min_val:.4f}, max={max_val:.4f})")

        if format == "pcm16":
            # Convert float32 [-1,1] to little-endian PCM16
            pcm = np.clip(audio, -1.0, 1.0)
            scaled = np.clip(pcm * 32767.0, -32768.0, 32767.0)
            pcm16 = scaled.astype("<i2")
            raw = pcm16.tobytes()
            chunk_bytes = max(1, int(sr * (chunk_ms / 1000.0)) * 2)

            def pcm_stream():
                mv = memoryview(raw)
                for offset in range(0, len(mv), chunk_bytes):
                    yield bytes(mv[offset : offset + chunk_bytes])

            headers = {
                "x-audio-format": "pcm16",
                "x-sample-rate": str(sr),
                "x-channels": "1",
                "x-chunk-ms": str(chunk_ms),
            }
            return StreamingResponse(pcm_stream(), media_type="audio/L16", headers=headers)

        if format == "wav":
            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, audio.astype(np.float32), sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            return StreamingResponse(buf, media_type="audio/wav")

        raise HTTPException(status_code=400, detail="Unsupported format")

    return app


# Default app for uvicorn
app = create_app()

#!/usr/bin/env python3
"""
Voice Designer — standalone FastAPI server for generating and saving custom TTS voices.

Loads the VoiceDesign model (Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16) on port 4010.
Generate sample voices from text descriptions, audition them, and save the ones you like
as .wav files for use with the CustomVoice TTS server.

Usage:
    cd ~/src/tts.server && uv run python scripts/voice_designer.py
"""

from __future__ import annotations

import base64
import io
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("voice_designer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

VOICES_DIR = Path(__file__).parent.parent / "voices"
VOICEDESIGN_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
SAMPLE_RATE = 24_000
PORT = 4010


# ── Request/Response models ──────────────────────────────────────────

class GenerateRequest(BaseModel):
    instruct: str = Field(..., min_length=1, max_length=2048, description="Voice description prompt")
    text: str = Field(..., min_length=1, max_length=2048, description="Sample text to speak")


class GenerateResponse(BaseModel):
    id: str
    audio_b64: str
    sample_rate: int


class SaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    audio_b64: str = Field(..., description="Base64-encoded WAV audio from /generate")


class VoiceEntry(BaseModel):
    name: str
    wav_path: str


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Voice Designer", description="Generate and save custom TTS voices")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference, loaded at startup
_model = None
_mx = None
_audio_write = None


def _load_models():
    global _model, _mx, _audio_write

    import mlx.core as mx
    from mlx_audio.audio_io import write as audio_write
    from mlx_audio.tts.utils import load_model

    _mx = mx
    _audio_write = audio_write

    logger.info("Loading VoiceDesign model: %s", VOICEDESIGN_REPO)
    _model = load_model(VOICEDESIGN_REPO)
    logger.info("VoiceDesign model loaded (sample_rate=%d)", SAMPLE_RATE)


@app.on_event("startup")
async def startup():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    _load_models()


@app.get("/health")
async def health():
    return {"status": "ok" if _model is not None else "error", "model": VOICEDESIGN_REPO}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    start = time.perf_counter()
    gen_id = uuid.uuid4().hex[:12]

    # VoiceDesign model uses generate_voice_design()
    results = list(_model.generate_voice_design(
        text=req.text,
        instruct=req.instruct,
        language="auto",
        stream=False,
    ))

    if not results:
        raise HTTPException(500, "Generation produced no results")

    # Collect audio chunks
    chunks = []
    for r in results:
        audio = r.audio
        if hasattr(audio, "__array__"):
            audio = np.asarray(audio, dtype=np.float32)
        elif not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32, copy=False)
        chunks.append(audio)

    audio_np = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    # Encode as WAV
    buf = io.BytesIO()
    _audio_write(buf, audio_np, SAMPLE_RATE, format="wav")
    wav_bytes = buf.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    elapsed = time.perf_counter() - start
    logger.info("Generated voice sample %s in %.2fs (%.1fs audio)", gen_id, elapsed, len(audio_np) / SAMPLE_RATE)

    return GenerateResponse(id=gen_id, audio_b64=audio_b64, sample_rate=SAMPLE_RATE)


@app.post("/save")
async def save_voice(req: SaveRequest):
    wav_path = VOICES_DIR / f"{req.name}.wav"

    # Decode and save WAV
    try:
        wav_bytes = base64.b64decode(req.audio_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio data")

    wav_path.write_bytes(wav_bytes)
    logger.info("Saved voice '%s' to %s (%d bytes)", req.name, wav_path, len(wav_bytes))

    return {"name": req.name, "wav_path": str(wav_path), "size_bytes": len(wav_bytes)}


@app.get("/voices")
async def list_voices():
    voices = []
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        voices.append(VoiceEntry(name=wav.stem, wav_path=str(wav)))
    return {"voices": voices}


@app.get("/voices/{name}/audio")
async def get_voice_audio(name: str):
    wav_path = VOICES_DIR / f"{name}.wav"
    if not wav_path.exists():
        raise HTTPException(404, f"Voice '{name}' not found")
    return FileResponse(wav_path, media_type="audio/wav", filename=f"{name}.wav")


@app.delete("/voices/{name}")
async def delete_voice(name: str):
    wav_path = VOICES_DIR / f"{name}.wav"
    if not wav_path.exists():
        raise HTTPException(404, f"Voice '{name}' not found")
    wav_path.unlink()
    logger.info("Deleted voice '%s'", name)
    return {"deleted": name}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

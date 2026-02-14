from __future__ import annotations

import logging
import asyncio
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("kortexa.tts.service")


class TTSService:
    """Qwen3-TTS VoiceDesign service for text-to-speech synthesis."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_concurrent_inference: int = 1,
    ):
        self.model_name = model_name

        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Use appropriate dtype for device
        if dtype is None:
            if self.device.startswith("cuda"):
                dtype = torch.bfloat16
            elif self.device == "mps":
                dtype = torch.bfloat16  # qwen-tts uses bfloat16 on MPS
            else:
                dtype = torch.float32
        self.dtype = dtype

        self.max_concurrent_inference = max(1, int(max_concurrent_inference))
        self.sample_rate = 24_000  # Qwen3-TTS outputs 24kHz audio

        self.model = None
        self._inference_gate = asyncio.Semaphore(self.max_concurrent_inference)

    def load_model(self) -> None:
        """Load the Qwen3-TTS VoiceDesign model."""
        if self.model is not None:
            return

        logger.info(f"Loading TTS model: {self.model_name}")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")

        try:
            from qwen_tts import Qwen3TTSModel

            kwargs = dict(
                device_map=self.device,
                dtype=self.dtype,
            )
            # Flash attention only works on CUDA
            if self.device.startswith("cuda"):
                kwargs["attn_implementation"] = "flash_attention_2"

            self.model = Qwen3TTSModel.from_pretrained(self.model_name, **kwargs)
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def unload_model(self) -> None:
        """Unload the model and free GPU memory."""
        if self.model is not None:
            logger.info("Unloading TTS model...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("TTS model unloaded")

    def _synthesize_sync(
        self,
        text: str,
        instruct: str = "",
        language: str = "Auto",
    ) -> Tuple[np.ndarray, int]:
        """Synchronous synthesis (runs on executor thread)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        wavs, sr = self.model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
        )

        # wavs is a list of numpy arrays, we want the first one
        audio = wavs[0] if wavs else np.array([], dtype=np.float32)
        return audio, sr

    async def synthesize(
        self,
        text: str,
        instruct: str = "",
        language: str = "Auto",
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio for given text with voice design instruction.

        Args:
            text: The text to synthesize.
            instruct: Natural-language voice description (timbre, emotion, style).
            language: Language of the text ("English", "Chinese", "Auto", etc.)

        Returns:
            (audio_float32_mono, sample_rate)
        """
        async with self._inference_gate:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._synthesize_sync(text, instruct, language),
            )

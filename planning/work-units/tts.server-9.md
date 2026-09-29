# TTS performance on shock

Owning issue: https://github.com/kortexa-ai/tts.server/issues/9

Measure the existing CUDA TTS service on the idle DGX Spark `shock`. Use the
Linux/CUDA path, the tracked Mira voice sample, and fixed short and long text.
Record cold startup, first-audio latency, warmed audio duration, wall time and
real-time factor. Keep TTS isolated from ASR and vision, then stop it.

The Factory surface preflight selected `alternate-plan`. The active claim covers
this note and `shock:tts.server`.

## Validation

- Before run: shock has no GPU process and no TTS service installed.
- Setup, model startup, benchmark and cleanup: pending.

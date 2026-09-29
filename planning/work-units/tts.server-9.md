# TTS performance on shock

Owning issue: https://github.com/kortexa-ai/tts.server/issues/9

Measure the existing CUDA TTS service on the idle DGX Spark `shock`. Use the
Linux/CUDA path, the tracked Mira voice sample, and fixed short and long text.
Record cold startup, first-audio latency, warmed audio duration, wall time and
real-time factor. Keep TTS isolated from ASR and vision, then stop it.

The Factory surface preflight selected `alternate-plan`. The active claim covered
this note and `shock:tts.server`.

## Validation

- Before run: shock had no GPU process and no TTS service installed.
- Tested revision: `394d3e2a651b4635a68be900fe1189954a210cd4`.
- Setup installed the project's locked dependencies (PyTorch 2.11.0+cu130)
  and system audio packages (ffmpeg, sox, libsox-fmt-all). The server loaded
  `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with the `faster-qwen3` CUDA backend on GB10.
- Cold startup included model download, CUDA graph capture and warmup for four
  reference voices; the checkpoint fetch took about 12 seconds. Startup itself
  was not separately timed.
- Streamed PCM used the tracked `mira` voice at 24 kHz. The short warm request
  returned 2.96 seconds of audio in 2.094 seconds; first audio arrived in 0.516s.
- Four long-passage runs produced 49.52–54.16 seconds of audio in 32.331–35.288s.
  Median audio duration was 52.04s, median wall time 33.885s, and median first
  audio latency 0.439s. Median audio real-time factor was 1.53x.
- Cleanup: sent SIGINT to the exact foreground server PID. Port 4003 is closed,
  no managed TTS unit is installed, and `nvidia-smi` reports no running process.
  The checkout, Python environment, audio packages and downloaded model cache
  remain on shock for a later run.

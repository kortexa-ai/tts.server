# Kortexa TTS Server

OpenAI-compatible text-to-speech server supporting macOS Apple Silicon (`mlx-audio`) and Linux/CUDA (`qwen-tts`).

This project exposes a small public API:

- `GET /health`
- `GET /v1/models`
- `GET /v1/voices`
- `POST /v1/voices/reload`
- `POST /v1/audio/speech`

The server is intentionally small and focused. It currently targets the Qwen3-TTS `CustomVoice` model family on macOS. Linux/CUDA setup is scaffolded, but the endpoint parity there is still in development.

OpenAPI docs are available at:

- `GET /openapi.json`
- `GET /docs`

## Status

| Platform | Backend | Model Repo | Streaming |
|----------|---------|------------|-----------|
| macOS Apple Silicon | `mlx-audio` | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | Native chunked |
| Linux/CUDA | `qwen-tts` | `Qwen/Qwen3-TTS-12Hz-1.7B` | Single-chunk fallback |

Both platforms expose the same OpenAI-compatible API. Custom voices from `voices/*.wav` work on both (MLX uses ref_audio injection, CUDA uses x-vector voice cloning).

## Setup

Run:

```bash
./setup.sh
```

What it does:

- macOS Apple Silicon: installs `ffmpeg`, creates the virtualenv, installs `mlx-audio` from GitHub
- Ubuntu/Linux: installs `ffmpeg`, installs CUDA-side Python deps (`qwen-tts`, PyTorch with CUDA)

`ffmpeg` is required for `mp3`, `aac`, and `opus` output.

## Run

```bash
./run.sh
```

Environment variables:

```bash
PORT=4003
HOST=0.0.0.0
TTS_MODEL_ID=qwen3-tts-customvoice-1.7b
TTS_MODEL_REPO=mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16
```

## Public API

### `GET /health`

Returns process/backend readiness.

Example response:

```json
{
  "status": "ok",
  "ready": true,
  "backend": "mlx-audio",
  "platform": {
    "system": "Darwin",
    "machine": "arm64"
  },
  "model": {
    "id": "qwen3-tts-customvoice-1.7b",
    "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
  },
  "sample_rate": 24000,
  "voice_count": 8,
  "default_voice": "aiden",
  "load_error": null
}
```

### `GET /v1/models`

OpenAI-style model discovery.

Example:

```bash
curl http://127.0.0.1:4003/v1/models
```

### `GET /v1/voices`

Custom discovery endpoint for available voice ids. There is no standard OpenAI endpoint for listing built-in TTS voice ids, so this server exposes them here.

Voice ids are the stable public identifiers clients should store and send back. They are lowercase, case-insensitive on input, and map to the speaker names exposed by the underlying model.

Response shape:

```json
{
  "object": "list",
  "default_voice": "aiden",
  "data": [
    {
      "id": "aiden",
      "object": "voice",
      "name": "Aiden",
      "model": "qwen3-tts-customvoice-1.7b",
      "default": true,
      "languages": ["auto", "english", "japanese"]
    }
  ]
}
```

Use the `id` value in `POST /v1/audio/speech`.

Recommended client flow:

1. `GET /health` and wait for `"ready": true`
2. `GET /v1/models` once and cache the public model id
3. `GET /v1/voices` and let the user pick a voice id
4. `POST /v1/audio/speech`

### `POST /v1/audio/speech`

OpenAI-compatible speech generation endpoint.

Supported request fields:

- `model`: required string from `GET /v1/models`
- `input`: required string, max 4096 chars
- `voice`: required string id or object `{ "id": "aiden" }`
- `instructions`: optional string
- `response_format`: optional `mp3 | wav | flac | pcm | aac | opus`
- `speed`: optional `0.25` to `4.0`
- `stream_format`: optional `audio | sse`

Notes:

- Non-streaming default `response_format` is `mp3`
- Streaming default `response_format` is `pcm`
- Streaming currently supports `response_format="pcm"` only
- Empty or whitespace-only `input` is rejected with `400`

#### Non-streaming example

```bash
curl http://127.0.0.1:4003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-customvoice-1.7b",
    "input": "Hello from Kortexa.",
    "voice": "aiden",
    "response_format": "wav"
  }' \
  --output speech.wav
```

#### Streaming audio example

```bash
curl http://127.0.0.1:4003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-customvoice-1.7b",
    "input": "Hello from Kortexa.",
    "voice": "aiden",
    "stream_format": "audio",
    "response_format": "pcm"
  }' \
  --output speech.pcm
```

#### Streaming SSE example

```bash
curl http://127.0.0.1:4003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-customvoice-1.7b",
    "input": "Hello from Kortexa.",
    "voice": "aiden",
    "stream_format": "sse",
    "response_format": "pcm"
  }'
```

SSE payloads are JSON messages in `data:` frames.

Chunk event:

```json
{
  "type": "audio.chunk",
  "index": 0,
  "audio": "<base64 pcm bytes>",
  "format": "pcm",
  "sample_rate": 24000,
  "voice": "aiden"
}
```

Done event:

```json
{
  "type": "audio.done",
  "format": "pcm",
  "sample_rate": 24000,
  "elapsed_seconds": 3.45
}
```

The server also sets `event: audio.chunk` and `event: audio.done` on SSE frames for clients that want named events.

### Errors

Errors use an OpenAI-style envelope:

```json
{
  "error": {
    "message": "Unknown voice 'robot'. Available voices: ['aiden', 'ryan']",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

Common cases:

- bad request or unsupported voice/model: `400`
- model/backend not ready: `503`
- unexpected server crash: `500`

## Smoke Test

List voices:

```bash
node tests/test.js --list-voices
```

Generate a sample:

```bash
node tests/test.js --voice aiden --format wav
```

Generate a streaming PCM sample:

```bash
node tests/test.js --voice aiden --stream --out tests/output/stream.pcm
```

## OpenAI Compatibility Notes

This server intentionally implements a small subset of the OpenAI speech API shape:

- endpoint path: `POST /v1/audio/speech`
- request fields: `model`, `input`, `voice`, `instructions`, `response_format`, `speed`, `stream_format`

Custom extension:

- `GET /v1/voices` for voice discovery

Reference docs from OpenAI:

- Audio speech endpoint: https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create

## Voice Designer

The Voice Designer is a standalone tool for creating and saving custom TTS voices using the Qwen3-TTS VoiceDesign model. It generates voice samples from text descriptions, lets you audition them, and saves the ones you like as `.wav` files that the main TTS server loads as additional voices.

### Architecture

- **VoiceDesign server** (`scripts/voice_designer.py`) — FastAPI on port 4010, loads `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- **React client** (`client/`) — Vite app with pill-based voice characteristic selectors that auto-generate description prompts
- **Custom voice integration** — saved `.wav` files in `voices/` are loaded at startup as additional voices alongside built-in speakers

### Quick start

```bash
./design.sh
```

This starts both the VoiceDesign server and the React client. Open the client URL shown in the terminal.

### Workflow

1. Select voice characteristics (gender, age, accent, register, etc.) or write a free-form description
2. Click "Generate 3 Samples" — each sample uses the same prompt but produces a different voice
3. Audition samples and save the ones you like (saved as `voices/{name}.wav`)
4. Click "Reload TTS Server" or restart the main server to pick up new voices
5. Use saved voices via the standard API: `"voice": "your-voice-name"`

### Voice Designer API (port 4010)

- `POST /generate` — `{ instruct, text }` → generates audio sample
- `POST /save` — `{ name, audio_b64 }` → saves voice to `voices/{name}.wav`
- `GET /voices` — lists saved voices
- `GET /voices/{name}/audio` — serves saved voice audio
- `DELETE /voices/{name}` — deletes a saved voice
- `GET /health`

### `POST /v1/voices/reload`

Re-scans the `voices/` directory and loads any new custom voices without restarting the server.

```bash
curl -X POST http://127.0.0.1:4003/v1/voices/reload
```

### How custom voices work

The VoiceDesign model generates speech from a text description prompt. Each generation produces a different voice. When you save a voice, the raw audio is stored as a `.wav` file.

At synthesis time, the main TTS server loads the saved `.wav`, passes it through the CustomVoice model's speaker encoder to extract a speaker embedding (~50ms), and uses that embedding for generation. This means custom voices work with the full `instructions` parameter for emotion/style control, just like built-in voices.

Custom voice names are case-insensitive in the API (stored with original case on disk, lowercased for lookup).

## Development Notes

- Both macOS/MLX and Linux/CUDA runtimes are fully wired to the public API
- Streaming on CUDA falls back to single-chunk delivery (qwen-tts does not support chunked generation)
- Custom voices on CUDA use x-vector-only voice cloning (speaker embedding from wav); the `instructions` parameter is not applied for custom voices on CUDA
- `GET /v1/voices` is a project-specific extension because voice discovery is otherwise annoying in exactly the way open source hobby servers should avoid

## License

MIT

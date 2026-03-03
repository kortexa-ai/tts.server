# Kortexa TTS Server

OpenAI-compatible text-to-speech server built for macOS Apple Silicon with `mlx-audio`.

This project exposes a small public API:

- `GET /health`
- `GET /v1/models`
- `GET /v1/voices`
- `POST /v1/audio/speech`

The server is intentionally small and focused. It currently targets the Qwen3-TTS `CustomVoice` model family on macOS. Linux/CUDA setup is scaffolded, but the endpoint parity there is still in development.

OpenAPI docs are available at:

- `GET /openapi.json`
- `GET /docs`

## Status

- Primary runtime: macOS on Apple Silicon
- Primary backend: `mlx-audio`
- Primary model repo: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`
- Linux/CUDA: not ready for the new public API yet

## Setup

Run:

```bash
./setup.sh
```

What it does:

- macOS Apple Silicon: installs `ffmpeg`, creates the virtualenv, installs `mlx-audio` from GitHub
- Ubuntu/Linux: installs `ffmpeg`, installs the CUDA-side Python deps, and leaves a note that the endpoint path is still in progress

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

## Development Notes

- macOS Apple Silicon is the only runtime fully wired to the public API today
- Linux/CUDA setup is intentionally partial and documented as in-progress
- `GET /v1/voices` is a project-specific extension because voice discovery is otherwise annoying in exactly the way open source hobby servers should avoid

## License

MIT

# Keep encoding off the HTTP loop

Owning issue: https://github.com/kortexa-ai/tts.server/issues/3

Move whole-file encoding after synthesis into a worker. Preserve the existing
inference gate, codec selection, output bytes, sample-rate/voice/model headers,
and HTTP error mapping. Do not change native PCM streaming or model settings.
Cancellation during encoding must not discard the worker's waveform or affect
another PCM response; the CPU encoder may finish after its caller disconnects.

Use real ASGI requests with a gated synchronous encoder to prove that health
and PCM delivery complete before the encoder is released. Exercise both public
whole-file routes, supported container formats, encoder errors and cancellation.
Replace only GPU inference with fixed PCM in tests; also decode a response from
the real ffmpeg encoder. Declare the HTTP test client as a development dependency
without changing resolved runtime packages.

Deliver by Git to clean Smarty main, validate before mutation, then restart only
the managed TTS service if shared GPU headroom and rollback checks pass. Verify
fresh process identity, health, PCM first bytes and whole-file decodability.
Do not restart unrelated services or rebuild WPE. Rollback is a focused revert
through Git plus the same managed TTS deployment workflow. Measure event-loop
responsiveness separately from model/GPU and physical speaker latency. Keep
live progress, measurements and deployment status on the owning issue.

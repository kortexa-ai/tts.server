# Custom-voice startup preparation

Owning issue: https://github.com/kortexa-ai/tts.server/issues/4

## Behavior

The faster CUDA backend captures generation graphs at load time, but reference
prompt extraction and first streaming decode are separate lazy operations.
Prepare them before HTTP startup finishes by consuming one chunk through the
normal streaming path for each registered reference-backed voice. Close each
stream before starting the next voice. Do not play, save, or retain the audio.

After all voice streams are closed, release unused CUDA allocator blocks once
so temporary reference-extraction buffers do not occupy shared GPU headroom.
Keep live prompt tensors and captured graph pools. Do not flush the allocator
on the request path. A cleanup failure must not trigger another model load.

Keep the voice registry, default selection, model, chunk size, and dependencies
unchanged. Log individual failures and continue with the other voices. Do not
perform startup inference from the voice-reload endpoint. Other backends are
outside this change.

## Validation

Use a fake SDK at the real loader boundary to verify graph-before-voice order,
serial execution, a one-chunk limit, nested generator closure, tuple and array
chunks, empty or failed generation, and preserved voice availability. Run all
HTTP and streaming regression tests. No test requires a GPU or real recording.

Verify cleanup runs only on CUDA, once after every voice has closed, including
when a voice failed. Test cleanup failure without losing model readiness.

For production verification, restart only the existing TTS service using
`ktxsvc`, then measure the first HTTP PCM request after readiness and warm
repeats. Record startup timing and shared GPU footprint in the issue. Use the
same installed environment, no second model process, and no speaker playback.
Compare with the pre-change cold request; do not describe this as a steady-state
latency gain or a fix for physical audio onset clipping. On failed deployment,
use a focused Git revert, sync, and the same service-specific restart path.

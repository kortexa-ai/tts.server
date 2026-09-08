# Whole-file request cancellation

Owner: [tts.server #5](https://github.com/kortexa-ai/tts.server/issues/5).
This work blocks the [realtime latency program](https://github.com/kortexa-ai/hermes-livekit/issues/14).

Skip whole-file inference when its caller disconnects before admission. Keep
already-running inference owned until the worker finishes, even if the HTTP
handler is cancelled. Do not let abandoned jobs queue behind the synchronous
model lock or unload the model while a worker still uses it.

## Contract and validation

- Cover both `/v1/audio/speech` container output and `/generate`.
- Preserve response formats, metadata, error semantics and off-loop encoding.
- Keep the existing streaming cancellation and serialized inference contract.
- Verify real TCP disconnects separately from explicit ASGI handler cancellation.
- Use event-gated fake inference behind the real service lock; test a live PCM
  request after abandoned whole-file work and shutdown during owned inference.
- Run the complete CPU test suite without CUDA visibility. No test captures a
  microphone, plays audio or loads model weights.

## Delivery

Validate the exact pushed main commit on Smarty before restarting TTS through
ktxsvc. Record the previous service PID, clean checkout, dependency health and
GPU allocation, and prove a focused Git revert path before mutation. If startup
needs more GPU headroom, use the explicit permission to stop ASR temporarily;
restore that exact managed service and verify it before completion. Leave LFM,
the unknown GPU process, Miso and other workloads alone. Keep restoration owned
by the deployment process, including failure paths.

Keep test results, measured behavior, deployment state and rollback evidence in
the issue rather than this contract.

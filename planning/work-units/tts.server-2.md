# Source review and audio correctness

Owning issue: https://github.com/kortexa-ai/tts.server/issues/2

## Scope

Review request lifecycles, audio framing, inference ownership, and latency in
tts.server. Trace the contracts with api.server, asr.server, and tts.server.
Keep fixes bounded and add deterministic regression tests for changed behavior.

## Validation and delivery

The 2026-09-06 source-only restriction was lifted on 2026-09-13. Reconcile
the draft with newer upstream fixes, run deterministic tests and production
validation, then deliver through Git and the documented Smarty service workflow.
Coordinate GPU windows over Agent Bus; the LegoLM and hamster-orchestra
experiments take priority. Record and restore the exact service baseline.

## Approach

1. Inspect code and existing test coverage.
2. Repair defects supported by source evidence and simplify affected paths.
3. Review the resulting diff, run regression and integration checks, and deliver.

Live validation results and delivery status belong in the owning issue.

Cross-repository review notes and test commands: api.server/docs/audio-code-review.md.

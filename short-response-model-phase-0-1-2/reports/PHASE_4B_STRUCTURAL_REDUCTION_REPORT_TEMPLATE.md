# PHASE 4B STRUCTURAL REDUCTION REPORT

## Scope

Describe the accepted Phase 4 source model, candidate matrix, and intentionally excluded experiments.

## Inputs

- Repository revision:
- Source model path and hash:
- Tokenizer hash:
- Calibration manifest:
- Benchmark version:
- Hardware/software manifest:
- Generation settings:

## Candidate matrix

| Candidate | Structural change | Control relationship | Checkpoint path |
|---|---|---|---|
| dense_reference | None | Untouched control |  |
|  |  |  |  |

## Compatibility verification

Record smoke-forward, save, fresh reload, tensor shape, and configuration checks.

## Behavioral results

Include aggregate metrics, category floors, critical/major semantic issues, and links to raw `results.jsonl` files.

## Footprint and runtime results

Include measured parameter count, serialized bytes, peak VRAM/RAM, prefill time, decode time, and total response latency. Do not substitute theoretical FLOPs.

## Recommendation

State which candidate is recommended and why. A dense recommendation is valid.

## Human review

Summarize raw-reply review, safety review, long-form degradation, rollback readiness, and final decision.

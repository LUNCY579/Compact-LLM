# Single-layer graft result

Status: COMPLETE.

Supa post-L12 + one frozen Qwen3-4B L16 residual correction. CPU FP32/eager, no cache.

| Smoke arm | Correct | Mean cosine | Mean KL | RMS range | Mechanical gate |
|---|---:|---:|---:|---|---|
| A0 | 6/8 | 1.000002 | 0.000000 | 1.00000–1.00000 | True |
| Z | 6/8 | 1.000002 | 0.000000 | 1.00000–1.00000 | True |
| I | 6/8 | 1.000002 | 0.000000 | 1.00000–1.00000 | True |
| N | 6/8 | 1.000002 | 0.000000 | 1.00000–1.00001 | True |
| R | 6/8 | 0.998636 | 0.005956 | 1.00000–1.00002 | True |
| S | 6/8 | 0.999075 | 0.006209 | 0.99961–1.00000 | True |
| G | 6/8 | 0.999612 | 0.003998 | 1.00000–1.00152 | True |
| A1 | 6/8 | 0.988599 | 0.073012 | 0.99985–1.00003 | True |

Z: exact zero; I: identity donor; N: near-zero; R: random; S: shuffled pairs; G: aligned; A1: native recurrence.

Primary mechanical gate: **True**.
A passed tiny-gate or zero-gate control does not demonstrate useful transfer.

Scenario-held-out bridge cosine: up 0.6120, down 0.8574.
Calibration: 2544 training tokens, 526 held-out tokens. No answer labels in fitting.

| Development arm | Correct | Wins / losses | Preservation |
|---|---:|---:|---|
| A0 | 12/16 | 0 / 0 | True |
| G | 12/16 | 0 / 0 | True |
| R | 12/16 | 0 / 0 | True |
| S | 12/16 | 0 / 0 | True |
| A1 | 12/16 | 0 / 0 | True |

A0/G/R/S/A1 use the same 16 previously inspected development cases. Results are exploratory; no independent efficacy claim.

This tests a single prespecified donor layer, insertion point, map fit, and gate. Failure does not rule out other bridges or layers.
All backbone weights stay frozen. No router, MLA, MTP, repeated graft, distilled model, sealed evaluation or published checkpoint.

See run_manifest.json for integrity, hashes and resource receipts; bridge_fit.json for alignment; raw_results.jsonl for individual outcomes.

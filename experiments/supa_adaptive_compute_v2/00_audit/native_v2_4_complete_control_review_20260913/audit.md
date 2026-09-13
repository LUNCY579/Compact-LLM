# Complete Native v2.4 control mechanics audit

Status: **PASS**. 3743 checks passed; 0 failed. Cutoff: 2026-09-13T17:41:10.088007+00:00.

A0 covers 128/128 and AM covers 128/128 frozen originals. No efficacy statistics are calculated. Confirmation requires the complete five-arm matrix.

AM alpha: 0.764362266301519; copied state SHA256: `776e3a6471dc3825f16e36edf2c0b064939a9b17a59e064fcf42c206658b9d6a`.

AM recorded RMS range: [1.0, 1.0004412787276908], inside [0.25, 4.0]. Invalid outputs: 0; safety stops: 0.

The audit reconciles raw JSON, source/config/data/calibration/GPU bindings, zero training inventories, baseline completeness, saved logits as bytes, and reconstructed dominant FLOPs. No models or tensor libraries were loaded.

Limitations:

- Receipt and artifact-byte reconciliation, not inference or numerical tensor replay.
- The recorded control state is bound to the separately audited calibration; real model state is not reloaded.
- Validity flags reuse the exact frozen output_flags function extracted by AST, with no module or torch import.
- Raw response-to-token decoding cannot be independently reconstructed without loading the tokenizer.
- Mean latency is observational and affected by machine contention; no causal cost comparison is claimed.
- This review covers A0 and AM only. Full five-arm architecture audit remains mandatory before confirmation.
- AM has 90 percent of A3 additional dominant matrix FLOPs; it is not an exact compute match.

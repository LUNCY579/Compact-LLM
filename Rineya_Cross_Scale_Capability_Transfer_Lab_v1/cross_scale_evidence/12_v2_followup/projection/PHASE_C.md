# v2 Phase C — Better T

Exploratory. Not training-free confirmatory evidence.

## D4 linear CKA

Calibration prompts only (SHA `d2f1f415…`). MERGEDEV / DEV-FULL / sealed answers were not read.

- Qwen3-0.6B vs Supa, 16 calibration prompts, CPU: identity mapping, alignment score 27.82. Cached at `cka_qwen06_vs_supa.json`.
- Qwen3-4B / 8B residual CKA: `not_run_compute`. GPU remained busy under the funnel snapshot rule (utilization ≥10% or ≥1 GiB VRAM). 4B/8B hidden-state collection was not taken on CPU.

Because same-family 0.6B CKA recovered a strictly increasing identity map, Phase C kept the frozen D2 layer map for the single new T.

## P5 direction-transfer SVD

One calibration-free residual SVD into the target basis: keep 0.6B singular vectors, transplant a rank-8 donor singular-value profile at strength 0.10, preserve the tied-embedding contract.

| Artifact | Status | Digest / notes |
|---|---|---|
| `qwen4b_dir_s010` | GO | 596,049,920 params, tied, finite logits, EOS, smoke “READY.” Weight SHA `34011e16…`, digest `9c44c7f1…` |
| 8B P5 | `not_run_compute` | GPU busy; 8B activation/projection not stacked after the 4B T already re-entered Phase A |

P5-4B as a **projected control** on MERGEDEV-fast: **13/16**, overall 0.8125, completion 0.75, format 0.625. Same semantic score as v1 `CONS_MEAN_A1500`; worse preservation than ungated Supa.

## Phase A re-entry for P5 only

`Δ_P5 = P5_4B − Qwen3-0.6B` digest `07822599…`. Injected into Supa at frozen α=0.15 (`P5_TASK_4B_A1500`, digest `1e2fdebd…`).

Validation used the Supa template (310 stored tensors, no separate `lm_head`; tied embeddings remain active). Official Qwen3-0.6B stores an extra `lm_head.weight` key and is the wrong structural template for a Supa merge.

MERGEDEV-fast: **12/16**, overall 0.75, completion 1.0, format 0.875. Ties ungated Supa; loses to v1 consensus (13/16). Not two-control eligible.

## Gate

No new T produced a two-control MERGEDEV-fast win. DEV-FULL and sealed remain unauthorized. No additional T families were stacked.

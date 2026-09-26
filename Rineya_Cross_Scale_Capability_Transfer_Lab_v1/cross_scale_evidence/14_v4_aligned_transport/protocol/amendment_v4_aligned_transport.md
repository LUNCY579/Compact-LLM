# Amendment v4 — Aligned Weight Transport, Soup, and Full-Scale Evaluation

Protocol version: `4.0.0-exploratory`

Parents: v1 `preregistration.md` `1.0.0`, `amendment_v2_exploratory.md`,
`amendment_v3_transport_diagnosis.md`, `amendment_v3_1_donor_side.md`. All
immutable and unchanged.

Frozen before any v4 weight is written.

## Why this stage exists

Every operator v1–v3 tested moved donor weights into the target frame by
truncation or spectral substitution. The donor's directions arrived meaningless,
which produced a squeeze that v3 measured directly:

- large coefficients collapse the model (P4 at strength 1.0; all eight P1 arms);
- coefficients small enough to stay coherent are functionally invisible — the v1
  and v2 candidates are indistinguishable from Supa-A1.1 at symmetric KL 0.0005,
  with identical argmax on 58 of 59 prompts.

v3.1 then produced the missing ingredient: a **validated per-depth basis map**
between donor and target residual streams, held-out mean cosine 0.982 on a
prompt-disjoint split. The lab already contains `direction_transfer.ridge_linear_map`
and `direction_transfer.project_weight` to apply such a map to weights; a grep
shows they are referenced **only by `tests/test_direction_transfer.py`** and by no
script or checkpoint in the study.

v4 uses that map for the first time.

## Corrected baseline

Correction 0001 established that the study target is **Supa-A1.1** at 45/64
Showcase, not Supa 1.7 at 40/64. Consequences frozen here:

- **Qwen3-4B is retired as a donor.** At 43/64 it scores below the target.
- The only donor is **Qwen3-8B** at 53/64, an 8-point gap.
- Both figures are soft: A1.1's card labels its 45/64 a "tuning-split checkpoint
  evaluation" claiming no significance, and the 8B's 53/64 was measured at NF4
  4-bit and never selected on this bank. The biases run opposite ways.

## Hypotheses

- **HA1 (coherence).** An aligned projection survives a materially larger
  coefficient than an unaligned one. Falsified if the aligned candidate collapses
  at coefficients where P4-s010 was coherent.
- **HA2 (functional visibility).** An aligned candidate is functionally distinct
  from Supa-A1.1 — symmetric KL against A1.1 at least an order of magnitude above
  the 0.0005 that v1/v2 candidates achieved. This is the kill criterion: a
  candidate that is functionally identical to the target cannot be an improvement
  whatever it scores.
- **HA3 (capability).** An aligned candidate, or a soup over aligned candidates,
  beats Supa-A1.1 on Showcase64 **and** holds on Stability192.

HA2 gates HA3. A candidate failing HA2 is not evaluated further.

## The operator, P6

Treat each layer as acting on the residual stream. With `A_d` the fitted map from
donor hidden space to target hidden space at matched relative depth, and `B_*`
the corresponding map for each inner space:

- read matrices (`q/k/v/gate/up`), donor `[inner_d, hidden_d]`:
  `W_t = B^T (W_d A_d)`
- write matrices (`o/down`), donor `[hidden_d, inner_d]`:
  `W_t = A_d^T W_d B`

Inner spaces, Qwen3-8B to Supa-A1.1:

| family | donor inner | target inner | map |
|---|---:|---:|---|
| `k_proj`, `v_proj` | 1024 | 1024 | **none required — dimensions coincide** |
| `q_proj`, `o_proj` | 4096 | 2048 | `B_attn`, fitted |
| `gate/up/down` | 12288 | 3072 | `B_mlp`, fitted |

`k_proj` and `v_proj` are the cleanest case in the whole study: donor and target
key/value spaces are both 8 heads x 128, so those tensors need only the validated
residual map and no additional fitted correspondence. They are therefore
evaluated as a separate arm before anything wider is attempted.

**All maps are orthogonal Procrustes, not ridge.** Orthogonal transforms preserve
norms and therefore commute with RMSNorm up to a scale, which ridge does not. v3.1
measured Procrustes at held-out cosine 0.777–0.932 against ridge's 0.873–0.982;
the lower-fidelity map is the correct instrument for rebasing weights, and using
ridge here would be a category error.

Depth correspondence is the frozen D2 uniform relative-depth map, which v3.1's E5
validated functionally: the donor-to-target CKA surface is diagonal-dominant.

## Arms, frozen

- `P6-KV`: `k_proj` and `v_proj` only. No fitted inner map.
- `P6-ATTN`: adds `q_proj`, `o_proj`.
- `P6-FULL`: adds `gate_proj`, `up_proj`, `down_proj`.
- Injection: `W = (1 - alpha) W_target + alpha W_projected`.
- Coefficient ladder: `alpha in {0.05, 0.10, 0.25, 0.50, 1.00}`. The ladder
  deliberately extends far beyond v1/v2's 0.15 ceiling, because HA1 predicts an
  aligned projection tolerates more. Report the largest coherent alpha per arm.
- Soup: uniform average over the coherent arms, in the family's own tradition.

## Controls, frozen

Every candidate is reported beside:

1. untouched Supa-A1.1;
2. a **norm-matched random projection** — replace `A_d`/`B` with random
   semi-orthogonal matrices of identical shape, everything else unchanged. This
   is the control that separates "the fitted map carries something" from "any
   rebasing of donor weights at this norm does this";
3. a **shuffled-depth projection** — correct maps, wrong donor layer.

## Evaluation

Development only. Showcase64 and Stability192 are the family's own banks, run
with the family's own committed `run_local_hf.py` and `evaluate_predictions.py`
at greedy / 24 new tokens / seed 17 / thinking-off, matching the protocol under
which the published family numbers were produced.

Showcase64 is **selection data** for family cores by the family's own
documentation, so a Showcase gain alone is not evidence. **Stability192 is the
reporting set**, and any claim must hold there.

The cross-scale study's own sealed holdout remains CLOSED and is not read.
Invisible-Limit TEST2 remains denylisted.

## Hardware envelope

Measured at freeze: 15.2 GiB RAM total, 4.7 GiB free, CUDA unavailable
(`torch 2.13.0+cpu`), 224 GiB disk free, GPU idle at 706 MiB used by an unrelated
process.

- Qwen3-8B (15.26 GiB) is never loaded. All donor reads stream layer by layer.
- Activation capture for fitting `B_mlp` subsamples token positions to keep the
  donor MLP-intermediate matrix under 1 GiB.
- One model resident at a time. Candidates are 0.6B-shaped and load normally.
- Every stage records wall time, peak RSS, and free-RAM floor. A stage that would
  push free RAM below 1.0 GiB is subsampled further rather than run.

## What v4 may claim

v4 is exploratory under the parent amendment policy. It may report
development-set comparisons against the frozen controls. It may not claim sealed
confirmation, and a Showcase-only gain is explicitly not a result.

## Amendment policy

Inherited from v1. Immutable after the first v4 weight is written.

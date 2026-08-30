# Cross-Scale Transfer Verdict Through DEV-FULL

## Decision

**No confirmatory GO. Do not access the sealed holdout.**

The study found a coherent, highly aligned spectral direction from Qwen3-4B
and Qwen3-8B, but the development-selected Supa injection did not improve the
larger DEV-FULL suite.

## What worked technically

- All four source architectures and tokenizer identities were hash-verified.
- P1 deterministic depth/width projection produced exact 0.6B-shaped state
  dicts, but all eight tested arms collapsed functionally.
- Embedding-ridge and embedding-Procrustes mappings also failed coherence.
- Target-basis rank-8 spectral transfer at full strength failed, while strength
  0.10 produced coherent 596,049,920-parameter 4B and 8B controls.
- The 4B and 8B spectral deltas are strongly aligned: cosine 0.9320, sign
  agreement 0.9264, magnitude correlation 0.9145.
- Both donors are construction-only; all projected/merged outputs retain the
  0.6B inference footprint.

## Baselines and projected controls

MERGEDEV results:

| Model | Semantic | Format | Completion |
|---|---:|---:|---:|
| Official Qwen3-0.6B | 42/59 | 45.8% | 66.1% |
| Supa ungated | 47/59 | 88.1% | 100% |
| Supa published gate | 47/59 | 88.1% | 100% |
| Projected 4B spectral control | 42/59 | 39.0% | 59.3% |
| Projected 8B spectral control | 43/59 | 47.5% | 67.8% |

The published Supa logit offset changed zero MERGEDEV responses. The 8B
spectral control was one case above 4B/Qwen, but far below Supa.

## Candidate funnel

- 8B absolute interpolation, alpha 0.01: tied Supa on MERGEDEV-fast.
- 8B task vector, alpha 0.20: tied Supa on MERGEDEV-fast.
- 4B+8B mean consensus, alpha 0.10: tied Supa on MERGEDEV-fast.
- Mean consensus, alpha 0.15 and 0.20: 13/16 vs Supa 12/16.
- Alpha 0.15 won the frozen smaller-coefficient tie-break.

On full MERGEDEV, alpha 0.15 scored 48/59 against Supa's 47/59: one
counterfactual fix, zero breaks, exact McNemar p=1.0, scenario-bootstrap 95%
CI [0.000, 0.061]. This was a development signal, not confirmation.

On DEV-FULL, both scored 144/180. The candidate fixed one paraphrase case and
broke one mathematics case. Exact McNemar p=1.0. Scenario-macro delta was
-0.0045 with 95% cluster-bootstrap CI [-0.0203, 0.0068] and only 0.233
bootstrap probability of a positive macro delta.

## Scientific interpretation

The tested results support three bounded conclusions:

1. Naive cross-scale coordinate truncation is structurally valid but
   functionally destructive, independent of tested depth map and magnitude
   selection.
2. Target-basis spectral information is a coherent transferable statistic,
   and 4B/8B spectra define a strongly shared small direction.
3. Under the frozen evaluation, that direction does not yield a robust Supa
   improvement. The one-case MERGEDEV gain is consistent with development
   selection noise because it disappears on DEV-FULL.

This does not establish that all cross-scale transfer is impossible. It rejects
the tested deterministic, embedding-aligned, and low-rank spectral assumptions
as a reproducible overall enhancement under the current gates.

## Claim boundary

Do not claim that the candidate is better than Supa, equivalent to a larger
model, or sealed-confirmed. Do not use the fresh sealed holdout unless a new
independent method passes the complete pre-sealed funnel.

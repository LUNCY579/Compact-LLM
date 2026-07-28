# PHASE 2 STOP GATE

## Purpose

Advance only when a one-, two-, or four-layer static configuration matches the all-layer quality reference within tolerance and materially reduces training resources.

## Candidate Checks

| Variant | Layers | Score | VRAM reduction | Result |
|---|---|---:|---:|---|
| bottom4 | [12, 13, 22, 27] | 0.944 | 0 | 4 | 0.920 | 8.8% | FAIL |
| top4 | [5, 11, 14, 16] | 0.935 | 1 | 4 | 0.900 | 8.8% | FAIL |
| single_15 | [15] | 0.921 | 0 | 4 | 0.920 | 9.9% | FAIL |
| pair_14_16 | [14, 16] | 0.919 | 1 | 4 | 0.900 | 9.6% | FAIL |
| pair_15_16 | [15, 16] | 0.917 | 0 | 4 | 0.920 | 9.6% | FAIL |
| pair_16_17 | [16, 17] | 0.913 | 1 | 5 | 0.880 | 9.6% | FAIL |
| random4 | [7, 13, 16, 19] | 0.912 | 1 | 3 | 0.920 | 8.8% | FAIL |
| single_16 | [16] | 0.908 | 0 | 0 | 1.000 | 9.9% | FAIL |
| single_17 | [17] | 0.908 | 1 | 4 | 0.900 | 9.9% | FAIL |
| single_18 | [18] | 0.902 | 1 | 4 | 0.900 | 9.9% | FAIL |
| single_22 | [22] | 0.899 | 1 | 5 | 0.880 | 9.9% | FAIL |
| single_13 | [13] | 0.897 | 1 | 5 | 0.880 | 9.9% | FAIL |
| single_14 | [14] | 0.890 | 0 | 0 | 1.000 | 9.9% | FAIL |
| single_11 | [11] | 0.885 | 0 | 0 | 1.000 | 9.9% | FAIL |
| single_05 | [5] | 0.875 | 0 | 0 | 1.000 | 9.9% | FAIL |
| single_09 | [9] | 0.875 | 0 | 0 | 1.000 | 9.9% | FAIL |
| single_00 | [0] | 0.866 | 0 | 0 | 1.000 | 9.9% | FAIL |
| single_12 | [12] | 0.866 | 1 | 8 | 0.820 | 9.9% | FAIL |
| single_27 | [27] | 0.846 | 1 | 8 | 0.820 | 9.9% | FAIL |

## Repeated-Seed Stability

- Repeat runs: 3
- Mean score: 0.905
- Score standard deviation: 0.0034
- Stability check: PASS

## Automated Gate Status

`FAIL`

## Recommended Static Candidate

No candidate passed every automated requirement.

## Required Human Review

Review `reports/PHASE_2_HUMAN_REVIEW_PACKET.md`, raw replies, and category scores for the recommended candidate, the all-layer reference, RANDOM4, and BOTTOM4. Confirm that the ranking is behaviorally meaningful and not a heuristic artifact. A critical semantic failure is disqualifying even when the average heuristic score is high.

- Human decision: `FAIL`
- Reviewer: Cheng Li
- Date: 2026-07-28
- Notes: No selective configuration met the predefined quality and
  resource thresholds. Preserve the negative result. Continue preference
  alignment using the accepted all-layer dense SFT reference under an
  explicitly approved dense-fallback deviation.

Dynamic IST-style selection is optional after this static gate. It must be reported separately because gradient masking does not automatically prove optimizer-state memory savings.

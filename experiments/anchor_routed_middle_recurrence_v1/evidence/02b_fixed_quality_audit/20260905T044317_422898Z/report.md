# Fixed recurrence quality audit

Status: INCONCLUSIVE.

Question: are the fixed-arm quality comparisons interpretable when the intact
baseline encounters the runtime stopping rule? This is an offline audit of the
immutable 16-case MERGEDEV-fast run; no model, new dataset, or retry was used.

| Arm | Completed / planned | Correct among completed | Censored by safety stop |
|---|---:|---:|---:|
| A0_dev | 14/16 | 10 | 2 |
| A1_dev | 14/16 | 10 | 2 |
| A3_dev | 14/16 | 11 | 2 |

Interpretation: intact development baseline censored by the frozen time budget. Censored baseline cases cannot be treated
as demonstrated language failures or credited as recurrence improvements.
The original run's raw scores and FAIL/PASS receipt are preserved. This audit
adds an explicit baseline-completion prerequisite to prevent an invalid routing
claim; it tightens interpretation and does not change a selection threshold,
rerun an example, or turn a failed candidate into a passing one.

The corrected comparator has a regression test for an apparent gain against a
timed-out baseline. Complete-case counts above are descriptive; they are not a
new selection/evaluation subset or a post-hoc efficacy test.

Non-claims: no demonstrated capability gain, adaptive routing benefit, PEFT
recovery, or sealed efficacy. Fixed recurrence's short smoke qualification and
canonical-state restoration are separate from this censored quality comparison.

Next experiment: preregister and rerun the identical A0/A1/A3 development matrix
with an adequate per-prompt CPU time budget; retain the same examples, weights,
recurrence counts, decoding and scoring. No other mechanism should change.

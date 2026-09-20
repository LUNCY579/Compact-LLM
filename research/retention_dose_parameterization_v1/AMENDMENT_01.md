# DOSEINT-001 Amendment 1: maximum balanced dual-bank-disjoint dose

Prospective amendment, before any model worker was launched or any new endpoint
was observed. The original 3,072-row D4 design is BLOCKED_AT_BUILD. The controller
refused it before creating a journal/reserving a worker because its historical
plan/contract lack the current selection-exclusion audit. The actual overlap is
275/384 selection parents. The original BUILD_GATE's PASS_DOSE_ONLY verified only
the historical dose bank and was insufficient to authorize the current native
lane. Preserve that record, its grant, and the failed launch verbatim.

Use the existing current builder, excluding the UNION of both banks (768 parents).
It independently checked all 36,752 source labels and found only 899 eligible
prerequisite_plan parents. The maximum balanced one-parent/one-row dose is
therefore 899 per family = 2,697 rows, with 380,921 processed tokens, 37,090
supervised tokens, 193 update groups, one exposure, seed 17. All rows fit 2048.
This changes both the dose and the draw: it is NOT an exact historical D4 repeat.
Do not describe a change from historical D4 as an interpolation or pure dose effect.

This amendment implements the nearest valid high-dose-plus-interpolation design
within the user's objective and one-training-run budget. An optional user design
question was sent before execution; absent contrary steering, the recommended
2,697-row disjoint design is the working assumption. This is agent judgment under
the goal's authority, not a claim that the user separately approved this amendment.

Replace original prepared data/plan/grant with prepared-dual899,
PLAN_DUAL899.json, GRANT_DUAL899.json, and BUILD_GATE_DUAL899.json. The original
grant is not consumed. The native lane, trainable tensors, FP32 precision,
optimizer recipe, alpha grid and execution order are unchanged. Validate the
complete controller.inputs() path before reserving the worker. No guard change.

All original dose-bank numerical bands, the primary conjunction, and paired
analysis rules remain fixed. A changed corpus makes these predictions uncertain;
report failed predictions as failed, without adjusting targets after results.

Both banks are now parent-disjoint from training. Add SELECT-VIEWS-001 acquisition
for every new arm using the identical model weights. This is the prospective
high-dose cross-bank bridge. Report each bank separately; no aggregate offsets,
pooled views, cross-bank paired tests or transport of historical rankings. Apply
the acquisition (net, nd, exact p, exact discordance threshold) reporting rule
within each bank. The SELECT bank bands below are new prospective predictions:

| alpha | SELECT acquisition /384 prediction band |
|---|---|
| 1.00 | 270–330 |
| 0.25 | 210–285 |
| 0.50 | 250–310 |
| 0.75 | 270–320 |

The dose-bank experiment remains report-only; SELECT's admitted status does not
retroactively promote the other bank. No model selection/promotion is performed.
Historical bridge legs remain BLOCKED_ON_COMPARABILITY for missing weight custody.

Updated numeric envelope: one native launch, no training retry, exactly 380,921
processed tokens; retain other native limits. Four materializations. Sixteen
probe legs (four arms × two acquisition banks + retention + long context),
900 seconds per leg, at most one retry per failed inference-only leg, bounded by
14,400 total probe-worker seconds and 10,000 evaluated rows /1,200,000 generated
tokens including failures/retries. Total session wall <=18,000 seconds including
preparation and failures. New disk <=17 GiB, always preserve the native 14 GiB
free-space floor. All other restrictions and original bands remain in force.

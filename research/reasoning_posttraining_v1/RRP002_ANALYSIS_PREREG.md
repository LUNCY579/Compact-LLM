# RRP-002 analysis pre-registration

Status: **REGISTERED BEFORE ANY LADDER ARM HAS RUN.** No arm exists. No candidate has been
trained. Nothing in this document was chosen after seeing an arm's result, and the whole point
of writing it now is that it cannot be.

Authored by the agent. This is an **analysis plan**, not operator-approved science: the operator
authorized compute and attested the data (`OPERATOR_DATA_APPROVAL.json`), and has not reviewed
these statistical choices. Nothing here may be reported as externally validated.

Implemented in `experiments/reasoning_posttraining_v1/statistics.py`, gated by 29 tests in
`tests/test_statistics.py`. MQ-001 supplies no statistics layer at all — `evaluation.classify`
grades only `exact` and `number`, and `analyze.summarize` skips unscoreable rows — so all of
this is new and none of it is inherited.

---

## 1. The uncomfortable number first: what this design can and cannot detect

The primary test is an **exact cluster-level sign-flip permutation test** on the mean
per-cluster difference, enumerated in full at or below 22 clusters. Everything else is
secondary. Asymptotic intervals are not used at all, because at k = 8 or k = 16 they are invalid.

A sign-flip test's most extreme two-sided outcome has probability `2 / 2^k`, so that is the
smallest p-value obtainable. **Measured, by enumeration, for the real cluster counts:**

| clusters that moved | smallest two-sided p | can reach α = 0.05 |
| --- | --- | --- |
| 3 | 0.2500 | **no** |
| 4 | 0.1250 | **no** |
| 5 | 0.0625 | **no** |
| 6 | 0.0312 | yes, only just |
| 8 | 0.0078 | yes |
| 16 | 0.00003 | yes |

Two consequences, both forced by measurement rather than preference:

**(a) A design with fewer than 6 moving clusters cannot produce a significant result at all,
however large the effect.** `resolution_floor` computes this before any data is examined and
`compare` sets `significance_refused`, returning the estimate and interval while refusing the
significance claim. A layer that reported "p = 0.125, not significant" there would be lying by
omission — the design has no resolution, which is a different statement from "no effect".

**(b) Zero-difference clusters buy no power.** A cluster whose difference is exactly zero
contributes nothing to the statistic *and* nothing to the permutation distribution, since
flipping its sign changes neither. So **589 clusters of which 6 moved has the resolution of
k = 6, not k = 589.** The module reports `effective_clusters_nonzero` and computes the floor
from it. This was found by measuring my own tool rather than by reasoning about it, and without
it the preservation bank's power would have been overstated by orders of magnitude.

**(c) The test detects consistency of direction, not magnitude.** Measured: 8 clusters all
moving the same way gives p = 0.0078 whether the shift is 0.30 or 0.003. Significance here means
"the sign was consistent across clusters". **That is exactly why a pre-declared practical margin
is mandatory and not optional** — it is the only thing standing between a consistent-but-trivial
effect and a reported improvement.

---

## 2. Cluster units, fixed now

The unit must live on the data so it cannot be chosen at analysis time; `paired_clusters` raises
if the key is absent rather than defaulting.

| bank | primary unit | k | secondary | why |
| --- | --- | --- | --- | --- |
| `RRP2-DEV-V1` | `family` | **8** | `group_id` | worlds in a family share grammar and solver; worlds in a group_id are variants of one case. Family is the conservative level. |
| `RRP2-DEV-V2` | `family` | **8** | `group_id` | as v1 |
| `RRP2-PRES-DEV` | `cluster_id` | **589** | `sub_bank` | `cluster_id` is the digit-blinded last user turn, carried at build time (CL-038) |

**The Forge banks support only 8 independent clusters.** That is a hard limit of the instrument,
not of the analysis: cross-family generalization claims on v1/v2 will have very little power, and
an arm must either move most families or make its case within-family on `group_id`. Stated here
so it cannot later be presented as a surprise.

**Sub-bank (b) of the preservation bank has 16 clusters for 192 rows.** Per-sub-bank results
must report their own k. A row-level interval there would be roughly three times too narrow.

Both the **row-weighted** and **cluster-weighted** point estimates are always reported together.
They disagree whenever clusters are unbalanced, and the preservation bank's are very unbalanced
(one cluster holds 18 rows while 570 hold one). Reporting only the row-weighted figure lets a
handful of templated rows drive the headline; reporting only the cluster-weighted one discards
real rows.

**Family weights: equal per family, not per row.** The research question is cross-family
generalization, and row counts per family differ. The cluster-weighted estimate implements this
directly; it is the inferential statistic, and the row-weighted figure is descriptive.

---

## 3. Primary contrasts — exactly four, fixed now

Only these four enter the multiplicity family. Everything else computed is **exploratory** and
will be labelled so. Choosing the family after seeing the p-values is how multiplicity control
gets quietly abandoned, so `holm()` is applied to this set and nothing else.

| # | contrast | bank | unit | test | margin |
| --- | --- | --- | --- | --- | --- |
| **P1** | arm reasoning accuracy > source | `RRP2-DEV-V2` | `family` | two-sided sign-flip | **+0.05** |
| **P2** | arm reasoning accuracy > shortcut floor | `RRP2-DEV-V2` | `family` | two-sided sign-flip | **floor + 0.05** |
| **P3** | preservation non-inferiority | `RRP2-PRES-DEV` | `cluster_id` | one-sided shifted | **−0.03** |
| **P4** | abstention calibration on minimal pairs | `RRP2-PRES-DEV` (c) | `pair_id` | two-sided sign-flip | **+6 of 40 pairs** |

### Why these margins

**P1 at +0.05 is a *learning* claim, not a capability claim.** The source sits at 0.0174. An arm
that improves on that has demonstrated the training did something — and nothing more. Any report
of P1 must say so in the same sentence.

**P2 at floor + 0.05 is the capability claim, and it is the one that matters.** The corrected v2
shortcut floor is **0.4391** on the evaluated subset (CL-040), reached by copying the
`{"status":...}` literal the prompt itself prints. An arm at or below that has learned a
prompt-copying shortcut, not planning. So P2 requires **≥ 0.4891 absolute accuracy**. This is a
demanding bar for a ~0.6B model starting from 0.0174, and it is set deliberately: the whole
programme exists to avoid mistaking floor-clearing for reasoning. **If P1 passes and P2 fails,
the honest headline is "the arm learned something that is not the task."**

**P3 at −0.03** is the largest preservation loss that will be tolerated. Failure to reject is
**not** evidence of preservation and will be reported as "preservation not demonstrated", never
as "preservation confirmed" — with 16 clusters in sub-bank (b) that failure is the expected
outcome for small samples.

**P4 at 6 of 40 pairs** is set at the design's resolution limit: 6 movers is precisely the
minimum that can reach p ≤ 0.05 (0.0312, measured above). Setting the practical margin *at* the
resolution floor means the plan never claims something it cannot detect and never detects
something it would not call meaningful.

---

## 4. Allowed model-selection reads, fixed now

Adaptive re-reading of a DEV bank is how a held-out surface silently becomes a training surface.
Each read below is counted in `RUN_LEDGER.jsonl`, and the count is part of the reported result.

| bank | purpose | reads allowed |
| --- | --- | --- |
| `RRP2-DEV-V2` | model selection across the entire ladder | **3** |
| `RRP2-DEV-V1` | cross-grammar check on an already-selected arm | **1 per selected arm**, never for selection |
| `RRP2-PRES-DEV` | pass/fail gate against the P3 margin | **gate only**, never for selection or ranking |
| `residual_v23 fresh_sealed` | — | **0, ANY_READ-barred** (another programme's seal, CL-035) |

Exceeding a count does not invalidate the measurement, but it **does** change what may be
claimed: any result produced after the budget is exhausted is reported as exploratory, and the
number of reads already spent is stated beside it.

---

## 5. What gets reported regardless of outcome

Registered so that a negative or null result cannot be quietly dropped:

1. All four primary contrasts, with estimate, interval, p-value, k, `effective_clusters_nonzero`
   and the resolution verdict — **whether or not any of them passes**.
2. Every bank's shortcut floor beside every model number, always.
3. The positive control for each bank. A bank whose control is below 100% is a broken
   instrument, and a broken instrument is not a model finding.
4. Both weightings, and any disagreement between them.
5. Every `significance_refused` verdict, in the same table as the estimates.
6. The read counters from §4.

---

## 6. What this design cannot show, stated before it is asked to

- **No independent confirmation.** Not achievable for RRP-002 (CL-017, bounded by CL-035). The
  banks are reserved and machine-verified, not independently authored, and none is human-reviewed
  per-world. Label: `MACHINE_VERIFIED_NOT_HUMAN_REVIEWED`.
- **No cross-bank comparison.** The three floors are 0.1458, 0.4345 and 0.4375. An accuracy is
  meaningful only against its own bank's floor.
- **No claim about sub-billion models in general.** Both length-stress splits are unreachable on
  this host (CL-016), which is a host limit and says nothing about the model.
- **Cross-family generalization on the Forge banks is underpowered at k = 8** and will be
  reported with that caveat attached, not as a clean null.
- **A single seed is a single seed.** MQ-001's three-seed replication was consistent in 3/3
  orders and still not significant; the same discipline applies here.

---

## Appendix A — what P2 actually demands, decomposed

Registered because the threshold "0.4891 absolute accuracy" is misleading in both directions if
quoted without this decomposition. Measured on the 230-row evaluated subset
(ANSWER 131 / IMPOSSIBLE 66 / UNKNOWN 33):

| predictor | correct | accuracy |
| --- | --- | --- |
| perfect abstention + **zero** reasoning | 99/230 | 0.4304 |
| `copy_prompt_sentinel` — **the floor** | 101/230 | **0.4391** |
| **P2 threshold** (floor + 0.05) | **113/230** | **0.4891** |

The v2 prompt prints its own status literal, so perfect abstention is nearly free and the floor
is very close to *"abstain perfectly, solve nothing"*. Therefore:

> **P2 = learn the abstention contract AND genuinely solve ≥ 14 of the 131 answerable worlds
> (10.7% of them).** The source currently solves **0** of 131.

This cuts both ways and both must be reported:

- It makes P2 **reachable**. It is not "go from 0.0174 to 0.4891 of general competence"; it is
  "acquire a contract the prompt already states, and solve one answerable world in nine".
- It bounds what passing P2 **means**. Passing licenses *"solves about 11% of the planning
  worlds"* and nothing stronger. No report may present it as general planning competence.

## Appendix B — a note on zero-difference clusters in the two tests

CL-046 established that a zero-difference cluster buys no power in the **superiority** test,
because flipping its sign changes neither the statistic nor the permutation distribution. The
opposite is true in the **non-inferiority** test: after the null is shifted by the margin, a
cluster that did not move is positive evidence that the loss does not exceed the margin, and it
contributes normally. So `effective_clusters_nonzero` is the honest resolution measure for P1
and P2, and the full cluster count is the honest one for P3. The implementation computes the
effective count after the shift is applied, which gives exactly this behaviour; it is not an
inconsistency.

## Appendix C — the layer is verified against a closed form, not against itself

With equal cluster magnitudes the sign-flip test reduces exactly to the two-sided sign test,
whose tail is a sum of binomial coefficients. Checked, sharing no code with the implementation:

- all-same-sign gives exactly `2 / 2^k`, verified for k = 2…14
- every `(k, m)` split at k = 6, 8, 10 matches the analytic binomial tail count
- one-sided p never exceeds two-sided, and p is monotone in the shift
- over 60 random inputs, p is always in range and never below `2 / 2^k_effective`

Four tests in `test_statistics.py` carry these.

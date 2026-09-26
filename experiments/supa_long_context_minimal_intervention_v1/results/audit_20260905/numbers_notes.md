# Per-example numerical audit

Audit command, from repository root (CPU only, no model imports):

```powershell
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/scripts/audit_result_numbers.py'
```

Script: `experiments/supa_long_context_minimal_intervention_v1/scripts/audit_result_numbers.py`.
Machine-readable evidence: `experiments/supa_long_context_minimal_intervention_v1/results/audit_20260905/numbers.json`.
Source observations: the six `results/phase0_eval*/per_example.jsonl` files listed
and SHA-256 identified in that evidence. Accepted answers are read from matching
cached `conditions.jsonl`; each raw prediction is normalized and rescored using
the official demo scorer. The audit itself independently implements the paired
bootstrap and exact McNemar calculation in `src/slcmi/stats.py` without importing
any model, CUDA, or training modules. Original records are never written.

## Checklist 4.1 — CONFIRMED, with report corrections

There are zero differences between recomputed frames and any existing
`summary.json`, secondary `analysis.json`, or
`results/amendment_01_core_gate_comparison.json` numeric field that the audit
checks. Every stored prediction normalization, correctness flag, malformed flag,
and checked condition metadata field also agrees with recomputation. Duplicate
condition/example keys and missing paired A/B/C records are rejected.

These recomputed condition-A counts match the claimed published targets; the
audit does not use a stored validity receipt as evidence of reproduction:

| Core / gate | Showcase | Stability | All | A/B/C correct, each of 256 | D correct, of 160 |
|---|---:|---:|---:|---|---:|
| A1.1 on | 45 | 125 | 170 | 170 / 147 / 97 | 111 |
| A1.1 off | 41 | 121 | 162 | 162 / 142 / 80 | 113 |
| A1 on | 43 | 125 | 168 | 168 / 147 / 96 | 111 |
| A1 off | 40 | 121 | 161 | 161 / 140 / 79 | 113 |

No published expected count was supplied for the A1/off cell.

## Accuracy and contrast recomputation

Primary answerable counts are A 148/192, B 118/192, C 67/192. Primary
unanswerable counts are A 22/64, B 29/64, C 30/64. On the oracle-eligible
frame the matched counts are A 134/160, B 105/160, C 62/160, D 111/160.

| Primary contrast / frame | Delta points | Row-bootstrap interval |
|---|---:|---|
| B−A, all | −8.984375 | [−14.062, −3.906] |
| B−C, all | +19.53125 | [+12.109, +26.953] |
| C−A, all | −28.515625 | [−35.938, −21.094] |
| B−A, answerable | −15.625 | [−21.354, −9.896] |
| B−A, oracle eligible | −18.125 | [−24.375, −11.875] |
| D−A, oracle eligible | −14.375 | [−22.500, −6.250] |
| D−B, oracle eligible | +3.750 | [−3.750, +11.250] |

The primary answerable abstention counts are A 2/192, B 21/192, C 50/192.
The observed C/A ratio is exactly 25, so the report's "twenty-six times" arose
from dividing rounded percentages. The B/A ratio is 10.5.

LCR dev all-row counts are A 59/200, B 74/200, C 59/200. Its answerable/oracle
frame has A 49/150, B 45/150, C 15/150, D 57/150. Unanswerable counts are
A 10/50, B 29/50, C 44/50. Answerable abstention counts are A 18/150,
B 46/150, C 109/150, D 37/150.

| LCR answerable contrast | Delta points | Row-bootstrap interval |
|---|---:|---|
| B−A | −2.666667 | [−10.667, +5.333] |
| D−A | +5.333333 | [−2.667, +13.333] |
| D−B | +8.000000 | [+0.000, +16.000] |
| B−C | +20.000000 | [+12.000, +28.000] |

Strict versus answer-substring scoring on LCR answerable A is exactly 49/150
versus 52/150: 32.6667% versus 34.6667%, a two-point difference. D is 57/150
strict versus 59/150 lenient. This confirms the containment sensitivity, but
does not independently establish the semantic validity of the lab's short target
or establish a maximum retriever ceiling. A small formatting gap is insufficient
to conclude all remaining errors are capability failures.

Task-family A→B correct counts (each denominator 32) independently reproduce:
alias_resolution 4→1; causal_disambiguation 32→30; count_with_exclusions 16→12;
latest_verified_record 27→26; multi_condition_identity 19→18;
object_state_tracking 26→27; rule_exception 15→14; temporal_correction 31→19.
These aggregate scores do not establish why individual evidence was omitted.

## Numerical claims outside summaries — REFUTED or qualified

* **REFUTED:** LCR evaluated context mean is **2296.365 tokens**, not 2381.
  Median is 2315.5; minimum/maximum 2112/2554. The repeated description as
  approximately 2.4K should be approximately 2.3K for these evaluated rows.
  Primary mean 1260.046875, median 1264, minimum/maximum 1048/1481 confirm.
* **Qualified:** Primary B/C/D selected token means are 560.33203125,
  543.94921875, and 541.19375. Mean per-row retention is **45.05855%,
  43.67273%, and 43.44638%**. The report's approximately 44.5%/43.2%/43.0%
  uses ratios of aggregate means, with D originally compared to the all-row A
  mean. Use a labeled definition and a matched denominator for D. B/C mean-token
  ratio is 1.0301183; equal chunk counts do not mean exactly equal token budgets.
  LCR mean selected tokens are B 524.45, C 520.06, D 523.50667; mean per-row
  retention is 22.91955%, 22.72170%, 22.93087% respectively.
* **REFUTED:** "Ungated they are identical on every contrast reported here."
  Ungated A1.1 and A1 all-row B−A are **−7.8125 versus −8.203125** points;
  D−A is **−13.125 versus −12.5**. Their answerable B−A deltas both equal
  −25/192 = −13.020833 points. Similar observed behavior is supported;
  equivalence or "no effect" is not established.
* **Qualified:** The gated core difference on B−A is **0.520833** points in
  the answerable frame, but **0.78125** over all rows. The frame must be named.
  The core-by-gate answerable interaction is −0.520833 points under the report's
  sign convention. Removing the A1.1 gate changes answerable B−A by +2.604167
  points; the ratio of ungated to gated loss is 5/6. Calling that ratio the
  fraction mechanistically "not the gate" is stronger than this arithmetic.
* **CONFIRMED, narrowly:** the A1.1 gate's net all-row A benefit of 8 correct
  decomposes into 9 additional unanswerable correct answers and 1 fewer
  answerable correct answer. It does not establish that the gate does
  "nothing else," including on other conditions or suites.
* **UNCERTAIN from these observations:** peak memory, whole-stage wall-clock,
  model-load counts, checkpoint-file preservation, isolated-logit tests,
  rounding-loss percentages, full-corpus supporting-fact totals, and published
  power calculations cannot be recomputed from these per-example observations.
  They need their own source/receipt audit and are not endorsed here. Per-row
  inference durations and generation counts are included in the machine output.

## Interval and interpretation caveats

The preregistered calculation uses 10,000 draws, seed 20260905, sorted example
IDs, and resamples paired rows. It treats variants of one scenario as independent;
the primary data contain 64 scenarios and LCR contains 50. An explicitly
unregistered whole-scenario sensitivity is provided in `numbers.json`.
For illustration: primary oracle D−A becomes [−24.841, −4.321]; LCR
answerable D−A becomes [−3.333, +14.000], and D−B becomes [−2.000, +18.000].
The main direction/uncertainty distinction survives, but these are descriptive
intervals, not a license for significance claims.

The existing exact-McNemar summaries round probabilities to six decimal places;
zero there is rounding, not an exact zero probability. Full probabilities are
preserved in the audit. Source mean subtraction also has minor floating
half-rounding artifacts; exact integer-difference deltas are supplied separately.
No multiplicity correction is applied. Shared rows across core/gate cells do not
provide independent replications. Two different suites do not isolate a causal
context-length effect. The existing RESULT's words "significantly worse/better"
violate its own restriction even though its final caveat says no significance
is claimed. Remove them.

The defensible observation is that this frozen query-selection procedure loses
to raw full context on answerable rows in both evaluated suites while beating
the random selection control. The missing-text/format confound remains in A−B
and A−D. Neither those contrasts nor answer-informed D justify claiming that
interference is ruled out, that D is a perfect-retrieval ceiling, or that the
model intrinsically lacks the remaining capability.

# MRS-2 — multi-reasoning scout protocol

**OPEN / DEVELOPMENT / SELECTION-EXPOSED FOREVER. Never confirmation, never release.**

MRS-2 is the successor instrument to MRS-1. MRS-1 is not amended, not reopened and not
re-run; its freeze, bank, evidence and negative result stand unchanged.

## Why a successor exists

MRS-1 executed natively on 2026-09-17. Both registered candidates passed
`NATIVE_QUALIFICATION` and then failed the frozen `SCHEMA_SMOKE` gate at **0/8 valid on
every role**: the family emits a bare in-vocabulary answer and terminates after two tokens
instead of the required JSON object. MRS-1 forbids prompt repair, and its posttrain contract
requires that a necessary incompatibility create a new experiment. MRS-1 therefore ends with
cells A and B **NOT PRODUCED**.

MRS-FP-1 (branch `research/mrs-format-probe-v1`) then measured five output contracts on the
8 permanently-open SMOKE items. `C1_BARE` was the only eligible contract, at 8/8 on every
gated role for both candidates. The probe also established the limitation precisely: this
family emits **exactly one field** and then stops. A prefill can choose the container; nothing
observed makes it emit a second field.

## What changed, and what did not

MRS-2 differs from MRS-1 in **one registered respect: the output contract.** Everything that
could bias a comparison is held fixed.

| Changed | Unchanged |
|---|---|
| Output contract is `C1_BARE`, bytes carried verbatim from the probe | Bank bytes, reused from MRS-1 by hash; never regenerated |
| Selector projection is answer-only | Item and world population, splits, and order |
| R2/R3 drop their unsatisfiable boolean-field sentences | Arm structure and the 11 unique calls per item |
| Smoke gate widened to all nine consumed roles | Call budget, token caps, and every resource guard |
| Constant-answer baseline reported with every arm | Decoding policy, native backend, statistics, oracle rule |

Role instructions are otherwise carried from MRS-1 verbatim.

## Roles

Every role emits one answer token and nothing else.

* **R0 — direct solver.** Solve normally.
* **R1 — alternative solver.** Work backward from the queried result, reconstruct supporting
  predecessor states or evidence links, eliminate inconsistent alternatives. A different
  computation, not a persona.
* **R2 — evidence/premise checker.** Audit which facts the query needs, including missing
  initial values, missing dependencies and superseding records; answer `UNKNOWN` if multiple
  answers remain possible. **Its abstention is its sufficiency verdict.**
* **R3 — consistency checker.** Rebuild the intermediate state table or earliest legal start
  times, check each transition, prerequisite, counterfactual deletion and latest-record rule,
  and recompute. **Its recomputed answer is its validity verdict.** It never sees gold.
* **D1/D2/D3 — repeated direct.** Independently registered rephrasings of the direct request.
  Decoding stays greedy; no sampling is introduced to manufacture diversity.
* **S1/S2 — selectors.** Choose one final answer from the original problem plus candidate
  records. Deployable procedures. They never receive gold, correctness labels, oracle output
  or any evaluation metadata.

**Registered reduction.** Under a single-field contract, S1 and S2 receive the same *shape*
of information and differ only in which roles supplied the candidates. MRS-1 intended S2 to
additionally see checker booleans; those are unobtainable here. The S1/S2 contrast is
therefore a contrast of candidate source, not of projected detail, and must be reported that
way.

## Arms

| Arm | Composition | Purpose |
|---|---|---|
| **A0** | R0, one call | practical baseline |
| **A1** | one direct call at the 320-token allowance | token-allowance control |
| **A2** | R0 + D1 + D2 + D3 → S1 | repeated direct attempts |
| **A3** | R0 + R1 + R2 + R3 → S2 | **primary treatment** |
| **A4** | analysis only | oracle candidate ceiling |
| **S0/S1/S2** | majority, answer-only selector, role selector | selector controls |

`A2` vs `A3` is the primary contrast: same number of complete model calls, different
computations. `A3` vs `A0` is the practical contrast.

**A1 caveat, registered in advance.** Token-budget matching is *vacuous* under a single-field
contract: the model terminates after the answer regardless of allowance. A1 is run to
demonstrate that, not to claim compute matching. No arm here is compute-matched, and none is
reported as such.

**A4 is never a treatment score.** The oracle reads gold only after generation, purely to
measure candidate-set headroom.

M2 sequential critique is **not dispatched**. Enabling it requires a new prospective budget,
not a silent extension.

## Population and order

The MRS-1 frozen bank, byte-identical: 120 items, 120 distinct worlds, four families
(`evidence_chain`, `register_machine`, `prerequisite_schedule`, `counterfactual_state`) at 30
each. Splits **SMOKE 8 / CAL 16 / CHECK 96**, processed in that order, in frozen
`public.jsonl` order within each partition.

* **SMOKE** gates format only; `correctness_inspected` is recorded false.
* **CAL** is descriptive. No selector is tuned on it — all selectors were frozen prospectively.
* **CHECK** (96 cases) is the scored population.

Gold lives in a separate file, is loaded only by the analysis module, and only after
generation.

## Gates, in order

1. Frozen-artifact and lineage verification; wrong branch or changed bytes fails closed.
2. Preload resource guards: 6 GiB host available, 6 GiB GPU free, 8 GiB disk. Not weakened.
3. One create-only `MODEL_LAUNCH` token per model. No silent retry.
4. `NATIVE_QUALIFICATION`: strict loading, cache-vs-full agreement, causality, deterministic
   repeat, tied storage, parameter inventory, state digest.
5. `SCHEMA_SMOKE` on 8 items, threshold 0.75 per role across all nine consumed roles. On
   failure: stop. **No prompt repair, no CAL, no CHECK.**
6. Generation over CAL then CHECK, with running guards and a 1800 s per-model ceiling.
7. Post-inference state digest must equal the pre-inference digest.

## Reporting requirements

Every arm is reported **alongside the constant-answer baseline** for its stratum. MRS-FP-1
observed a candidate that answered one fixed token for all items and thereby scored above
zero purely from the answer prior; without the baseline in view an accuracy number is
unreadable. Per-role degeneracy (distinct answers, modal share, constant flag) is reported
for the same reason. These are reporting requirements, not new pass/fail thresholds.

Repairs and damage are reported as raw counts as well as net gain. Treatment regressions are
never hidden inside an aggregate.

The MRS-1 `checker_errors` counters are structurally zero here because no checker booleans
exist. That is an absence of measurement, not evidence of reliable checking, and is labelled
as such.

## Statistics

The experimental unit is the task/scenario, never a generated token, and repeated model calls
are never pooled as independent rows. Paired exact counts, exact McNemar (descriptive,
exploratory, not multiplicity-adjusted), and world-clustered percentile bootstrap intervals
(2000 draws, seed 1978, descriptive 95%).

This is a **small open-development mechanism scout**. A positive result is not confirmation.

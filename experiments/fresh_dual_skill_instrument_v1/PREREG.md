# FRESH-DUAL-SKILL-001 — preregistration

Written **2026-09-22, before any arm was scored on this bank.** The bank file `T_BANK.json`
(sha256 `26c01effe9610230f7876bbe118792f85094605fe3dde5f873c78c67e1c493ba`) already existed when
this was written; no model had been run on it. Everything below is fixed from this point.

## Why this instrument exists

The peer probe `TITAN-OPALLO-SKILL-VECTOR-001` produced an uninterpretable functional read. Its
"S bank" — intended to measure the skill Titan-Sparse donates — contained no transition-contract
rows; rows S-01/02/05/07 are arithmetic state-tracking. The donor scored **0/8** on its own bank
while the *other* donor scored 3/8, and every cell sat between 0/8 and 3/8. Its scorer also used
substring matching (`norm(gold) in norm(output)`), so gold `5` is satisfied by output `15`.

The blocker was believed to be that no admissible fresh bank could exist, because the historical
banks are parent-contaminated against these donors and the confirmation instruments are spent.
That is false for this skill: all three transition families are **procedural and
oracle-verifiable**, so a fresh bank can be *generated* rather than authored.

## The instrument

192 rows, 64 from each recorded family: `ledger_replay`, `prerequisite_plan`, `register_machine`.
System prompts are the verbatim strings from `process-transition-probe-001`. Payloads are freshly
sampled; gold comes from `oracles.py`.

**Oracle validation.** `oracles.py` reproduces **384/384** of the recorded `reference` strings on
`process-transition-probe-001/CANDIDATE_VIEWS.jsonl` (128/128 in each family). The oracles were
written from the system prompts, not copied from any solver, so that is a genuine collision.

**Disjointness.** Every generated payload was checked against an exclusion set of **6,610**
transition payloads harvested from every `*.jsonl` under `rineya-dataset-retention-records` that
contains one — including all `microfit-train.jsonl` training corpora, `SELECT_VIEWS.jsonl`, and
the recorded probe itself. Keys are order-normalized, so key reordering cannot hide a duplicate.

**Scoring.** Exact string match of the stripped generation against `reference`. This is the rule
the recorded bank uses: on `process-transition-probe-001`, exact / stripped / JSON-equal agree at
234 / 234 / 234 for Titan and 232 / 232 / 232 for Titan-Sparse. No substring matching.

**Shortcut floor.** 6 of 192 rows have `answer == input state`, so a degenerate copy-the-input
policy scores exactly 6/192 (3.1%). Any score is read against that floor, not against 0.

## Build gates — decided now, before any arm is scored

The bank is **admissible only if all three pass.** Each can fail on achievable data.

| gate | requirement | why it can fail |
|---|---|---|
| **G1 base floor** | Supa scores **≤ 38/192** (20%) | Supa scores 6/384 (1.6%) on the recorded bank. If the fresh bank is much easier, Supa clears 20% and the bank is not measuring an *acquired* skill. |
| **G2 donor ceiling** | Titan-Sparse scores **≥ 77/192** (40%) | Titan-Sparse scores 232/384 (60.4%) on the recorded bank. This is the gate the peer's bank failed (donor 0/8). If the fresh rows are harder or off-distribution, the donor falls below 40% and the bank cannot measure preservation of its skill. |
| **G3 separation** | Titan-Sparse − Supa **≥ 58 rows** (30pp) | Requires a real, large donor effect to be present. |

If any gate fails, **the bank is rejected and no arm comparison is reported from it.** A failed
gate is a fact about this instrument, not a reason to retune the generator and retry — a
regenerated bank after seeing endpoint scores would be selection on the outcome.

## What will be measured, if the gates pass

Primary endpoint: transition exact-match count out of 192, per arm.

Arms: the nine already-materialized `TITAN-OPALLO-SKILL-VECTOR-001` arms — `BASE` (=Supa),
`SPARSE`, `OPALLO`, `AVG` (0.5, 0.5), `ADD-100` (1, 1), `ADD-S75`, `ADD-O75`, `ADD-75`,
`CONFLICT-LOCAL`. Note their `OPALLO` is the **unrounded FP32** reconstruction, not the published
BF16 core; it is a legitimate FP32 endpoint but is not the published artifact.

Registered comparison of interest — the attenuation-vs-interference decomposition:

- **Attenuation cost** = SPARSE − AVG. Both donors at half strength.
- **Interference cost** = SPARSE − ADD-100. Both donors at full strength, so any loss is
  composition, not attenuation.

Prediction registered now, from `ATTENUATION_CORRECTION_20260918` (235 → 223 → 219) and from
`DELTA-SUBSPACE-GEOMETRY-001` (alignment inside the contended subspace ≈ 0.012):
**attenuation cost > interference cost**, and ADD-100 should retain most of SPARSE's transition
score. If instead ADD-100 collapses while AVG holds, the near-orthogonality reading is wrong and
the subspace-collision finding was the operative one after all.

## Limits, stated now

- This is **one instrument on one skill**. It measures the transition contract only. It says
  nothing about arithmetic, long context, abstention, or general capability, and a transition
  score is not a quality or release claim.
- n = 192 with exact-match scoring. Differences of a handful of rows are not resolvable; only
  large effects are.
- Arithmetic (Opallo's donated skill) needs its own generated bank in the
  `percentage_reasoning` / `time_duration` / `ratio_proportion` format, with a numeric-extraction
  scorer that must itself be gated. It is **not** built here, so no joint-composition claim can
  be made from this instrument alone.
- "Matched construction is not matched difficulty." The fresh bank's difficulty is measured by
  the gates above, never assumed equal to the recorded bank's.
- No authorization is granted, amended, or implied by this file. `protocol.json` for
  SOUP-MINI-001 is untouched and its gates are unaffected; this is a separate exploratory
  instrument, not that experiment's CAL or CHECK.

---

# Amendment 1 — attempt 2, written before the v2 gate result was read

The v2 calibration job was launched before this amendment was written, but **its output had not
been read** when this text was committed. The stopping rule below is therefore a commitment, not
a description of a known outcome.

## What happened to attempt 1

v1 (`T_BANK.json`, sha256 `26c01eff…`) was calibrated and **failed G2**:

| gate | requirement | observed | |
|---|---|---|---|
| G1 base floor | Supa ≤ 38/192 | 2/192 | PASS |
| G2 donor ceiling | Titan-Sparse ≥ 77/192 | **63/192 (32.8%)** | **FAIL** |
| G3 separation | ≥ 58 rows | 61 | PASS |

Per-family, against the recorded bank's Titan-Sparse rates:

| family | recorded | v1 fresh | gap |
|---|---|---|---|
| `ledger_replay` | 41% | 44% | +3pp |
| `register_machine` | 65% | 38% | −27pp |
| `prerequisite_plan` | 75% | 17% | −58pp |

The bank was rejected. No arm comparison was computed from it and none will be.

## The repair, and why it is not goalpost-moving

The v1 generator was off-distribution on two families, measurable **from the recorded bank's
inputs alone**:

- `prerequisite_plan`: recorded `add` is exactly one bit in 128/128 rows, `delete` is 0 or 1 bit,
  and all 128 rows are applicable. v1 used multi-bit masks and made 20% inapplicable — a case the
  recorded bank never contains.
- `register_machine`: recorded `value` spans −8…11 (median 1). v1 sampled 0…96.

v2 resamples event parameters from the recorded **empirical input marginals**. No model output
entered the choice of parameters. `ledger_replay` was already matched and its generator is
unchanged in substance.

I state plainly what does leak: the *decision to repair* came from seeing G2 fail. The repair
targets are outcome-independent, but the trigger was not.

## Stopping rule — binding

1. **There will be no attempt 3.** If v2 fails G2, the instrument line is closed and reported as
   a failure. I will not adjust the generator, the gate, the row count, or the family mix again.
2. **The gates are unchanged**: G1 ≤ 38/192, G2 ≥ 77/192, G3 ≥ 58. They were set before any
   calibration and are not renegotiated now.
3. **If v2 passes**, the result is reported as *attempt 2 of 2*, never as a clean first pass, and
   any downstream arm comparison inherits that caveat.
4. v2's shortcut floor rose to 13/192 (6.8%) because matched `delete` marginals produce more
   no-op rows. Scores are read against 13, not 0.

## Registered prediction for v2, before reading

If the diagnosis is right, `ledger_replay` should stay near 41–44% and the other two families
should rise toward their recorded rates (65%, 75%). If instead `ledger_replay` also moves a lot,
the marginal-matching story is wrong and the difficulty gap is coming from something I have not
identified — in which case G2 passing would not be trustworthy even if it happens.

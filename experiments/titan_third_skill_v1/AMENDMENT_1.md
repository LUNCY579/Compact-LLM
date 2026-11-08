# TRISKILL-001 — Amendment 1 (committed before any Stage A or Stage B arm is scored)

**Status when written:** δ_C `train-001` is running in the native lane. No arm has been built or
scored, and no Stage A or Stage B row has been read. The only model outputs that exist are Phase 0's,
on DEV.

**Origin:** a read-only review by the model-soup session (the CONFIRM-002 lane), 2026-09-26. Every
point below was re-checked first-hand before adoption.

**Scope:** this amendment only **adds** requirements, descriptive endpoints and a wording rule. It
cannot turn a failing limb into a pass. PREREG §8 limited amendments to execution failures; that limit
is relaxed here, once, because the registration window is still open (no arm exists), and the change
tightens rather than loosens the design.

## 1. The C banks are clustered by world, so a world-level test is added

**Measured first-hand.**
- In C_FRESH, the full and distractor variants share the gold answer in **112 of 112** worlds.
- On Phase 0 DEV C, full-vs-distractor semantic correctness agrees in **0.84 / 0.91 / 0.91** of
  worlds (BASE / SPARSE / TT). Independence would give 0.75 / 0.71 / 0.71.

So rows within a world are positively correlated, and a row-level McNemar p is anticonservative.

**Added requirements.** The per-world net change is the sum over that world's JSON variants of
(arm correct − reference correct), on semantic correctness. The exact sign test is run over worlds
with a non-zero net.

| limb | as registered | added, and now **also required** |
|---|---|---|
| A1 (Stage A) | C100 − BASE ≥ +30 of 192, row McNemar p < 0.01 | exact **two-sided** sign test over the 64 DEV+CAL worlds, p < 0.01 |
| C3 (Stage B) | TTC − TT ≥ +34 of 336, row McNemar one-sided p < 0.05 | exact **one-sided** sign test over the 112 C_FRESH worlds, p < 0.05. **The world-level p is the one that enters Holm.** |
| C4 (Stage B) | upper bound over the 1,200 T+C rows < +5 | the bootstrap resamples T rows individually and **C_FRESH worlds as clusters** (10,000 resamples, seed 20260926) |

A2's g ≥ +20 stays row-level. It is a development eligibility screen, not a claim.

The exact sign test needs no calibration cell: its null distribution is exact under exchangeable
worlds.

## 2. The headline wording for content vs contract is fixed now

Phase 0 measured TT at **0 of 96** format-valid on C, against Supa's 47. The composite may therefore
acquire C's content while keeping TT's bare-answer convention.

Let f be the fraction of TTC(β*)'s semantic-correct C_FRESH rows that are also strict-correct (the
`{"answer": N}` contract held). The headline must use exactly one of these forms, chosen by rule:

- C1–C4 all pass **and f ≥ 0.90**: "a third skill arrives, with its output contract, with no
  measurable loss of the first two."
- C1–C4 all pass **and f < 0.90**: "C's content arrives with no measurable loss of the first two;
  its `{"answer": N}` output contract is overridden by Titan-Transition's bare-answer convention
  (f = …)."
- Any limb fails: the failing limb or limbs are named first. Then the content/contract statement
  follows, using f as descriptive.

f is also reported for C(β*) solo, which separates "δ_C never learned the contract" from "TT
overrides it".

## 3. Broad retention is added as a descriptive endpoint (no verdict)

A third donor raises the total update norm, and broad retention has tracked update norm across this
programme's arms.

**Added, descriptive only:** the n = 47 `references_retained` panel, measured through the recorded
retention harness (the path CLOCKFIX-001's retention leg used, with an HF-layout conforming receipt).
It is run for BASE, TT and TTC(β*) at Stage B.

**Canary:** BASE must reproduce its recorded 47/47. If it does not, the endpoint is reported as
"not measured" and nothing is inferred from it. It enters no rule, no β choice and no headline,
beyond one sentence reporting TT vs TTC(β*).

## Unchanged

Every bank, arm, coefficient, margin and Stage A/B rule not named above is unchanged. That includes
A2, A3's plateau pick, C1, C2, C3's +34-row floor, C4's +5 bound, the TTR null, and the sealed
`HSKILL-CONFIRM-002` staying unread.

# G1_QUALITY demonstrated on real arms — it catches D40, and it is badly calibrated

First execution of the `G1_QUALITY` gate (HOTFIX-003) on official weights rather than
synthetic fixtures. Implementation `7d48de21`, pinned CPU runtime, official
Supa 1.7.0-devfreeze. Parents are the freshly requalified receipts under
`outputs/ladder-7d48de21-v1/`. Open calibration and pilot splits only; no sealed or DEV
access.

## Result

| arm | verdict | parent |
|---|---|---|
| B28 frozen source | **PASS** 8/8 | `G0` |
| D40 ordinary width | **FAIL** | `G1_WIDTH` |

D40 is the whole point: it passes `G1_WIDTH`, passes the full `compare_paths` parity grid
with **zero** violations, and generates a degenerate `</think>` loop. Every pre-existing gate
accepted it. This one does not.

## The detail, and why it is uncomfortable

    G1_QUALITY_B28  PASS                    G1_QUALITY_D40  FAIL
      PASS ce_increase_within_margin          PASS ce_increase_within_margin   <-- MISSES IT
      PASS kl_to_source_within_margin         FAIL kl_to_source_within_margin
      PASS top1_agreement_above_floor         PASS top1_agreement_above_floor  <-- by 0.0032
      PASS generation_terminates              FAIL generation_terminates
      PASS generation_not_degenerate          PASS generation_not_degenerate   <-- MISSES IT
      PASS source_unchanged                   PASS source_unchanged
      PASS calibration_unchanged              PASS calibration_unchanged
      PASS pilot_unchanged                    PASS pilot_unchanged

| metric | limit | B28 | D40 |
|---|---|---:|---:|
| CE increase (nats) | ≤ 0.5 | +0.0000 | **−0.1308** |
| KL to source | ≤ 1.0 | 0.0000 | **1.4909** ✗ |
| top-1 agreement | ≥ 0.5 | 1.0000 | **0.5032** |
| termination rate | ≥ 0.8 | 1.000 | **0.750** ✗ |
| distinct outputs / 24 | ≥ 2 | 13 | **4** |

**At `7d48de21` D40 failed only two of the five quality checks; under HOTFIX-004 it fails
three** (`kl_to_source_within_margin`, `generation_terminates`, `generation_not_degenerate`). Three findings, all against my own design:

1. **A CE-only gate would have passed D40 outright.** Its CE is *better* than the source
   (−0.1308). This is the single strongest justification for the gate having five checks
   rather than one, and it is now demonstrated rather than argued.
2. **`top1_agreement_above_floor` passed by 0.0032.** D40 sits at 0.5032 against a 0.5000
   floor. HOTFIX-003 claimed "no plausible setting of these thresholds changes any arm's
   verdict" — **that claim is wrong for D40 on this check.** A floor of 0.51 would have
   failed it; 0.50 passes it. The overall D40 verdict is unaffected because two other checks
   fail, but the mitigation argument as written is overstated and is corrected here.
3. **`min_distinct_outputs: 2` is far too lax.** D40 emits a repeated `</think>` loop and
   still produces 4 distinct strings across 24 prompts, so the nondegeneracy check passes a
   model that is the textbook example of degeneracy. An absolute floor of 2 cannot work
   across different row counts; it should scale with the number of prompts.

## B28's pass is largely tautological

For `arm == 'B28'` the implementation sets `model = reference` and then `b = a`, so
`ce_increase`, `kl_to_source` and `top1_agreement` are **0.0000, 0.0000 and 1.0000 by
construction**, not by measurement. Only `generation_terminates` (1.000) and
`generation_not_degenerate` (13 distinct) are genuinely informative for the source arm.

That is defensible — the source is the reference, so comparing it to itself is definitionally
exact — but it means **B28 passing is not evidence the gate is discriminating.** All the
gate's discriminating power is on converted arms, where the comparison is real. The receipt
should not be read as "the gate validated the source".

## Recommended changes (not applied — they would change the digest again)

1. Scale the nondegeneracy floor with prompt count, e.g. `distinct >= max(2, ceil(rows/4))`.
   Under that rule D40's 4/24 fails, as it should.
2. Raise `min_top1_agreement` off the 0.5032 boundary, or drop the check in favour of KL,
   which it partly duplicates and which caught D40 cleanly at 1.4909 vs 1.0.
3. Amend HOTFIX-003's threshold-provenance paragraph: the claim that no plausible threshold
   changes any verdict is false on the top-1 check for D40 and must be narrowed to the
   *overall* verdict rather than the per-check outcome.

These are deferred rather than applied because each changes `implementation_manifest()` and
would invalidate the receipts just produced. They are recorded as the next queued change.

## Honest summary

The gate does the job it was built for — it is the only stage in the ladder that rejects a
structurally perfect model which cannot generate — but this first real run shows it is
**under-calibrated**, and it found that by catching D40 on two checks while three others
waved a degenerate model through. Unit tests on synthetic fixtures could not have surfaced
any of that.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
SRC=C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
E=experiments/architecture_benchmark_001
L=$E/outputs/ladder-7d48de21-v1
$PY -m experiments.architecture_benchmark_001 qualify-quality --source $SRC \
  --calibration $E/data/calibration.jsonl --parent $L/G0/attempt-0001/G0.json \
  --arm B28 --out $E/outputs/quality-7d48de21/B28 --device cpu
$PY -m experiments.architecture_benchmark_001 qualify-quality --source $SRC \
  --calibration $E/data/calibration.jsonl --parent $L/G1_WIDTH/attempt-0001/G1_WIDTH.json \
  --arm D40 --out $E/outputs/quality-7d48de21/D40 --device cpu    # exits non-zero by design
```

C28 and C40 cannot be quality-gated: their `G1_CED_*` parents hold FAIL receipts, so
`validate_receipt` refuses the stage. A failed construction cannot be quality-qualified, which
is the intended ordering.

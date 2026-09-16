# Dosing one family separately clears its floor and holds long context in every seed — and does not restore its rate

**RRP-014. Three runs at `ledger_replay=128, prerequisite_plan=64, register_machine=64`
(256 rows), seeds 17/23/29. Registered in `RRP014_ASYMMETRIC_DOSE_PREREG.md` before any
arm was built. Three of six predictions falsified.**

RRP-013 left two rungs and no clean win: D1 (384) acquires all three skills and fails
retention in 2 of 6 seeds; D0 (192) passes every gate and never once demonstrates
`ledger_replay`, because the registered gate has no acquisition term. The deficit was one
family in every seed — the one carrying the strongest shortcut.

So the dose was split by family. This arm gives `ledger_replay` D1's dose and leaves the
other two at D0's.

**It is an interpolation, not a new draw.** Doses are nested, and this was verified per
seed against the built corpora rather than assumed: in all three seeds A1's
`ledger_replay` parents are **exactly** D1's 128, both other families are **exactly**
D0's 64, and all 256 fall inside D1's 384 with none outside. A difference between A1 and
D1/D0 cannot be a sampling artifact.

## The result

| seed | new /384 | floors | panel | ret /47 | **gate** | long ctx | vs Supa (paired) | `UNKNOWN` /64 |
|---|---:|:--:|---:|---:|:--:|---:|---|---:|
| 17 | 213 | **3/3** | 72 | 43 | **PASS** | 157 | −4, p = 0.3438 n.s. | 12 |
| 23 | 208 | **3/3** | 77 | 42 | **PASS** | 158 | −3, p = 0.6900 n.s. | 12 |
| 29 | 204 | **3/3** | **61** | 38 | **FAIL** | 156 | −5, p = 0.3593 n.s. | **14** |
| *Supa* | *14* | *0/3* | *47* | *47* | — | *161* | — | *13* |

| | A1 (256) | D1 (384) | D0 (192) |
|---|---|---|---|
| all three floors cleared | **3/3** [0.292, 1.000] | **6/6** [0.541, 1.000] | **0/3** [0.000, 0.708] |
| gate | 2/3 [0.094, 0.992] | 4/6 [0.223, 0.957] | 3/3 [0.292, 1.000] |
| **long context n.s. vs Supa** | **3/3** [0.292, 1.000] | 4/6 [0.223, 0.957] | 2/3 [0.094, 0.992] |
| long-context range | **156–158 (spread 2)** | 147–160 (spread 13) | 153–155 (spread 2) |

## The two findings

**1. This is the only configuration measured in which long context is statistically
indistinguishable from the untrained source in every seed.** 157, 158, 156 — p = 0.3438,
0.6900, 0.3593 — a spread of two rows. D1 managed 4 of 6 with a 13-row spread; no
rehearsal arm managed any. It is also the only arm to hold abstention at or above Supa's
13 in a seed (14 at seed 29) while clearing every acquisition floor.

Paired at matched seeds against D1, A1 is better on retention and abstention in **all
three**:

| seed | A1 panel | D1 panel | A1 ret | D1 ret | A1 long ctx | D1 long ctx | A1 `UNK` | D1 `UNK` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 72 | 77 | **43** | 40 | 157 | 160 | **12** | 8 |
| 23 | **77** | 71 | **42** | 38 | 158 | 159 | **12** | 8 |
| 29 | 61 | 67 | **38** | 36 | **156** | 148 | **14** | 5 |

**2. Per-family dosing is not per-family independent.** `ledger_replay` received
*identical parents* in A1 and D1, and scored consistently lower:

| | `ledger_replay` rate (floor 0.2422) |
|---|---|
| A1, three seeds | 0.4766, 0.4531, 0.4375 — **all clear, all below D1's band** |
| D1, six seeds | 0.4844 – 0.6328 |
| D0, three seeds | 0.2812 – 0.3203, all below floor |

Cutting the *other two* families from 128 to 64 cost this family roughly 0.15 on a task
whose training rows did not change. So acquisition is partly a function of **total**
training volume, not only of a family's own dose. The lever works — the floor clears in
3/3 where it cleared in 0/3 — but it does not buy D1's rate.

This lands between the two outcomes registered: the strong falsifier (floor still fails at
128/family) did not occur, and neither did the clean per-family result.

**3. The gate failure moved from retention to the panel.** D1's two failures were both
`retained = 36`; A1's single failure is `panel = 61`, one under threshold, with retention
at 38 — comfortably passing. Across D1's six seeds the panel never fell below 64. The
gate's two terms are trading against each other as the dose composition changes, which
means neither term alone characterises the failure mode.

## Predictions versus outcome

1. *"All three floors clear in 3/3 seeds."* — **Held.** 3/3, and this was the prediction
   the experiment existed to test.
2. *"`ledger_replay` lands in D1's band (0.48–0.63), not between D0 and D1."* —
   **Falsified**, consistently: 0.4375–0.4766, all three below 0.4844.
3. *"Gate passes 3/3."* — **Falsified.** 2/3, and on the panel rather than the retention
   term I expected.
4. *"New skill totals 195–225."* — **Held.** 204–213.
5. *"Long context not repaired; ≥2 of 3 n.s."* — **Held, and beaten.** 3/3 n.s. It is
   still not *repaired* — every seed sits 3–5 rows below Supa — but it is the first arm
   where that gap is never significant.
6. *"Abstention lands 8–13."* — **Falsified upward.** 12–14; seed 29 exceeds Supa's own
   13. Note this endpoint is **not independent** of long context: the 64 unanswerable rows
   sit inside the same 256-row suite.

## The honest reading

**A1 is the best-behaved configuration this programme has produced, and it is 2/3 on the
gate.** That is the same headline sentence as D1's, and it is worth being precise about
what changed underneath it:

- Against D0, A1 is strictly better: it acquires all three skills, which D0 never does.
- Against D1, A1 is better on retention and abstention in every matched seed, has far more
  stable long context, and uses two-thirds of the data — at the cost of ~25 points of new
  skill and a lower `ledger_replay` rate.
- Three seeds cannot pin a pass rate. A1's gate interval is [0.094, 0.992]. RRP-013
  demonstrated six seeds could not pin one either.

**No configuration measured passes the gate reliably.** D1 is 4/6, A1 is 2/3, D0 is 3/3
only by not learning. The pattern across RRP-007 through RRP-014 is that every rung trades
somewhere, and the gate sees only part of the trade.

## What this does not license

No promotion, no release. `CONFIRMATION_INSTRUMENT_GAP.md` establishes that every
long-context and literal-`UNKNOWN` instrument here is spent, reserved, or locked — so the
strongest statement available about the long-context result above is that it was
**measured**, report-only by registration, on a suite marked `SPENT`. A 3/3
indistinguishability result is exactly the kind of finding that would need a fresh
instrument to mean anything, and `DATASET_SURVEY.md` §2 establishes none exists here.

**The gate was not amended to add an acquisition term**, even though RRP-013 exposed that
gap and this arm was designed around it. Closing it requires its own registration written
before the arm it judges; doing it here would be fitting the rule to the result that
revealed it.

## Reproduce

```bash
bash rineya-dataset-retention-records/run-asymmetric.sh
```

Artifacts `PROCESS_rel_a1s*.json`, `RETENTION_PROBE_rel_a1s*.json`,
`LONGCTX_rel_a1s*.json`. Tabulation, exact Clopper–Pearson intervals and exact two-sided
McNemar: `rineya-dataset-retention-records/dataset-survey-001/tabulate.py`.

# The gate cannot see acquisition, and the lower dose exploits that

**RRP-013. Nine training runs: D1 (384 rows) at six selection seeds, D0 (192 rows) at
three. Registered in `RRP013_RELIABILITY_PREREG.md` before any arm was built. Two of five
predictions falsified.**

The question was narrow: D1 was **2/3** on the gate, and 2/3 is not a pass rate. What is
D1's actual pass rate, and does a dose *below* 384 — never sampled — pass more reliably?

Both answers arrived. The second one is a trap.

## D1 at six seeds

| seed | new /384 | floors | panel | ret /47 | **gate** | long ctx | vs Supa (paired) | `UNKNOWN` /64 |
|---|---:|:--:|---:|---:|:--:|---:|---|---:|
| 17 | 239 | 3/3 | 77 | 40 | **PASS** | 160 | −1, p = 1.0000 n.s. | 8 |
| 23 | 217 | 3/3 | 71 | 38 | **PASS** | 159 | −2, p = 0.8450 n.s. | 8 |
| 29 | 236 | 3/3 | 67 | **36** | **FAIL** | 148 | −13, **p = 0.0351** | 5 |
| 41 | 227 | 3/3 | 72 | 39 | **PASS** | 147 | −14, **p = 0.0288** | 5 |
| 47 | 225 | 3/3 | 64 | 37 | **PASS** | 157 | −4, p = 0.5034 n.s. | 12 |
| 53 | 228 | 3/3 | 67 | **36** | **FAIL** | 154 | −7, p = 0.1435 n.s. | 12 |
| *Supa* | *14* | *0/3* | *47* | *47* | — | *161* | — | *13* |

**Gate 4/6 = 0.667, exact 95% CI [0.223, 0.957].** Registered prediction 1 — "D1 lands at
4/6 or 5/6" — held.

Three things the extra seeds bought:

- **Six seeds still cannot pin a pass rate.** Going from 3 to 6 moved the interval from
  [0.094, 0.992] to [0.223, 0.957]. The earlier "2/3" and this "4/6" are the same point
  estimate carrying genuinely different evidence, which is why the registration required
  k/n with an interval rather than a verdict. Narrowing the width below 0.3 would take
  roughly thirty seeds.
- **The gate is, for D1, a retention gate.** Both failures are `retained = 36`, one row
  under threshold. The panel never caused a failure: its minimum across six seeds was 64
  against a threshold of 62.
- **Acquisition never varies.** 6/6 seeds clear all three floors, `ledger_replay` at
  0.4844–0.6328 against its 0.2422 floor. What varies is retention, not learning.

## D0 at three seeds: passes every time, learns less every time

| seed | new /384 | floors | panel | ret /47 | **gate** | long ctx | vs Supa (paired) | `UNKNOWN` /64 |
|---|---:|:--:|---:|---:|:--:|---:|---|---:|
| 17 | 197 | **2/3** | 72 | **44** | **PASS** | 153 | −8, **p = 0.0386** | **13** |
| 23 | 183 | **2/3** | 69 | 38 | **PASS** | 153 | −8, p = 0.2153 n.s. | 10 |
| 29 | 178 | **2/3** | 62 | 38 | **PASS** | 155 | −6, p = 0.2101 n.s. | 12 |

**Gate 3/3 = 1.000, exact 95% CI [0.292, 1.000].**

**Floors: 0/3 = 0.000, exact 95% CI [0.000, 0.708].** Not one D0 seed acquired all three
skills.

The deficit is entirely one family, and it replicates in every seed:

| | `ledger_replay` rate (floor 0.2422, `copy_before`) |
|---|---|
| D1, six seeds | 0.6328, 0.5000, 0.5703, 0.5234, 0.4844, 0.5391 — **all clear** |
| D0, three seeds | **0.3203, 0.2812, 0.3047 — all below** |

`ledger_replay` carries by far the strongest shortcut of the three families: copying the
previous state answers 24% of it correctly, against 7% and 5% for the others. At
64 parents/family the model does not learn enough to be distinguishable from copying. At
128 it clears with room to spare, in every seed measured.

## The finding

**D0 passes the gate 3/3 while acquiring 0/3. D1 acquires 6/6 while passing 4/6.**

The pre-registered gate is `panel >= 62 AND retained >= 37`. It contains **no acquisition
term**. So a configuration that trains on less data passes it more reliably *because* it
learns less — less learning means less disturbance to the weights the panel and the
reference set depend on.

That is not D0 being better. It is the gate being blind to the thing the experiment exists
to produce. Read through the gate alone, D0 looks like a strict improvement over D1:
better retention (38–44 vs 36–40), better abstention (10–13 vs 5–12), tighter long context
(153–155 vs 147–160), and a perfect pass record. Read with the floors, D0 has failed to
demonstrate one of the three skills at all.

**The floors were published before any model call and the gate was registered before any
arm was built, so neither can be adjusted now to absorb this.** The correct conclusion is
that the gate was under-specified, and that is recorded rather than patched retroactively.

## Predictions versus outcome

1. *"D1 lands at 4/6 or 5/6."* — **Held.** 4/6.
2. *"D0 passes more often than D1."* — **Held** as stated (3/3 vs 4/6), but the intervals
   overlap almost entirely ([0.292, 1.000] against [0.223, 0.957]); with three seeds this
   is not established, and the reason it passes undercuts the value of it passing.
3. *"D0's new skill is 150–210, still clearing all three floors."* — **Falsified.** The
   score landed in band (178–197) and the floor claim was wrong in all three seeds.
4. *"D0's long context is at least as good as D1's; ≥2 of 3 seeds n.s."* — **Held.** 2/3
   n.s., and D0's range (153–155) has a better worst case than D1's (147–160) while having
   a lower median.
5. *"Abstention stays broken in every arm (5–8)."* — **Falsified.** D1 spans 5–12 and D0
   spans 10–13, including a seed at exactly Supa's 13. Abstention is seed-dependent, not
   uniformly destroyed. Note this endpoint is **not independent** of long context: the 64
   unanswerable rows sit inside the same 256-row suite.

**The registered falsifying outcome did not trigger.** I registered that if D0 also failed
roughly a third of its seeds, variance would be intrinsic to the draw and no rung would be
reliable. D0 passed 3/3, so dose *is* a lever on the gate — it simply buys the gate with
acquisition.

## A note on the paired test

D0 seeds 17 and 23 post **identical** long-context totals (153, −8) with p = 0.0386 and
p = 0.2153. McNemar is computed over discordant pairs, not over the margin: the same net
loss is significant or not depending on which rows moved. The totals column and the p
column are not two views of one number.

## What this determines about the next arm

The dose that acquisition needs and the dose that retention can afford **differ by
family**, and the knob has been global this whole time.

`select()` orders each family's pool by `digest([salt, "parent", parent_id])`, which does
not depend on the count. Doses are therefore **nested**, and this was verified against
artifacts already on disk rather than assumed: D0 seed 17's 192 parents are a strict subset
of D1 seed 17's 384 — 192 of 192 inside, 0 outside.

So an asymmetric dose of `ledger_replay=128, prerequisite_plan=64, register_machine=64`
(256 rows) selects **exactly D1's rows for the family that needs dose and exactly D0's rows
for the two that do not**. It is an interpolation between two measured arms using rows
already drawn, so any difference cannot be a sampling artifact.

## What this does not license

No promotion, no release. `CONFIRMATION_INSTRUMENT_GAP.md` establishes that every
long-context and literal-`UNKNOWN` instrument here is spent, reserved, or locked; the
long-context and abstention columns above are **report-only by registration**, and no arm
was selected on them. `DATASET_SURVEY.md` establishes that no corpus on this host can
rehearse long context, so the damage in both dose rungs is currently unaddressable.

## Reproduce

```bash
bash rineya-dataset-retention-records/run-reliability.sh
```

Artifacts `PROCESS_rel_*.json`, `RETENTION_PROBE_rel_*.json`, `LONGCTX_rel_*.json` under
`experiments/architecture_benchmark_001/evidence/retention-design-v1/`. Tabulation,
exact Clopper–Pearson intervals and exact two-sided McNemar:
`rineya-dataset-retention-records/dataset-survey-001/tabulate.py`.

# MQ-001 three-seed replication: consistent, and not significant

**Verdict: response-only supervision beats whole-sequence in 3 of 3 order seeds, the gain is
mostly not a polarity artifact this time, and the dependence-aware test still does not reach
p < 0.05. No objective claim is made.**

Six cells, all at commit `1e41cc5`, each bound to its own qualified plan.

---

## What ran

| | |
|---|---|
| Plans | `prepared-003/004/005`, order seeds 17 / 23 / 29, `plan_sha` `83347512` / `ba0333d5` / `d5c4c43f` |
| Same rows | all three hold the **same 238 training rows** (`selection_seed` 17); only the order differs — verified identical row sets, pairwise-distinct orders |
| Qualification | three separate PASS runs, 6/6 checks, 220 tensors / 314,618,880 params, `source_state_before == source_state_after` |
| Every cell | 13 updates, 24,910 processed tokens, finite losses and gradients, frozen backbone unchanged, DEV 64/64 terminated |

| seed | `whole_1e-5` | `response_1e-5` |
|---|---|---|
| 17 | 7/51 · respCE 2.3640 | **13/51** · **2.0195** |
| 23 | 7/51 · respCE 2.3775 | **13/51** · **2.0983** |
| 29 | 8/51 · respCE 2.3597 | **13/51** · **2.0861** |

Supervised targets: whole 24,672, response **1,944** — a 12.7× difference at identical
processed tokens.

---

## Positive

**The direction replicates perfectly.** Response-only wins all three seeds on exact match and
all three on response CE, and the response-CE ranges do not overlap (2.020–2.098 against
2.360–2.378).

**The gain is mostly NOT the polarity artifact that destroyed the previous result.**

| slice | whole → response, per seed |
|---|---|
| polar golds (8 rows, now balanced 4 YES / 4 NO) | +2 / +2 / +1 |
| **non-polar (43 rows)** | **+4 / +4 / +4** |

Last time the polarity cluster *was* the effect. Here the majority of the gain sits on
non-polar rows, identically in all three orders.

**The family-level pattern reproduces.** Aggregating rows within task family:

| family | s17 | s23 | s29 | all three same sign |
|---|---:|---:|---:|---|
| `sequence_pattern` | +2 | +2 | +2 | yes |
| `logical_implication` | +1 | +1 | +1 | yes |
| `evidence_inference` | +1 | +1 | +1 | yes |
| `unit_conversion` | +2 | +1 | 0 | |
| `constraint_selection` | 0 | +1 | +1 | |
| `contradiction_detection` | +1 | +1 | 0 | |
| `physical_commonsense` | 0 | 0 | +1 | |
| **`estimation_sanity`** | **−1** | **−1** | **−1** | yes |

**Order-seed variance is small here.** Whole spans 7–8 rows and 0.018 of response CE across
three orders. That contrasts sharply with RAB-001, where seed alone moved a 150-row panel by
15–24 discordant rows. Whatever else is true, this instrument is stable across order.

**The instrument fix did what it was supposed to.** The DEV baseline rose from 1/41 to 3/51
because punctuation no longer decides correctness, and an always-YES policy now scores 4/8
instead of sweeping.

---

## Negative

**It is not significant.** Aggregating within family and sign-testing across the 8 families
with a non-zero effect: 7 favour response, 1 favours whole, **two-sided p = 0.0703**.
Row-level per seed is 0.070 / 0.070 / 0.125 — also not significant.

**More seeds cannot fix this.** The three seeds share one 51-row DEV set, so they are repeated
measurements of one test set rather than new items. With three seeds the sign test *floor* is
p = 0.25. Power has to come from more DEV items or more families; running seeds 41, 47, 53
would add cost and no significance.

**The response arm still has a YES-bias.** It emits 8–11 YES predictions against a baseline of
0, and pays for it: `estimation_sanity` is −1 in every seed. The balanced DEV now *prices* that
bias instead of rewarding it, which is the point of the fix — but the bias is real and would
matter anywhere polar answers are not balanced.

**Response CE is close to definitional.** `response` trains exactly the objective that metric
measures, while `whole` spends gradient on prompt tokens. The 3/3 CE win is weaker evidence
than it looks and should not be quoted as independent confirmation.

**The evidence set is incomplete by the runbook's standard.** All six cells ran in-process via
`--_worker`, so `launch.json` and `supervisor.json` are absent and there was no watchdog. The
GPU lease, both memory guards, `binding.json`, frozen-backbone enforcement and checkpointing
all ran. Each cell records this in `EVIDENCE_NOTE.txt`.

---

## Two corrections to my own work

**A health criterion I pre-registered is invalid.** `MQ001_CELL_PREREG.md` fixed "training loss
lower at the end than at the first update". That compares *different rows* — each update's
`objective_ce` is computed over a different group. Seed 23's response arm went 2.224 → 2.895
while improving DEV by 10 rows and cutting response CE by 0.73. The valid progress signal is
DEV response CE, measured on the same 480 targets before and after; it improved in all six
cells.

**I under-read host memory by ~0.35 GiB for several turns.** My readiness probe imported
`resources` → torch and counted its own footprint against the budget, so I reported ~6.29 GiB
when the host actually had ~6.64. I gated too high on a number I had made worse myself.

---

## Defects found in the package along the way

1. **The in-process worker never creates its output directory** — `bounded_child` does. Run it
   directly and the first exception dies inside the error handler with `FileNotFoundError`
   writing `failure.json`, losing the real diagnosis. Worth fixing in `worker()`.
2. **The supervisor imports `run.py` at a measured 0.507 GiB** purely to validate a plan and
   fork a worker. On a 15 GiB host that is the difference between running and being refused.
3. **The `train` pre-load guard is marginally under its own need** — measured ~6.23 GiB against
   a 6.0 gate.

---

## Status of every claim

| claim | status |
|---|---|
| harness trains correctly end to end | **established**, 6/6 cells |
| response-only wins on this DEV at lr 1e-5 | **consistent across 3 orders, p = 0.0703, NOT significant** |
| the gain is mostly not polarity | **supported** — +4/+4/+4 non-polar vs +2/+2/+1 polar |
| response-only is better in general | **not established**, and not testable at this DEV size |
| anything about 3e-6 or 3e-5 | **unrun** — 4 of 6 registered cells never executed |

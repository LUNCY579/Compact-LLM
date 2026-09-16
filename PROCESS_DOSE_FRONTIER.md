# The dose frontier: more transitions keep helping, and keep costing more than they help

**RRP-008, four rungs, one seed each. Not a claim about any single rung.**

RRP-007 left one question sharp: the new skill replicated exactly across three
seeds while the retention gate passed twice and failed once, never retaining more
than the 37 minimum. This ladder asks whether forgetting is a function of dose.
Everything except the number of parents is held fixed.

## The frontier

| rung | rows | processed | `ledger` | `prereq` | `register` | **new skill** | **panel** | lost | **retained** | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| **D1** | 384 | 54k | 81/128 | 90/128 | 68/128 | **239/384** | **77** | 7 | **40** | **PASS** |
| D2 | 768 | 108k | 112/128 | 95/128 | 69/128 | 276/384 | 68 | 12 | 35 | **FAIL** |
| D3 | 1,536 | 217k | 122/128 | 92/128 | 75/128 | 289/384 | 71 | 10 | 37 | pass, exactly |
| D4 | 3,072 | 434k | 124/128 | 102/128 | 78/128 | **304/384** | 59 | 17 | **30** | **FAIL** |

Source baseline: 14/384 new skill, 47/150 panel. **Every rung clears all three
per-family floors**, including D1 at an eighth of D4's data.

## What the curve says

**The new skill rises monotonically and does not saturate.** 239 → 276 → 289 →
304. The gain flattens (+37, +13, +15) but never reverses. Registered prediction 1
said D4 "will not be much better than D3 on `ledger_replay`" — correct there
(122 → 124, nearly capped) — but wrong overall: `prerequisite_plan` 92 → 102 and
`register_machine` 75 → 78 still had room.

**Retention does not fall monotonically.** 40 → 35 → 37 → 30 is not a clean
slope, and the pre-registered analysis rule exists for exactly this: *"a difference
between rungs smaller than the seed spread is not read as a dose effect."* The
measured D3 seed spread is **panel 58–71, retained 35–37**. Against that:

- **D1 (panel 77, retained 40) is above the D3 spread on both** — a real effect.
- **D2 (68, 35) is inside the D3 spread on both** — not a dose effect. D2's "gate
  failure" is the same population as D3's passes; it is seed noise landing on the
  wrong side of a hard threshold.
- **D4 (59, retained 30) is below the spread on retention** — a real effect.

So the honest statement is **three points, not four**: low dose retains better,
high dose retains worse, and the middle is indistinguishable from noise.

Registered predictions 2 and 3 both hold: D4 fails the gate, and a lower rung
retains better while still clearing every floor.

## D4 is where degradation becomes visible, not just numerical

`ratio_proportion` goes **backwards** at D4 — 16/50 source → 13/50 arm, losing 9
of the source's answers — and `time_duration` drops a termination for the first
time anywhere in this programme (49/50). Every other rung terminates 50/50 in
every class. That is the shape of a model being pulled off its distribution, not
merely scoring lower.

## The recommendation, by the gate rather than by the headline

**D1: 384 rows, ~54k processed tokens.** It is the only rung that clears both arms
of the D3 gate with margin (panel 77 ≥ 62, retained 40 ≥ 37), it clears all three
transition floors, and it does so on **a quarter of D3's data and an eighth of
D4's**.

The cost is real and should be stated with the recommendation: D1 scores
**239/384** on the new skill against D4's 304. Choosing D1 buys 10 retained panel
answers for 65 transition answers. Whether that is the right trade depends on what
the core is for — but only D1 is defensible under the gate that was registered
before any of this was measured.

**The seed-1 result was an overshoot.** RRP-007 ran at D3 and passed retention by
exactly zero margin. A quarter of that data would have passed comfortably and
scored 239 instead of 289.

## Limits

Single seed per rung; the ladder characterises a curve and licenses no claim about
an individual point. D3's three-seed spread is the only variance estimate here and
it is carried onto every comparison above. The ladder stops at 1,024 parents per
family because `prerequisite_plan` has only 1,027 available after excluding the 384
scored parents — a data limit, not a choice. Retention is measured on one
150-row panel covering 3 of v3's 39 reply classes; "retained" means that panel, not
general capability. No rung is a release candidate.

## Reproduce

```bash
bash rineya-dataset-retention-records/run-dose-ladder.sh
```

Pre-registration with endpoints, four predictions and the analysis rules is in
`forge-v2-prereg-001/RRP008_DOSE_PREREG.md`, written before any rung was built.
Per-rung artifacts are `PROCESS_rrp008_D{1,2,4}.json` and
`RETENTION_PROBE_rrp008_D{1,2,4}.json`; D3 is RRP-007 seed 17 and was not re-run.
Checkpoints were deleted after scoring under `DISK_RECLAIM_001.md`; receipts,
plans, prepared corpora and all scores are retained.

**Sequencing note:** train every rung before scoring any rung. The RAB scorers
write evidence into the worktree and the RRP controller refuses to launch on a
dirty worktree, so a train→score→train loop refuses itself. See
`docs/LANE_INTERLEAVE_HAZARD.md`; it cost three launches here.

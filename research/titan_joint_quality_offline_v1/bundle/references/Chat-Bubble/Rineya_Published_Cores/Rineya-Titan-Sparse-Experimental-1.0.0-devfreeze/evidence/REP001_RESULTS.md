# REP-001 — the pruning effect replicates in direction 3/3, and it is a repair, not a bonus

**Two new independent training draws (seeds 23, 29) plus the seed-17 anchor. Every comparison
is paired within draw: both arms descend from the same training run, so only the mask
differs. Registered in `REP001_PREREG.md` before any corpus was drawn. Selects nothing,
promotes nothing; long context and abstention are report-only throughout.**

## The paired table

`ctrl` = the full FP32 update. `k25` = the top 25% of that update by |δ|, per tensor.

| seed | arm | acq /384 | floors | arith | strict | retained /47 | long ctx | paired p vs Supa | `UNKNOWN` |
|---|---|---:|:--:|---:|---:|---:|---:|---:|---:|
| 17 | ctrl | 235 | 3/3 | 71 | 64 | 40 | 157 | 0.5716 | 9 |
| 17 | **k25** | 232 | 3/3 | 72 | 63 | 41 | **164** | 0.6291 | 12 |
| 23 | ctrl | 229 | 3/3 | 71 | 56 | 38 | 152 | 0.1221 | 8 |
| 23 | **k25** | 228 | 3/3 | 66 | 55 | 39 | 152 | 0.0931 | 8 |
| 29 | ctrl | 228 | 3/3 | 65 | 51 | 36 | **144** | **0.0115** | 4 |
| 29 | **k25** | 230 | 3/3 | 68 | 57 | 40 | **159** | 0.8450 | 12 |

Paired deltas, k25 minus its own control:

| seed | acq | arith | strict | retained | long ctx | `UNKNOWN` |
|---|---:|---:|---:|---:|---:|---:|
| 17 | −3 | +1 | −1 | **+1** | **+7** | +3 |
| 23 | −1 | −5 | −1 | **+1** | **+0** | +0 |
| 29 | **+2** | **+3** | **+6** | **+4** | **+15** | **+8** |

All six arms matched: digest `a329eb7bb0`, transformers 4.51.3, `max_context` 4096, 0 rows
skipped. Clopper–Pearson bounds, floor verdicts and exact McNemar recomputed from raw
artifacts.

## Finding 1 — the direction replicates 3 of 3; the magnitude does not

Registered prediction 5 was that `retained ≥ control` **and** `long context ≥ control` in
2 of 2 new draws. **It held, 2/2 — and 3/3 including the anchor.** Pruning never once made
either endpoint worse.

But the magnitude ranges from **+0 to +15** long-context rows. So the promoted core's
specific headline (+7 at seed 17) is **not** a reliable expectation. The defensible
expectation is "never worse, sometimes substantially better", which is a weaker and more
honest claim than the single-seed result supported.

**A drafting weakness of mine, recorded:** predictions 2–4 were written as `≥`, so seed 23's
exactly-zero long-context delta satisfies them. The letter held where the substance was null.
A minimum effect size should have been registered, not just a direction.

## Finding 2 — pruning is a repair, and it pays most where the damage is worst

Order the draws by how much their control forgot:

| control long ctx | 157 (seed 17) | 152 (seed 23) | **144 (seed 29)** |
|---|---:|---:|---:|
| pruning's long-context gain | +7 | +0 | **+15** |

Seed 29's control is the **only arm in this programme with statistically detected
long-context damage**: 144/256, paired exact McNemar **p = 0.0115** against Supa's 161, with
abstention collapsed to 4/64. Pruning repairs it to 159/256 at **p = 0.8450** — no longer
detectable — and restores abstention to 12/64.

At seed 29 pruning is **strictly better on every single endpoint**: acquisition +2,
arithmetic +3, strict +6, retained +4, long context +15, abstention +8. There is no trade at
all in that draw.

So magnitude pruning behaves like **insurance against a bad draw** rather than a uniform
improvement. That is a more useful property than a fixed gain, because the failure mode it
protects against — a draw that quietly destroys long context and abstention — is exactly the
one the registered gate cannot see.

## Finding 3 — draws vary in what they forget, not in what they learn

| across the three controls | span |
|---|---|
| acquisition | 228–235, **span 7** |
| long context | 144–157, **span 13** |
| literal `UNKNOWN` | 4–9 |
| retained | 36–40 |
| arithmetic strict | 51–64, **span 13** |

Acquisition is stable; retention damage is not. Registered prediction 6 (control acquisition
span ≤ 40) held easily at 7.

This replicates RRP-013's conclusion — *"Acquisition never varies. What varies is retention,
not learning."* — on the new selection-admissible bank, with genuinely independent draws.
It also means **a single-seed retention number is close to uninformative**, which retroactively
justifies this screen and undercuts any single-draw retention claim in this programme,
including the ones in both promoted cores' records.

## Finding 4 — the retained ≥ 43 ceiling failed a third time

Registered prediction 7 held: no arm reached `retained ≥ 43` while clearing all three floors.
The best is 41 (seed 17 k25) and 40 (seed 29 k25). This corner has now failed on the α line,
the k line, and across three independent draws — it looks like a property of the recipe, not
of seed 17.

## Operational record

Seed 29's **first** training attempt failed with
`ValueError: native execution requires a clean checkout`. The cause was mine: the seed-23
batteries write their result files into the pilot clone, and the native training lane refuses
a dirty worktree. The refusal consumed no training tokens and did not breach the budget
(1 launch and 1 retry remained). The retry used the **same seed, plan and grant** with the
clone cleaned after every battery, which is a fixed environmental precondition rather than a
re-roll at a different setting. The failed attempt is retained at
`s29/journal/runs/train-001/failure.json`.

**Driver lesson:** any pipeline that interleaves batteries and training must clean the clone
between them. Batteries dirty it; training refuses it.

## What this changes

1. **Both promoted cores' retention figures are single-draw, and single-draw retention
   figures vary by 13 long-context rows.** Their acquisition figures are much more
   trustworthy than their retention figures. This belongs in their records.
2. **The k=0.25 promotion stands on replicated ground, in direction.** Pruning never hurt
   either preservation endpoint in three independent draws, and in the worst draw it was
   strictly dominant. But its expected gain should be quoted as a range, not as +7.
3. **Three draws still cannot pin a rate.** Three is better than one and nowhere near enough.
4. **The confirmation instrument remains the binding blocker.** Every long-context and
   abstention number here is report-only on a spent suite.
   `quality_improvement_established` stays false; `release_permitted` stays false.

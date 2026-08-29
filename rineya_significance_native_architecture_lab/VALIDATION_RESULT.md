# Held-out validation: CONFIRMED

Run: [`halo_stability192_validation_v1`](results/runs/halo_stability192_validation_v1)
Classification: `HELD_OUT_VALIDATION_PREREGISTERED_PLAN`. Halo 1.2.0.
768 case-conditions on the stability split — 192 cases, 48 scenario packs,
**zero scenario overlap** with the showcase split used for selection.

The analysis plan was written to
[`preregistered_plan.json`](results/runs/halo_stability192_validation_v1/preregistered_plan.json)
before any inference: one primary contrast, exact two-sided cluster sign test
over the 48 packs, confirm iff `p < 0.05`.

## Result

| Arm | stability-192 | showcase-64 | unanswerable | resident | persistent MB |
|---|---:|---:|---:|---:|---:|
| `B0` full cache | **86/192** (44.8%) | 30/64 | 5/48 | 1341 | 155.1 |
| **`POOL13_SINK48`** | **86/192** (44.8%) | 29/64 | **6/48** | **374** | **45.4** |
| `POOL7_SINK48` | 75/192 (39.1%) | 28/64 | 4/48 | 374 | 45.4 |
| `POOL1` = shipped B3 | 59/192 (30.7%) | 19/64 | 0/48 | 374 | 45.4 |

### Primary contrast — pre-declared

```text
POOL13_SINK48 vs POOL1
  packs   25 better / 6 worse / 17 tied  of 48
  cases   36W / 9L
  cluster exact two-sided p = 0.00088
  VERDICT: CONFIRMED
```

This is the first cluster-significant positive result in the program.

### Secondary — descriptive only

```text
POOL7_SINK48 vs POOL1      packs 17b/4w/27t   p = 0.00720
B0            vs POOL1     packs 24b/3w/21t   p = 0.00005
B0            vs POOL13_SINK48   packs 10b/9w/29t   p = 1.00000
```

The shipped mechanism is significantly worse than full cache (`p = 0.00005`),
with 27 fewer correct rows. `POOL13_SINK48` restores the **same total score** as
full cache, with the gap undetectable at 48-pack power (`p = 1.0`). It does not
recover exactly the same cases: B0 vs POOL13 has 10 better and nine worse packs.

## What was actually bought

```text
accuracy   86/192, identical to full cache
bytes      0.293x of full cache      (45.4 MB vs 155.1 MB)
wall       0.770x of full cache
abstention 0/48 -> 6/48, at or above full cache's 5/48
cost       two config integers: pool_width 13, sink_tokens 48
```

No training, no new mechanism, no additional persistent state. Both changes are
answer-blind. `pool_width` was already implemented at `runtime/policy.py:121-133`
and had never been enabled in any executed configuration; `sink_tokens` needed
to reach past token 38, where the abstention instruction sits.

## What this does NOT establish

**This is not a result about the Significance Matrix.** Score pooling is a
Cornerstone-2 span factor, explicitly excluded from Cornerstone-1 scope by
[CORNERSTONE1_ARCHITECTURE_HYPOTHESES.md](CORNERSTONE1_ARCHITECTURE_HYPOTHESES.md).
Sink coverage is positional. Every Cornerstone-1 finding stands unchanged:
`C` is arrival-chunk age (`N-044`), `S/C^e` is a recency prior (`N-045`),
the score is indistinguishable from a position counter (`N-046`) and from a
shuffle of itself (`N-054`).

**The decisive specificity control is complete (`N-073`).** Under the same
pooling/sink structure, real `S/C` scores 86/192, position-only 81/192,
eligible-pool-shuffled 71/192, and random 62/192. Real versus position is not
detected as different (`p = 1.0`), so the Significance Matrix score is **not
credited**. Real versus shuffled is also not significant (`p = 0.21004`); real
versus random is `p = 0.04329`. The supported gain is structural pooling plus
sink/prompt protection.

Other limits:

- One core. The [family evidence](PUBLISHED_FAMILY_REPORT.md) shows all five
  cores lose 10-13 answers with retention near 0.6, so transfer is plausible,
  but no core other than Halo has been run under pooling.
- One budget (256), one chunk schedule (256), one seed, CPU bf16.
- `pool_width=13` was selected on showcase-64 across 14 arms without
  multiplicity control. The held-out confirmation licenses the *configuration*;
  it does not mean 13 is optimal. The width trend was noisy (`POOL9` dipped).
- Wall-clock is local CPU accounting, not a deployment or GPU claim.
- Unanswerable accuracy remains poor in absolute terms (6/48) — restored to
  baseline, not fixed.

## Bearing on the rest of the program

Five screens attacked the ranking *function* — exponent, exposure definition,
aggregator, cohort normalisation, salience gating — and every one returned null.
One screen attacked the ranking's *spatial structure* and closed the entire
compression gap.

The corrected oracle predicted this: it showed 13 of 13 recoverable answers sat
in selection (`N-062`-`N-064`), with no residual read-side gap. That prediction
held on held-out data.

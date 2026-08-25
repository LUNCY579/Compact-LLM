# Halo showcase-64 selection-policy screen

Run: [`halo_showcase64_policy_screen_v1`](results/runs/halo_showcase64_policy_screen_v1)
Classification: `EXPLORATORY_SCREEN`, unregistered. Halo 1.2.0 only.

Six arms on the complete 64-case showcase split, 384 case-conditions. Every arm
holds prompt tokens, chunk boundaries, query access, decoding, and seed fixed
and varies only the selection policy. Execution reuses
`run_published_core_eval.execute_condition` unchanged.

## Motivation

The [showcase-64 evaluation](SHOWCASE64_EVAL_REPORT.md) found zero compression
loss on late-needle cases and heavy loss on early, middle, and distributed
needles. The [implementation-theory audit](IMPLEMENTATION_THEORY_AUDIT.md) then
established that `C` is arrival-chunk age and that `S/max(C,1)^e` acts as a
recency prior, shifting heavy-slot retention toward the newest cohort by
+12.0 percentage points. Both point at the same question: **is the content
signal doing anything the recency window does not already do?**

## Results

| Arm | heavy | recent | budget | `e` | Exact | early | middle | late | dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` full cache* | — | — | full | — | **30/64** | 3 | 6 | 10 | 11 |
| `B512` | 444 | 64 | 512 | 1.0 | 26/64 | 3 | 6 | 9 | 8 |
| `R192` | 60 | 192 | 256 | 1.0 | **25/64** | 4 | 2 | 9 | 10 |
| `B4_POS` position | 188 | 64 | 256 | — | 22/64 | 0 | 0 | 11 | 11 |
| `E20` | 188 | 64 | 256 | 2.0 | 21/64 | 2 | 4 | 9 | 6 |
| `B3`* | 188 | 64 | 256 | 1.0 | 19/64 | 1 | 3 | 10 | 5 |
| `B2`* | 188 | 64 | 256 | 0.0 | 17/64 | 2 | 2 | 10 | 3 |
| `E05` | 188 | 64 | 256 | 0.5 | 16/64 | 2 | 1 | 10 | 3 |
| `R008` | 244 | 8 | 256 | 1.0 | **14/64** | 1 | 1 | 7 | 5 |

\* reference arms from `halo_showcase64_eval_v1`, same protocol.

## Finding 1 — content significance does not beat pure recency

`B4_POS` replaces the significance score with absolute position at identical
budget, sinks, and recent window. It scored 22/64 against B3's 19/64.

```text
B4_POS vs B3   case 7W/4L p=0.5488   cluster 4better/3worse/9tied  p=1.0000
B4_POS vs B2   case 9W/4L p=0.2668   cluster 5better/4worse/7tied  p=1.0000
```

The raw three-answer edge does **not** survive cluster-level testing. The honest
statement is not "recency is better" but **"significance selection is
statistically indistinguishable from pure recency"**. That is still decisive for
the mechanism's value proposition: recency is free, while significance requires
scoring forwards, persistent `S` and `C` state, and matrix bytes. Measured
significance-matrix overhead on this run is about 1.5% of KV bytes on top of a
selection that a position counter reproduces.

`B4_POS` also shows the cleanest possible needle signature: **0 early, 0 middle,
11 late, 11 distributed**. Pure recency captures every distributed case B0 got
(11/11) while capturing nothing early or middle — exactly the behaviour of a
contiguous window.

## Finding 2 — the dose-response runs against the mechanism

Holding total budget at 256 and shifting slots between recency and
content selection:

```text
R008  recent=8    heavy=244    14/64
B3    recent=64   heavy=188    19/64
R192  recent=192  heavy=60     25/64
```

Accuracy rises monotonically as budget moves **away** from significance
selection. The extreme contrast is cluster-significant:

```text
R192 vs R008   case 12W/1L p=0.0034   cluster 8better/0worse/8tied  p=0.0078
```

Eight scenario packs improved, zero degraded. This is the strongest result in
the screen and the only contrast that survives cluster-level testing. Giving the
significance matrix more room to work makes Halo worse on this suite.

## Finding 3 — the exponent sweep approaches the positional limit

```text
e = 0.0  ->  17/64
e = 0.5  ->  16/64
e = 1.0  ->  19/64
e = 2.0  ->  21/64
position ->  22/64   (the e -> infinity limit)
```

As `e` grows, `C^e` dominates the ranking and the ordering converges on
newest-cohort-first — that is, on the positional policy. Accuracy improves along
that path and tops out at the positional arm. This independently confirms the
audit's conclusion: the exponent is a recency-prior strength dial, and the
mechanism performs best when tuned to discard its own content signal.

No individual exponent contrast is cluster-significant (`E20` vs `B2`,
`p=0.6875`); the value is in the direction of the trend, not any single step.

## Finding 4 — allocation buys more than capacity

```text
R192  budget 256   25/64
B512  budget 512   26/64
R192 vs B512   cluster 4better/5worse/7tied  p=1.0000
```

Recency-weighted allocation at 256 entries matches significance-weighted
allocation at 512 entries. On this suite, **reallocating a fixed budget is worth
roughly a doubling of it** — about a 2x storage efficiency difference, obtained
by using less of the mechanism.

`B512 vs B3` was 6 clusters better / 1 worse (`p=0.1250`), directionally
favouring more capacity but not cluster-significant.

## Interpretation limits

- Exploratory and unregistered; no frozen endpoint, no advance gate, and no
  multiplicity control across the arm family.
- Scenario packs hold four correlated variants, so the 16-pack contrast is the
  honest unit and is used for every claim above.
- One core, one suite, one chunk schedule, one budget family. `R192` is not a
  recommended deployment configuration; it is a probe that localises where the
  accuracy is coming from.
- Reference arms `B0`, `B2`, `B3` come from a separate run under the same
  protocol and are compared descriptively.
- None of this measures retention endpoints, only answers.

## What follows

The screen does not say attention-based KV retention is worthless — B3 still
holds 30% absolute accuracy at about a third of the KV bytes. It says the
*content-selection component as currently specified* is not earning its
storage against a positional baseline, and that the exponent dial improves
results only insofar as it approximates that baseline.

The [implementation-theory audit](IMPLEMENTATION_THEORY_AUDIT.md) identifies why
and lists the falsifiable repairs (P1-P6). The first of those to be tested is
cohort-relative ranking, in
[`halo_showcase64_cohort_ranking_v1`](results/runs/halo_showcase64_cohort_ranking_v1).

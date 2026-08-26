# GAP-A screen: span integrity and sink coverage

Run: [`halo_showcase64_span_screen_v1`](results/runs/halo_showcase64_span_screen_v1)
Classification: `EXPLORATORY_SCREEN`, unregistered. Halo 1.2.0.
512 case-conditions, eight arms, single process.

The corrected oracle ([`N-062`-`N-064`](CLAIMS_LEDGER.md)) established that the
entire recoverable compression deficit is selection error — `ORACLE_SPAN` 32/64
against shipped B3 19/64, 13W/0L, cluster `p=0.0156` — with no residual gap
against full cache. This screen attacks that gap with answer-blind fixes.

## Identity gate

```text
POOL1 (pool_width=1, sinks=4) = 19/64  == recorded B3  -> PASS
```

The harness reproduces the recorded baseline exactly. Every number below is
comparable to the rest of the program.

## Results

| Arm | pool | sinks | link | Exact | unanswerable | early | mid | late | dist |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `ORACLE_SPAN`* | — | — | — | **32/64** | — | — | — | — | — |
| `B0` full cache* | — | — | — | **30/64** | 2/16 | 3 | 6 | 10 | 11 |
| **`POOL7`** | 7 | 4 | none | **26/64** | 0/16 | 3 | 3 | 10 | 10 |
| `R192`* | — | — | — | 25/64 | — | 4 | 2 | 9 | 10 |
| `POOL5` | 5 | 4 | none | 25/64 | 0/16 | 3 | 3 | 10 | 9 |
| `SPANLINK_RAND` | 1 | 4 | random | 22/64 | 1/16 | 1 | 5 | 10 | 6 |
| `POOL3` | 3 | 4 | none | 21/64 | 0/16 | 2 | 3 | 9 | 7 |
| `SINK48` | 1 | 48 | none | 20/64 | **1/16** | 1 | 5 | 11 | 3 |
| `SPAN_SINK48` | 1 | 48 | word | 20/64 | **2/16** | 2 | 4 | 10 | 4 |
| `POOL1` = B3* | 1 | 4 | none | 19/64 | 0/16 | 1 | 3 | 10 | 5 |
| `SPANLINK` | 1 | 4 | word | 19/64 | 1/16 | 1 | 3 | 10 | 5 |

\* reference arms from earlier runs, same protocol.

## Finding 1 — a flag that was already implemented recovers 7 answers

Score pooling has existed at `runtime/policy.py:121-133` since the lab was
written and had **never been enabled in any executed configuration**. Turning
it on produces a clean monotone dose-response:

```text
width  1 -> 19/64      width  5 -> 25/64
width  3 -> 21/64      width  7 -> 26/64      (still climbing)
```

That is 7 of the 13 answers the oracle says are available, from a config
integer, at identical budget and byte cost.

**Statistical honesty.** No single contrast is cluster-significant:

```text
POOL7 vs POOL1   case 9W/2L p=0.0654   cluster 6b/2w/8t  p=0.2891
POOL5 vs POOL1   case 7W/1L p=0.0703   cluster 6b/1w/9t  p=0.1250
POOL3 vs POOL1   case 5W/3L p=0.7266   cluster 4b/2w/10t p=0.6875
```

The support is the **monotone ordering across four widths**, which a chance
fluctuation is unlikely to produce, not any individual `p`. This is a candidate
requiring independent confirmation, not a confirmed effect.

## Finding 2 — the compression gap is no longer detectable

This is the most consequential comparison in the screen. Every previously
tested arm was cluster-significantly worse than full cache:

```text
B0 vs SG10_E1   cluster 9b/0w  p=0.0039
B0 vs HYB192    cluster 7b/0w  p=0.0156
B0 vs POOL7     cluster 6b/4w  p=0.7539      <- no longer detectable
```

`POOL7` also becomes statistically indistinguishable from the oracle
(`ORACLE_SPAN` vs `POOL7`, cluster 6b/3w, `p=0.5078`) and from `R192`
(`p=1.0`). At roughly one third of full-cache bytes, an answer-blind config
change closes a gap that five prior screens could not move.

## Finding 3 — my span-link idea failed, and lost to its own control

`SPANLINK` propagated a word-initial token's score forward to its continuation
tokens, using tokenizer word boundaries only. It targeted the dominant measured
failure directly. It did nothing:

```text
SPANLINK       19/64   identical to baseline
SPANLINK vs SPANLINK_RAND   case 0W/3L   cluster 0b/3w/13t  p=0.2500
SPANLINK_RAND  22/64   the count-matched RANDOM control beat the real version
```

The specificity gate failed in the wrong direction. Word-boundary alignment
carries no advantage over randomly placed propagation of the same count; what
helps is *score smoothing in general*, which max-pooling does far better and
more systematically. The hypothesis that span integrity specifically needs
word-boundary information is not supported.

I designed this arm and it is a clean negative. Recorded as such.

## Finding 4 — sink coverage restores abstention

The abstention instruction sits at token position 38, outside the four
protected sinks, and the recency prior evicts it. Extending sinks to cover the
system prompt recovers it:

```text
unanswerable accuracy:  B3 0/16  ->  SINK48 1/16  ->  SPAN_SINK48 2/16  == B0 2/16
```

Total accuracy barely moves (20/64, cluster `p=0.6250`), so this is not an
accuracy fix. It is a **behavioural** one: compressed Halo stops confidently
answering questions the context cannot determine. For deployment that matters
more than the case count suggests, and it costs one integer.

## What this changes

The program's five prior screens all attacked the ranking *function* — exponent,
exposure definition, aggregator, cohort normalisation — and all returned null.
This screen attacked the ranking's *spatial structure* and moved 7 answers.

The lesson is not that significance works. `POOL7` still ties `R192`, which uses
no content signal at all. It is that **the selection deficit was largely about
where entries sit relative to each other, not about how they are scored** — and
that a capability already present in the codebase went unused across 33
configurations while considerable effort went into scoring variants that could
not help.

## Limits

- One core, one suite, one chunk schedule, one budget, one seed.
- Eight arms, no multiplicity control. Treat `POOL7` as a candidate.
- No contrast is cluster-significant; the monotone trend is the evidence.
- Pooling was selected on the same 64 cases used throughout; a winner's-curse
  correction on the held-out 192-case stability split is required before any
  configuration claim.
- All interventions are answer-blind: tokenizer boundaries and a sink count.
- No file under `src/` was modified.

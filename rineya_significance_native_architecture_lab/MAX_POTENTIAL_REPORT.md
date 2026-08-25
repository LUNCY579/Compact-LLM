# Cornerstone-1 maximum potential: a bounded answer

Run: [`halo_showcase64_oracle_bounds_v1`](results/runs/halo_showcase64_oracle_bounds_v1)
Classification: `ORACLE_BOUND_NOT_A_POLICY`. Halo 1.2.0, complete 64-case showcase split.

This closes the question the program has been circling: not "does the shipped
mechanism work" (it does not, `N-054`), but **how much could any version of it
possibly be worth?**

## 1. The dynamic range of KV content selection

Two answer-informed bounds fix the range. Both read the answer key during
inference by construction; they are bounds, never policies, and can never be
cited as Halo performance.

| Arm | Exact | early | middle | late | dist |
|---|---:|---:|---:|---:|---:|
| `B0` full cache | **30/64** | 3 | 6 | 10 | 11 |
| `ORACLE_KEEP` answer tokens first | **25/64** | 2 | 8 | 10 | 5 |
| `R192` positional-heavy, no oracle | **25/64** | 4 | 2 | 9 | 10 |
| `B4_POS` position | 22/64 | 0 | 0 | 11 | 11 |
| `SUM_SHUF` shuffled scores | 21/64 | 0 | 1 | 10 | 10 |
| `MAX_E0` | 20/64 | 1 | 2 | 11 | 6 |
| `B3` shipped | 19/64 | 1 | 3 | 10 | 5 |
| `RANDOM` uniform ranking | 17/64 | 0 | 1 | 9 | 7 |
| `B2` | 17/64 | 2 | 2 | 10 | 3 |
| `ANTI_ORACLE` answer tokens last | **14/64** | 2 | 0 | 10 | 2 |

```text
ORACLE_KEEP vs ANTI_ORACLE   case 11W/0L p=0.0010   cluster 5b/0w/11t p=0.0625
```

Selection is not irrelevant. Moving from worst-case to best-case content
selection at a fixed budget is worth **11 answers out of 64**, with zero
case-level losses. That is the entire space any ranking function — significance
matrix included — is competing inside.

## 2. The decisive result: the oracle ceiling is already matched without content

```text
ORACLE_KEEP  25/64      R192  25/64
ORACLE_KEEP vs R192   case 8W/8L p=1.0000   cluster 4b/5w/7t p=1.0000
```

`R192` is pure structure: budget 256, sinks 4, **recent 192**, only 60
content-selected slots, no oracle, no answer information of any kind. It ties
perfect answer-aware selection exactly.

**At this budget and schedule, the maximum achievable by content selection is
already reachable with no content signal at all, by reallocating budget to the
recency window.** That is the ceiling on the Significance Matrix's potential
here, and it is not above what a positional policy already delivers.

The two reach 25 by different routes — `ORACLE_KEEP` recovers middle needles
(8, above even full cache's 6) while `R192` recovers distributed ones (10 vs 5).
The routes differ; the destination does not.

Meanwhile the shipped mechanism is statistically indistinguishable from ranking
at random:

```text
B3 19/64 vs RANDOM 17/64   case 7W/5L p=0.7744   cluster 4b/3w/9t p=1.0000
```

## 3. The signal exists — it just cannot be converted

This is the part that argues *against* dismissing the idea outright.

Measuring whether `S` ranks answer-bearing tokens above the rest of the
eligible pool, with a **frequency-matched** negative class (non-answer tokens of
equal rarity, so the result is not a rare-token artifact):

| Comparison | head-mean AUC | best head per cell |
|---|---:|---:|
| answer vs all eligible (rarity confounded) | 0.5523 | 0.7211 |
| **answer vs frequency-matched non-answer** | **0.5412** | **0.7178** |

`S` carries real, frequency-controlled information about where the answer lives.
In the best KV head of each layer-boundary cell it reaches AUC **0.72**, and
557 of 1036 head-mean cells are above chance.

So the mechanism's premise is not false. Cumulative attention does partly know
where the evidence is. The failure is downstream: that signal is spread across
16 heads of which roughly half are uninformative, every head must keep the same
number of entries, and the pooled ranking that results performs no better than
its own shuffle.

**A correction to the record.** My first pass at this measurement returned
AUC 0.43 — apparently anti-correlated — because answer strings were matched
against common sub-word tokens, so hundreds of filler positions were labelled
as answer-bearing. Restricting to rare sub-tokens and adding the
frequency-matched control reversed it. The 0.43 figure was an artifact and
should not be used.

## 4. Axis-by-axis potential

**KV retention** — the mechanism's home ground. Bounded above at 25/64 by
`ORACLE_KEEP`, which is matched by a no-information positional policy. Headroom
over the shipped mechanism exists (19 → 25) but is fully capturable without any
significance signal.

**Layers.** No layer is individually decisive; head-mean AUC ranges narrowly and
the useful signal is concentrated in *heads*, not layers — best-head AUC 0.72
against a head-mean of 0.54. If anything is worth pursuing it is head
selection, not layer selection. Note that head-wise adaptive allocation is
established prior art (Ada-KV, LAVa) and the lab's own Experiment B already
failed its specificity gate on entropy-based allocation.

**Inference time.** Compression is a genuine win here and is the mechanism's
most defensible property:

```text
B3 vs B0   wall clock 0.853x      persistent bytes 0.331x
significance-matrix overhead      1.56% of its own KV
extra scoring forwards            4 per case (one per boundary)
```

67% less persistent state and 15% less wall clock. But note this is a property
of *bounded KV* generally, not of significance — `R192` obtains the same
savings with better accuracy.

**Attention.** Already tested and closed: Experiment C's bounded historical
attention prior scored 5/12 in every arm and failed specificity (`N-036`).

**Embedding and vocabulary.** Out of mechanism scope, and I will not invent a
result here. The Significance Matrix operates on post-RoPE key/value cache
entries. It has no defined interaction with the embedding table or vocabulary,
and extending it there would not be the same mechanism.

**Model-agnosticism.** The runtime attaches as a Transformers `Cache` subclass
plus an SDPA attention-interface patch, so it is architecture-general *in
design* and would attach to any model using that interface. Empirically it is
gated to Qwen3 28L/1024/16Q/8KV/128/151936 geometry, and answer-level evidence
exists for exactly one core. Aegis (19 layers) and R2 (32k vocabulary) are
excluded by geometry checks. `N-009` remains `NOT_RUN`.

## 5. The honest bottom line

At this budget, schedule, core, and suite:

- selection is worth at most **11 answers of 64** (`ANTI_ORACLE` 14 → `ORACLE_KEEP` 25);
- the top of that range is **already reachable without any content signal** (`R192` 25);
- the shipped mechanism sits at 19, **indistinguishable from random ranking**;
- `S` nonetheless contains genuine answer-location signal (best-head AUC 0.72)
  that the current pooling fails to convert;
- and full cache (30) stays above every compressed arm.

The maximum potential of Cornerstone-1 *as a retention-ranking mechanism* is
therefore bounded and, at this operating point, already matched by a simpler
positional policy. The one place a real effect might still live is
**head-selective use of the score**, where the measured signal is strongest and
where nothing in this program has yet been tested with adequate power.

## Limits

- One core, one suite, one chunk schedule, one budget family, one seed.
- `ORACLE_KEEP` is a **lower** bound on the true oracle: it locates answers by
  rare sub-token match and found positions in 44 of 64 cases, so paraphrased or
  multi-hop evidence is missed. On the remaining 20 cases it degenerates to
  ranking by `S`.
- Bounds are exploratory and unregistered; no frozen endpoint, no advance gate,
  no multiplicity control across the five screens now run.
- Scenario packs hold four correlated variants; all claims use the 16-pack unit.
- AUC measurements used 16 answerable cases, layer-boundary cells pooled.
- No file under `src/` was modified in any screen.

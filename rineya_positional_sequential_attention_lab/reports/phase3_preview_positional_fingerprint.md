# Phase 3 (preview) — positional fingerprints, and choosing the primary metric

> **RETRACTED IN PART — see §6.** Four claims in this preview were overturned by a later
> audit: the stated sample size, the location of the worst position, the sign of the
> quadratic coefficient, and the strength of the margin claim. The metric decision this
> report exists to make (summed gold NLL) stands. Read §6 before quoting anything here.

**Status: preview, not the Phase 3 result.** One length bucket (4K), one split
(calibration), `local_position` fixed to `middle`, n = 33 items per model × position,
teacher-forced only, no confidence intervals, no paired bootstrap. Its purpose is to
settle the metric question before committing to the full run, and it does. The
numbers below will be superseded by `scripts/phase3_baseline_fingerprint.py`.

All four models are scored on **byte-identical token sequences** (Phase 1 §1), so
every cross-model comparison here is paired at the item level.

## 1. The metric question, settled

The Phase 2b calibration made gold NLL the primary metric. This run asked a follow-up:
*mean-per-token* NLL looked suspiciously insensitive, so the answer was decomposed
into its first token and the rest.

Pooled over positions, `exact_retrieval`:

| model | NLL, first answer token | NLL, remaining tokens |
| --- | ---: | ---: |
| qwen3 | 4.418 | **0.002** |
| supa_a11 | 0.229 | **0.003** |

Once the first token is given, the rest of a verbatim-copied identifier is free — an
induction head finishes the string with near-certainty. So for `exact_retrieval`,
mean-per-token NLL divides the entire retrieval signal by the answer length.

The obvious conclusion — "use first-token NLL" — is **wrong in general**. On
`semantic_single_hop` the first answer token is the article *the*, which carries no
retrieval information at all, and first-token NLL is flat to three decimal places
across every position (qwen3: 4.21–4.60 over the whole grid) while the summed NLL
varies by 4.3 nats. Each metric is blind on exactly the family the other resolves.

**Decision: the primary metric is the summed gold NLL over the answer**, which is
informative for both families. Within a scenario the answer — and therefore its token
count — is identical across positional variants, so summed and mean NLL differ only by
a constant factor and give identical *paired* comparisons; summing simply stops the
reported effect sizes from being silently divided by answer length. First-token NLL
and first-token margin are retained as secondaries, and for `exact_retrieval` the
summed and first-token columns coincide, which is itself the evidence that the tail is
free.

## 2. Lost-in-the-middle exists, but it is not a symmetric U

Summed gold NLL by evidence position (lower is better), `exact_retrieval`:

| p | 0.02 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 0.98 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **qwen3** | 4.39 | 8.71 | **9.32** | 7.91 | 7.21 | 4.50 | **1.30** |
| **halo** | 0.04 | 1.35 | **3.17** | 0.51 | 1.03 | 0.77 | **0.005** |
| **nimbus** | 0.31 | 1.78 | **3.50** | 1.22 | 1.19 | 1.13 | **0.012** |
| **supa_a11** | 0.02 | **2.82** | 0.31 | 0.005 | 0.005 | 0.015 | 0.006 |

Qwen3 degrades in the middle, so H1 survives — but the shape is **asymmetric**. The
worst position is p = 0.33, not p = 0.50, and the end is far better than the
beginning (1.30 vs 4.39). That is a strong recency preference with an early-middle
trough, not the classic symmetric U. A quadratic fit will report β₂ > 0 *and* β₁ < 0
here, and reporting only β₂ would hide the larger of the two effects.

**Each model's worst position is different**: qwen3 and halo/nimbus at p = 0.33,
supa_a11 at p = 0.17. "Worst-position accuracy" is therefore not a comparison at a
fixed p, and any intervention tuned at one model's trough is being tuned somewhere
else on another's curve.

## 3. Margin: Qwen3 cannot do this task greedily at all

Margin at the first answer token (gold logprob − best non-gold; > 0 means greedy
decoding emits the gold token):

| p | 0.02 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 0.98 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3 | −2.99 | −7.78 | −8.65 | −7.13 | −6.47 | −2.60 | **+2.03** |
| halo | +5.90 | +0.67 | −0.36 | +2.69 | +1.98 | +2.63 | +8.27 |
| nimbus | +4.31 | −0.17 | −1.38 | +1.33 | +1.58 | +1.81 | +7.21 |
| supa_a11 | +6.37 | −1.55 | +5.41 | +6.98 | +7.02 | +6.28 | +7.10 |

Qwen3's margin is negative at six of seven positions: it would not emit the correct
identifier anywhere except at the very end of the context. This is the mechanism
behind the near-zero exact-match accuracy in Phase 2b, and it is why accuracy could
not resolve position — the model is below the decision threshold almost everywhere,
so an accuracy metric sees a flat floor while NLL sees a 9-nat range.

## 4. A dissociation between the two families

The direction of the Rineya family's advantage **reverses** between families.
Summed gold NLL at p = 0.02:

| family | qwen3 | halo | nimbus | supa_a11 |
| --- | ---: | ---: | ---: | ---: |
| `exact_retrieval` | 4.39 | **0.04** | **0.31** | **0.02** |
| `semantic_single_hop` | **9.89** | 13.65 | 12.61 | **7.45** |

Halo and Nimbus are one to two orders of magnitude better than Qwen3 at exact
identifier retrieval and *worse* than Qwen3 at semantic single-hop, at every position.
Supa-A1.1 is the only model better than Qwen3 at both.

This matters for Phase 9's model selection. Picking the "best long-context model" on
`exact_retrieval` alone would pick Halo or Supa; on `semantic_single_hop` it would
pick Supa or Qwen3. The selection criterion has to be stated over both families
before the numbers are looked at, and the dissociation itself is a result worth
carrying into the final report — the Halo/Nimbus branch appears to have traded
semantic retrieval for verbatim retrieval.

Also note that on `semantic_single_hop` the margin is negative for qwen3, halo and
nimbus at **every** position and roughly zero for supa_a11, so that family is near a
capability floor for greedy decoding across the panel. NLL still resolves it; accuracy
will not.

## 5. A confound caught in this run's own first pass

The first version of this analysis pooled all `natural`/4K items per position. That
slice is not balanced: the `local_position_sweep` grid emits only p ∈ {0.02, 0.50,
0.98}, so those three positions carried 33 middle + 21 first + 21 last items while the
other four carried 33 middle-only. Local rhetorical position is a *separate factor*
under test (brief §8), and mixing it into the global-position axis puts a spurious
step at exactly the three positions that define the edge-middle gap. Every number
above is restricted to `local_position == "middle"`. The full Phase 3 run must either
hold local position fixed or model it explicitly; it may not pool.


---

## 6. Retractions

A later adversarial audit checked every number in this preview against `pilot5.csv`.
Four claims did not survive. All four are my errors, and all four are the kind that a
"preview" label does not excuse, because each was stated as a fact rather than as a
direction.

**1. The sample size is wrong.** This report says "n = 33 items per model × position".
The real per-cell n is **11** for `exact_retrieval` and **22** for `semantic_single_hop`;
33 is their *sum*, and every table here is per-family. Each conclusion below rests on
three times less data than claimed.

**2. "The worst position is p = 0.33, not p = 0.50" is not supported at this n.** Paired
per-scenario tests on the same data: 0.33 − 0.17 = +0.615 ± 0.949 (p = 0.53), and
0.33 − 0.50 = +1.411 ± 0.667 (p = 0.061, CI [−0.08, +2.90], straddling zero). The *modal*
per-scenario worst position for Qwen3 is actually **0.17** (5/11 scenarios) not 0.33
(4/11). The related claim that "each model's worst position is different" is likewise
unsupported — only Supa's argmax is reliably distinguishable.

**3. The quadratic-coefficient claim has the sign backwards.** This report says "a
quadratic fit will report β₂ > 0 *and* β₁ < 0". Running the lab's own
`metrics.ushape_fit` on this report's own Qwen3 row gives **β₁ = −4.389 (se 1.034),
β₂ = −24.627 (se 3.816)**, R² = 0.937. β₂ is *negative*. The sign convention is the trap:
`ushape_fit`'s β₂ > 0 signature is defined for a higher-is-better metric, and summed NLL
is lower-is-better, so an inverted-U in NLL gives β₂ < 0. The underlying observation — a
recency gradient dominating a curvature term — is right; the coefficient statement is not.

**4. The margin claim is too strong.** "Qwen3 would not emit the correct identifier
anywhere except at the very end" overstates a mean. Per item, Qwen3 greedily emits the
gold token on **21 of 77 items (27 %), spread across five of the seven positions** — 4/11
at p = 0.02, 6/11 at p = 0.83 (median margin **+0.50**, i.e. positive), 8/11 at p = 0.98.
The *mean* margin is negative at six of seven positions, which is what the table shows;
"would not emit it anywhere" does not follow from a negative mean over a skewed
distribution.

**What stands.** The metric decision — summed gold NLL as primary, because accuracy is at
floor — is unaffected by all four and is confirmed by the full Phase 3 run. The
dilution finding (first-token vs rest) also stands, and in fact turned out to matter more
than this report realised: see the withdrawal in
[`phase3_positional_fingerprint.md`](phase3_positional_fingerprint.md) §3.

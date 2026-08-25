# Cornerstone-1 implementation-theory audit

Scope: does the runtime implement the stated Cornerstone-1 theory, and does
that theory carry the information it claims to carry?

Verdict in one line: **the implementation is faithful to the stated theory, but
the theory as specified is degenerate — the realized-exposure counter `C` is
mathematically identical to arrival-chunk age, so `S/max(C,1)^e` cannot express
anything beyond a per-cohort age discount.**

This is an implementation-theory finding, not a coding defect. Nothing below
reports a bug in the sense of code failing to do what it was written to do.

## 1. What is correct

These were read and, where noted, checked against live Halo execution.

| Property | Status | Evidence |
|---|---|---|
| `(K, V, S, C, position, entry_id)` stay aligned through gather | correct | `atomic_gather` gathers every field with one validated index set; `_validate_indices` rejects duplicates, inactive references, and non-trailing padding |
| Inactive-slot hygiene | correct | `validate()` forces `-1` metadata and zero score/count on inactive slots, and rejects duplicate active `entry_id` per layer/head |
| GQA reduction | correct | `reduce_gqa_attention` reshapes `[b, Hq, q, n]` to contiguous KV groups; Qwen3's 16Q/8KV mapping is explicit |
| Post-RoPE scoring | correct | scoring uses post-RoPE `q` and pre-`repeat_kv` `k`, float32, with bottom-right causal masking; parity `exact_tensor_equal` |
| Score/exposure separation for controls | good design | `compress(ranking_counts=...)` changes only the ranking denominator; persistent `C` still gathers with K/V, so a one-boundary perturbation cannot contaminate later boundaries |
| Fail-loud on unscored entries | good design | compression raises if any `C <= 0`, explicitly to stop a missing-significance run silently degrading into a positional policy |
| Query-probe rollback | correct | verified exactly in the recorded family smokes |
| Source-primitive parity | 4/4 exact | `results/source_parity.json` |

The alignment discipline in particular is stronger than most research code. The
audit below is not a criticism of the engineering.

## 2. The central finding: `C` is arrival-chunk age

`TorchSignificanceState.update` is documented as accumulating "one scoring
event over every currently resident entry", and does exactly that:

```python
exposure_tensor = torch.full_like(self.counts, float(exposure))   # exposure = 1.0
self.counts = (self.counts + exposure_tensor).contiguous()
```

Every live call site supplies the default scalar:

```
runtime/cache.py:548            self.layers[layer_idx].record(contribution)
runtime/heterogeneous_cache.py:762   self.layers[layer_idx].record(contribution)
```

Neither passes an `exposure` argument. Combined with exactly one scoring event
per non-final chunk boundary, this gives, for an entry admitted in chunk `k`
evaluated at boundary `b`:

```text
C_i = b - k + 1
```

`C` is therefore a deterministic step function of arrival chunk. Measured on a
real Halo showcase case by instrumenting the compression callback:

```text
boundary 0: 256 entries, distinct C = [1]
boundary 1: 512 entries, distinct C = [1, 2]     positions 256-511 -> 1, 0-255 -> 2
boundary 2: 512 entries, distinct C = [1, 2, 3]
boundary 3: 512 entries, distinct C = [1, 2, 3, 4]
```

`C` was a deterministic function of absolute position at every boundary. Across
the whole 64-case run, `C` can take **at most 5 distinct values for a median of
397 resident entries**.

### Corollaries

1. **Within a cohort, `e` changes nothing.** For entries `i`, `j` admitted in
   the same chunk, `C_i = C_j`, so `S_i/C_i^e > S_j/C_j^e` iff `S_i > S_j`. The
   exponent can only reorder entries *across* cohorts.
2. **`S/max(C,1)^e` is a per-cohort constant rescaling** — a discrete age
   discount with as many levels as there have been boundaries.
3. **Ledger row `N-020` is a theorem, not an experimental result.** "Age-derived
   and head-averaged counts exactly reproduced true `C` in 11/11 clusters"
   *must* hold, because `C` is age by construction. That row should be restated
   as an identity confirmed empirically, not as evidence weighing on the
   hypothesis.
4. **Hypothesis H2 is not testable in this regime.** H2 asks whether realized
   exposure "contains information beyond age". As implemented it is *defined* as
   age, so the question is vacuous until exposure is made non-uniform.

## 3. Measured direction: the normalization is a recency prior

Normalization is not inert — it changes real selections. Measured over 224
layer-0 head-level selections on eight showcase cases, the Jaccard overlap
between the `e=0` and `e=1` retained sets was mean `0.818`, and **no** selection
was identical. So roughly 18% of retained entries differ.

The direction is the problem. Heavy-slot retention by cohort age:

| `C` (1 = newest) | `e=0` raw `S` | `e=1` `S/C` | change |
|---:|---:|---:|---:|
| 1 | 16.4% | 28.4% | **+12.0pp** |
| 2 | 41.4% | 41.0% | -0.4pp |
| 3 | 25.9% | 19.8% | -6.1pp |
| 4 | 13.1% | 8.6% | -4.5pp |
| 5 | 3.2% | 2.2% | -1.0pp |

Observation normalization monotonically moves retention from older cohorts to
the newest one. **As implemented, `e=1` is a recency prior.**

This is mechanically inevitable: dividing by age always penalises old entries.
It also explains the showcase-64 needle result, where late-needle cases lost
nothing (10/10) and early-needle cases collapsed (1/3 retained). The mechanism
intended to protect long-range evidence is, in this implementation, biased
against it.

The intent was defensible — an entry scored more often accumulates more `S`, so
converting a sum into a per-event mean removes that inflation. But it
over-corrects: an evidence token that receives a single strong attention spike
at the boundary where the query matches it has that spike divided by its age,
while an identical spike on a fresh entry is divided by 1. Sum favours old
entries; mean favours new ones. Neither estimates "is this entry unusually
important for its age".

## 4. The richer theory exists in the codebase but is not wired up

`core/scoring.py` implements `realized_exposure(visibility, ..., mode=...)`
with a `"visible_queries"` mode that counts how many query rows could actually
attend to each key — a genuinely non-uniform, causally-derived exposure. It is
exported from `core/__init__.py` and covered by
`tests/test_core_ranking_counts.py`.

It is never called by any runtime module or script. The live path uses the
scalar default instead.

`TorchSignificanceState.update` already accepts `exposure: float | torch.Tensor`
and broadcasts a tensor correctly. **The capability to make `C` informative is
already built and tested; only the call site does not use it.**

## 5. Where the potential is

Ranked by expected information gain per unit of work. Each is falsifiable.

**P1 — Make exposure non-uniform (one line at the call site).**
`record(contribution)` becomes `record(contribution, exposure=<tensor>)`. This
is the minimum change that makes H2 a meaningful question at all. Until `C`
varies for a reason other than arrival time, every `S/C^e` result is a
restatement of age.

**P2 — Eligibility-conditioned exposure.** Increment `C` only at boundaries
where the entry was in the eligible heavy pool, i.e. not protected by the sink
or recent windows. Today a token sits in the recent window accumulating `S`
during the period when attention is highest, then enters the heavy pool with an
inflated mean. Counting only competitive exposure makes `S/C` "mean attention
received while actually competing for retention", which is both better
motivated and genuinely entry-dependent.

**P3 — Rank within cohort instead of dividing by it.** `C` is a clean cohort
label. Use it as one: `z_i = (S_i - mean(S | cohort)) / std(S | cohort)`, or the
scale-free `S_i / mean(S | cohort)`. Both remove the age confound *symmetrically*
and introduce no recency prior. This is the cheapest way to test whether the
age discount is what has been costing early-needle retention.

**P4 — Change the `S` aggregator.** `S` is a plain sum over events. A max or
exponentially-weighted max preserves a single decisive spike, which is what
retrieval evidence looks like, instead of diluting it across the entry's
lifetime.

**P5 — Raise scoring resolution.** With one scoring event per chunk boundary,
`C` has at most 5 levels for ~400 entries. Any normalizer built on it is
quantised to that resolution. More frequent scoring costs compute but is the
only way to get a fine-grained exposure signal.

**P6 — Span-aware selection (Cornerstone 2, reported separately).** Four of the
eleven showcase-64 compression losses were partial multi-token entity
corruptions ("Lucan Tern" to "Lucan T. T"), the signature of per-token
independent top-k. `pool_width` already implements max-pooled span propagation.
Any gain here belongs to Cornerstone 2 and must not be credited to the
significance matrix.

## 5b. Empirical tests of the audit's predictions

Two screens were run on the complete 64-case showcase split to test the audit
against behaviour rather than code reading.

**The positional prediction held.** If `S/C^e` is a recency prior, substituting
raw position for the score at identical budget should not lose anything.
[`halo_showcase64_policy_screen_v1`](POLICY_SCREEN_REPORT.md) found exactly
that: `B4_POS` 22/64 versus B3 19/64, cluster-level 4 better / 3 worse / 9 tied,
`p=1.0`. Significance selection is statistically indistinguishable from a
position counter. The exponent sweep converged on the positional arm as `e`
grew (`e=0` 17, `e=0.5` 16, `e=1` 19, `e=2` 21, position 22), and moving budget
from content selection to the recency window produced the only
cluster-significant contrast in the screen (`R192` 25/64 versus `R008` 14/64,
8 clusters better / 0 worse, `p=0.0078`).

**P3 recovered the needles the age discount was costing.** Ranking entries
against their own cohort instead of dividing by it
([`halo_showcase64_cohort_ranking_v1`](results/runs/halo_showcase64_cohort_ranking_v1)):

| Ranking | Exact | early | middle | late | dist |
|---|---:|---:|---:|---:|---:|
| `B0` full cache | 30/64 | **3** | **6** | 10 | 11 |
| `CMEAN` `S / mean(S \| cohort)` | 22/64 | **3** | 4 | 10 | 5 |
| `CSS` cohort z-score | 22/64 | 2 | 3 | 11 | 6 |
| `B3` `S/C` | 19/64 | 1 | 3 | 10 | 5 |
| `B2` raw `S` | 17/64 | 2 | 2 | 10 | 3 |

`CMEAN` restored early-needle retention to the full-cache level (3, matching
B0's 3, against B3's 1) and improved middle needles from 3 to 4. This is the
specific deficit the audit predicted the age discount would cause.

Honesty about strength: neither cohort arm is cluster-significant against B3
(`CMEAN` 3 better / 1 worse, `p=0.6250`; `CSS` 4 better / 1 worse, `p=0.3750`).
The 19-to-22 improvement is directional, and the needle-profile shift is the
more informative part of the result.

**The unexpected finding: content and position selection are complementary.**
`CMEAN` and `B4_POS` both scored 22/64, but they are not the same 22:

```text
CMEAN = 22   B4_POS = 22   overlap = 15   union = 29
CMEAN-only = 7            B4_POS-only = 7
```

Their needle signatures are nearly anti-correlated — `B4_POS` takes
0 early / 0 middle / 11 late / 11 distributed, `CMEAN` takes
3 early / 4 middle / 10 late / 5 distributed. Restricted to the 30 cases Halo
solves at full cache, the union covers 23/30 where B3 alone covers 19/30.

That reframes the whole mechanism. B3 is *already* a hybrid — 4 sinks, 188
content-selected, 64 recent. The reason it does not capture both regimes is
that its content criterion is itself a recency prior, so the heavy slots
**duplicate** the recency window instead of complementing it. Making the
criterion age-neutral is what turns those slots into a genuinely different
selector.

The 23/30 union is an oracle bound, not an achievable policy. It bounds what a
correct combination could reach, and it is well above anything the current
mechanism attains.

**The obvious way to cash in that complementarity did not work.**
[`halo_showcase64_hybrid_v1`](results/runs/halo_showcase64_hybrid_v1) combined
cohort-relative content selection with a wider recency window, which is the
direct reading of the complementarity result:

| Arm | recent | heavy | Exact | within B0-correct | early | middle | late | dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `R192` `S/C` | 192 | 60 | **25/64** | **22/30** | 4 | 2 | 9 | 10 |
| `HYB192` cohort | 192 | 60 | 22/64 | 22/30 | 0 | 2 | 10 | 10 |
| `HYB128` cohort | 128 | 124 | 21/64 | 20/30 | 1 | 4 | 10 | 6 |

`HYB192` did not beat `R192` (1 cluster better / 3 worse, `p=0.6250`) and did
not beat `CMEAN` or `B4_POS` either. Worse, it *lost* the property that
motivated it: at 60 heavy slots the early-needle recovery disappeared entirely
(0, against `CMEAN`'s 3 at 188 heavy slots).

The lesson is that cohort ranking's early/middle recovery requires a large
heavy pool, while the distributed-needle coverage requires a large recency
window, and a fixed 256-entry budget cannot fund both. The complementarity is
real as a set-overlap fact but does not compose under this budget. The 23/30
oracle bound stands unclaimed.

No compressed arm tested here closes the gap to full cache. `B0` versus
`HYB192` is 7 clusters better / 0 worse, `p=0.0156` — the compression deficit
is the one contrast in this whole program that is unambiguously
cluster-significant.

## 6. What this does not say

- It does not say attention-derived KV retention is worthless. Compression held
  30% absolute accuracy at roughly one third of the KV bytes.
- It does not say `S` is uninformative. The audit concerns the denominator `C`.
- It does not establish that P1-P5 will work. They are falsifiable proposals
  derived from a measured defect, not results.
- It does not overturn the `NO_NOVELTY_CLAIM` verdict. Making exposure
  non-uniform moves the mechanism closer to, not further from, published
  bias-correction work such as NACL and AhaKV.

## 7. Recommended ledger consequences

- Restate `N-020` as an implementation identity rather than evidence.
- Mark hypothesis H2 `NOT_TESTABLE_IN_CURRENT_IMPLEMENTATION` until exposure is
  non-uniform.
- Record the measured cohort-composition shift as the mechanism explanation for
  the needle-profile asymmetry.

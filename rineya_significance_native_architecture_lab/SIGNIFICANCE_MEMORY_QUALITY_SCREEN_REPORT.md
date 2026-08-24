# Experiment D quality screen — matched-total-byte memory slots

## Status

`COMPLETE / EXPLORATORY_LOCAL_NOT_REGISTERED / NEGATIVE_OR_NEUTRAL`

This bounded screen tested all five deterministic `M=4` aggregation controls
against `M=0` hard eviction on the exact 12 case IDs frozen in
`halo_endpoint_validation_v2_registered.yaml`. The case IDs and endpoint v1
were reused; the memory conditions, byte budgets, and contrasts were not
registered. All confidence intervals and unadjusted p-values below are
exploratory.

Authoritative artifacts:

- [screen configuration](configs/halo_significance_memory_quality_screen.yaml)
- [summary](results/runs/halo_significance_memory_quality_screen_v1/summary.json)
- [whole-cluster statistics](results/runs/halo_significance_memory_quality_screen_v1/statistics.json)
- [72 scored cells](results/runs/halo_significance_memory_quality_screen_v1/cases.jsonl)
- [sealed predictions](results/runs/halo_significance_memory_quality_screen_v1/predictions.unscored.jsonl)
- [matching audit](results/runs/halo_significance_memory_quality_screen_v1/matching_audit.json)
- [total-byte contract](results/runs/halo_significance_memory_quality_screen_v1/total_byte_budget.json)
- [same-process identity gate](results/runs/halo_significance_memory_quality_screen_v1/m0_identity_gate.json)
- [manifest](results/runs/halo_significance_memory_quality_screen_v1/manifest.json)
- [content integrity](results/runs/halo_significance_memory_quality_screen_v1/content_manifest.json)

## Design

The full grid ran; successive halving was not used:

```text
12 whole clusters × 6 conditions = 72 closed-loop cells
```

The six conditions were:

1. `M0_HARD`;
2. `M4_MEAN`;
3. `M4_RECENCY`;
4. `M4_RAW_S`;
5. `M4_NORMALIZED_S_OVER_C`;
6. `M4_RANDOM`, with deterministic entry-ID-keyed weights.

All arms used the same normalized-significance ordinary KV selection policy,
chunk schedule, query tokens, three query probes, seed, bf16 Halo checkpoint,
and 32-step greedy decode contract. Only the memory aggregation changed among
the five `M=4` arms.

To make realized persistent-state length identical, predictions stop at the
first EOS but cache appends continue to the fixed 31-append reserve. These
post-EOS appends never enter prediction text or correctness scoring. They make
the total-byte comparison exact and intentionally rule out a latency claim.

## Total persistent-byte match

One ordinary retained position across Halo's 28 layers and 8 KV heads costs
120,064 bytes:

```text
K + V + S + C + absolute position + semantic entry ID
```

Four memory slots per layer/head cost 491,904 bytes, including memory K/V,
validity, source position, creation/update boundaries, source count, and
weight mass. The maximal integer memory-arm budget is therefore:

```text
M0: 128 × 120,064                         = 15,368,192 bytes
M4: 123 × 120,064 + 491,904              = 15,259,776 bytes
unavoidable integer slack below M0        =    108,416 bytes
```

Using 124 ordinary positions would exceed M0, so every M4 arm uses 123. The
108,416-byte slack is less than one ordinary budget unit and is disclosed in
every cell. With the common final chunk and fixed decode reserve added, every
M4 cell remains exactly 108,416 bytes below its paired M0 capacity.

## Pre-outcome gates and sealing

The same-process `M=0` adapter was compared with the original hard-eviction
path before any evaluation-view access. Final logits, all 32 decode argmaxes,
prediction tokens, K/V/S/C, positions, entry IDs, counters, and cache lengths
were bitwise exact; maximum logit delta was `0.0`, memory bytes were zero, and
no memory write occurred.

Every cell then durably wrote:

- its unscored prediction checkpoint;
- exact ordinary resident positions;
- exact bf16 memory K/V bits;
- all slot validity and position metadata;
- SHA-256 hashes for both state artifacts.

Only after all 72 cells existed and query/byte matching passed did the runner
hash and open the frozen evaluation view. Exact-token endpoint scoring uses
ordinary resident positions only: a blended memory slot is not falsely counted
as retaining any exact evidence token.

## Results

| Condition | Ordinary B | Slots/head | Correct | Mean endpoint v1 |
|---|---:|---:|---:|---:|
| M0 hard eviction | 128 | 0 | 5/12 | 0.211310 |
| M4 mean | 123 | 4 | 5/12 | 0.184524 |
| M4 recency | 123 | 4 | 5/12 | 0.186012 |
| M4 raw S | 123 | 4 | 5/12 | 0.186012 |
| M4 normalized S/C | 123 | 4 | 5/12 | 0.183036 |
| M4 random | 123 | 4 | 5/12 | 0.191964 |

All answer-correctness contrasts are exactly zero: every arm solved the same
five clusters and missed the same seven. Memory changed generated text on one
or two clusters per M4 arm, but never changed correctness.

Whole-cluster endpoint contrasts against M0 were:

| Contrast | Mean delta | 95% cluster bootstrap CI | Exact sign-flip p |
|---|---:|---:|---:|
| M4 mean − M0 | -0.026786 | [-0.049107, -0.008929] | 0.015625 |
| M4 recency − M0 | -0.025298 | [-0.050595, -0.007440] | 0.031250 |
| M4 raw S − M0 | -0.025298 | [-0.043155, -0.010417] | 0.0078125 |
| M4 normalized S/C − M0 | -0.028274 | [-0.053571, -0.010417] | 0.0078125 |
| M4 random − M0 | -0.019345 | [-0.035714, -0.005952] | 0.031250 |

These negative exact-token contrasts do not isolate a harmful effect of memory
attention. Total-byte matching necessarily replaces five exact ordinary KV
positions with four blended slots, and endpoint v1 correctly gives a blended
slot no exact-token credit. The answer endpoint is the relevant recovery check;
it showed no gain or loss.

Among the byte-matched M4 controls, normalized `S/C` did not outperform a
simpler aggregation:

| Contrast | Mean endpoint delta | Exact sign-flip p |
|---|---:|---:|
| normalized S/C − mean | -0.001488 | 1.000 |
| normalized S/C − recency | -0.002976 | 0.625 |
| normalized S/C − raw S | -0.002976 | 1.000 |
| normalized S/C − random | -0.008929 | 0.125 |

Random weighting was numerically the best M4 exact-token control, while
normalized `S/C` was the lowest. None of the 15 pairwise tests has a
multiplicity adjustment, and the screen has only 12 clusters, so these
differences are specificity diagnostics rather than confirmatory findings.

## Interpretation

This screen provides no support for the proposed finding that normalized
significance-weighted memory slots recover reasoning information lost by hard
eviction. Under the matched-total-byte contract:

- answer accuracy was identical across all six arms;
- normalized `S/C` had no advantage over mean, recency, raw `S`, or random;
- every M4 arm retained fewer exact ordinary evidence tokens than M0;
- strict exact evidence survival remained 0/12 in every arm.

The result is local to a tiny four-slot, stable-ID-bucket implementation and 12
difficulty-calibrated Halo clusters. It does not rule out other slot counts,
learned projections, token-aware reconstruction objectives, or architectures
that can decode compositional information from summary states. It does rule
out claiming a benefit for this deterministic normalized-`S/C` prototype.

No latency or deployment-memory claim is supported. Dense SDPA attends the
physical slots, timings include deliberate fixed post-EOS accounting, and the
screen measures correctness/retention rather than production efficiency.

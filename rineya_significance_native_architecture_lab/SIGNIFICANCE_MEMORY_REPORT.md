# Experiment D — bounded significance memory slots

## Status

`BOUNDED_PROTOTYPE_COMPLETE / RUNTIME_COMPATIBILITY_ONLY`

This is a modular Architecture Experiment D prototype for the published
Rineya-Halo Qwen3-0.6B core. It establishes a working fixed-slot memory path,
not an answer-quality or architecture-improvement result. No answer was
decoded, no answer key was loaded, and no large sweep was run.

Authoritative artifacts:

- [resolved configuration](configs/halo_significance_memory_smoke.yaml)
- [run summary](results/runs/halo_significance_memory_smoke_v1/summary.json)
- [runtime diagnostics](results/runs/halo_significance_memory_smoke_v1/runtime_diagnostics.json)
- [slot diagnostics](results/runs/halo_significance_memory_smoke_v1/slot_diagnostics.json)
- [manifest](results/runs/halo_significance_memory_smoke_v1/manifest.json)
- [content integrity](results/runs/halo_significance_memory_smoke_v1/content_manifest.json)

## Architecture contract

Every model layer and KV head owns exactly `M` physical slots. A slot is a
weighted aggregate of already post-RoPE evicted keys and their aligned values.
Stable semantic entry ID modulo `M` determines slot membership; physical cache
indices never determine membership.

The implemented unlearned controls are:

1. mean pooling;
2. exponential recency weighting;
3. raw cumulative `S` weighting;
4. exposure-normalized `S/C` weighting;
5. deterministic random weighting.

Weights are captured at eviction. The prototype does not continue updating a
slot's significance after eviction and does not use a learned projection.

Each slot carries explicit metadata:

- `valid`: whether at least one evicted entry has entered the slot;
- `source_positions`: the greatest represented absolute source position;
- `created_at`: the first compression boundary that created the slot;
- `last_updated_at`: the boundary after which current contents may be read;
- `source_counts`: the number of raw evicted entries summarized;
- `weight_mass`: the aggregation mass needed for bounded later merges.

`source_positions` is diagnostic. Aggregated keys are already post-RoPE and are
not rotated again or assigned a synthetic position. Invalid physical slots are
fully masked. A runtime visibility check rejects reads before
`last_updated_at`, so no write can become retroactively visible.

Memory writes accept only K, V, S, C, absolute positions, semantic entry IDs,
and the causal boundary position. The API has no answer/evidence argument, and
the smoke passes only prompt context and question into inference.

## Closed-loop integration

At a compression boundary, the ordinary significance policy chooses retained
KV entries. The exact per-head complement is then merged into memory before
the ordinary cache is gathered. Future Qwen SDPA calls prepend the fixed slots,
expand validity across the GQA query-head groups, and prepend the corresponding
mask columns. Significance updates are computed over the actual augmented
attention distribution, then sliced back to the ordinary cache tail so S/C
remain aligned only with ordinary KV.

Memory is read-only during a temporary query probe. Exact ordinary-cache
rollback plus an exact memory-state hash verifies that the probe changes no
slot. Ordinary appends do not change slot count or slot state.

`M=0` delegates directly to the existing hard-eviction implementation. It is
therefore an actual disabled-feature identity path rather than a masked
approximation.

## Real Halo CPU smoke

The frozen Halo checkpoint hash was verified. One 128-token prompt was split
into two 64-token chunks with a 32-entry ordinary history budget, one
query-conditioned compression boundary, CPU bf16 SDPA, and no answer decode.

| Gate | Result |
|---|---:|
| `M=0` final-logit equality with hard eviction | exact; max delta `0.0` |
| `M=0` ordinary K/V/S/C/position/ID state | exact |
| `M=0` persistent memory bytes | `0` |
| `M=4`, normalized `S/C`, final logits | all finite |
| `M=4` ordinary semantic / resident length | `128 / 96` |
| `M=4` layer writes | `28 / 28` |
| valid / physical slot capacity | `896 / 896` |
| later attention calls with valid slots | `28` |

The `M=4` slot ledger is exact:

| Persistent tensor class | Bytes |
|---|---:|
| memory K | 229,376 |
| memory V | 229,376 |
| validity | 896 |
| source positions | 7,168 |
| creation boundaries | 7,168 |
| last-update boundaries | 7,168 |
| source counts | 7,168 |
| aggregation mass | 3,584 |
| **memory total** | **491,904** |

For this smoke, ordinary retained K/V uses 11,010,048 bytes, S/C uses 172,032
bytes, alignment metadata uses 344,064 bytes, and memory adds 491,904 bytes.
Total persistent architecture state is 12,018,048 bytes. This run is not a
matched-total-byte quality comparison.

## Invariant coverage

The focused tests cover:

- exact fixed shapes and zero-byte `M=0` state;
- every declared unlearned weighting variant;
- deterministic random-control replay;
- stable semantic-ID bucketing and bounded repeated merges;
- exact component-wise byte accounting;
- rejection of unscored and future-position writes;
- absence of oracle/answer inputs from the write signature;
- exact hard-eviction cache and logit identity at `M=0`;
- GQA validity expansion and ordinary bottom-right mask preservation;
- ordinary-cache alignment after eviction and append;
- exact query rollback with read-only memory;
- finite tiny-Qwen and published-Halo closed-loop execution at `M=4`.

## What this does not show

The smoke does not compare answer accuracy, evidence recovery, reasoning, or
long-context quality. It does not show that normalized `S/C` beats mean,
recency, raw `S`, or random controls. It also does not establish a latency
benefit: dense SDPA still attends the physical slots and the measurements are
small CPU phase timings.

A scientifically useful next step would require a frozen eligible benchmark,
matched total persistent bytes, all five deterministic controls, cluster-level
statistics, and a prospectively registered endpoint. Until then the only
supported conclusion is that the bounded memory mechanism is implemented,
disabled-path exact, and runtime-compatible with Halo on this smoke.

## Completed exploratory follow-up

The requested matched-total-byte follow-up is now complete on 12 frozen Halo
clusters and all five controls. See the
[quality-screen report](SIGNIFICANCE_MEMORY_QUALITY_SCREEN_REPORT.md). It found
identical 5/12 answer accuracy across `M=0` and every `M=4` arm, lower exact
ordinary-token retention in every M4 arm, and no specificity advantage for
normalized `S/C`. The follow-up is exploratory/local and does not change the
smoke's narrow runtime-evidence classification.

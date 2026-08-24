# Efficiency report

Status: five published-family CPU accounting smokes `COMPLETE`; local endpoint
validation and Halo count/schedule controls `COMPLETE`; Experiments B-F have
bounded CPU ledgers; truly ragged adaptive storage, synchronized deployment
latency, GPU VRAM, and deployment efficiency remain `NOT_RUN`. The existing
Tier-A byte-ledger run is `UNDER_METHODOLOGY_REVIEW` and is not a current
scientific result.

## Published Qwen3-0.6B family KV accounting

The five in-scope cores share 28 layers, 8 KV heads, head dimension 128, and ran
in bf16:

```text
KV bytes per stored context entry
  = layers * KV heads * head_dim * (K + V) * bytes_per_value
  = 28 * 8 * 128 * 2 * 2
  = 114,688 bytes
  = 112 KiB
```

| Stored entries | KV-only bytes | Binary size |
|---:|---:|---:|
| 112 | 12,845,056 | 12.25 MiB |
| 512 | 58,720,256 | 56 MiB |
| 1,024 | 117,440,512 | 112 MiB |
| 2,048 | 234,881,024 | 224 MiB |
| 40,960 | 4,697,620,480 | 4.375 GiB |

These formulas assume the same entry count at every layer/KV head. Adaptive
allocation must report the exact sum rather than reusing this shortcut.

## Significance-state accounting

With float32 `S` and `C` per layer, KV head, and stored entry:

```text
S+C bytes per entry = 28 * 8 * (4 + 4) = 1,792 bytes
overhead relative to bf16 KV = 1.5625%
```

Absolute positions, stable entry IDs, selection workspaces, traces, controller
state, memory slots, and any recovery store are additional.

## Capacity and byte-reporting rule

Do not report one ambiguous cache budget. Record:

1. retained-history entries carried between chunks;
2. active-chunk entries in the current forward;
3. transient-query entries before rollback;
4. stored entries after the final chunk;
5. peak live entries;
6. KV, scores, counts, identity metadata, optional feature state, and total
   persistent bytes;
7. transient workspace bytes separately.

Compressed policy comparisons match retained-history capacity and total
persistent bytes. `B2` retains count storage even when `e=0`. Positional,
shuffled, and random controls reserve equivalent auxiliary state or reduce
their retained history. `B0` remains an unmatched capability reference.

## Completed published-family CPU smokes

Halo, Halo Uncensored, Lyra, Nimbus, and Pulse each loaded verified model,
tokenizer, and config hashes on CPU in bf16. Every run used the same one-case
scope and final state:

```text
48 retained history + 64 active chunk = 112 stored entries (B48+C64)
10 transient query tokens were used and rolled back (32 maximum configured)
```

| Persistent component after final chunk | Bytes |
|---|---:|
| K/V | 12,845,056 |
| S+C | 200,704 |
| absolute-position and stable-ID metadata | 401,408 |
| total persistent | 13,447,168 |

Every manifest is explicitly `matched=false`; these were not comparative memory
experiments. The two diagnostic regions performed different work and cannot
define a speedup ratio. Each run used one 192-token prompt and decoded no
answer. Together they establish compatibility on five separately identified
cores, not throughput, long-context scaling, continuous peak RSS, deployment
latency, GPU memory behavior, or quality.

Runtime status: local recorder implementing audited semantics; four audited
source primitives match exactly, while end-to-end source benchmark reproduction
is `NOT_RUN`. This parity artifact contains no efficiency measurement.

## Bounded Halo mini evaluation accounting

The three-case `halo_mini_eval_20260823_v2` used a 256-entry retained-history
allowance, 256-token active chunks, and up to 64 transient query tokens. B2 and
B3 shared the same retained-entry allowance. Their stored entries after prefill
were `377`, `395`, and `398`; after decode they were B2
`379/399/400` and B3 `379/398/400`, reflecting different generated lengths.
B0 retained full context (`1145/1163/1166` entries after prefill) and was not
memory matched.

| Case | B0 persistent bytes | B2 persistent bytes | B3 persistent bytes | B0 / compressed |
|---|---:|---:|---:|---:|
| RLC-001-B | 131,547,136 | 45,504,256 | 45,504,256 | 2.89x |
| RLC-010-B | 133,726,208 | 47,905,536 | 47,785,472 | 2.79-2.80x |
| RLC-019-B | 133,955,584 | 48,025,600 | 48,025,600 | 2.79x |

These are end-of-decode persistent-state readings, not peak RSS savings. B0 is
a capability reference rather than a byte-matched competitor.

B0, B2, and B3 each scored 1/3. B2 and B3 were both correct on the one
B0-eligible case, so there was no measurable quality gain in this bounded
slice. The run provides no inferential efficiency result. Condition wall times
perform different decode trajectories on three cases and are not speedup
estimates.

## Halo retention-pilot accounting

`halo_retention_pilot_v2` ran 11 B0-selected R1/R3/R4 clusters with 128 retained
history entries, 128-token active chunks, and at most 64 transient query tokens.
B2, B3, and B4 shared the retained-entry budget. Terminal persistent bytes vary
with generated-answer length, so the means below are accounting observations,
not fixed allocation targets or peak-memory savings.

| Condition | Mean terminal persistent bytes | Mean MiB | Correct | Strict evidence |
|---|---:|---:|---:|---:|
| B2 raw | 25,344,419 | 24.170 | 5/11 | 0/11 |
| B3 normalized | 25,409,908 | 24.233 | 6/11 | 0/11 |
| B4 positional | 26,195,782 | 24.982 | 1/11 | 0/11 |

The B3-B2 terminal-byte difference is driven by differing decode lengths and
does not establish a memory-efficiency penalty or gain. Sampled RSS is also not
a controlled peak-memory comparison. The strict all-token endpoint saturated
at zero in every condition. Post-hoc graded-retention differences favoring B3
are scientific diagnostics, not efficiency results, and the B3-B2 answer test
was `p=1.0`.

## Locally registered endpoint-v2 accounting

The first prospective dataset stopped at B0 (`9/24`) and therefore incurred no
compressed-policy allocation. Difficulty-calibrated v2 passed B0 (`18/24`) and
ran 12 clusters under B2/B3/B4 with the same B128/C128/query contract. Mean
terminal persistent bytes were B2 `24,893,269`, B3 `24,993,323`, and B4
`25,743,723`. These means vary with decode length; they are not exact matched-
terminal capacities or peak RSS comparisons. The positive B3-B2 endpoint
contrast and negative answer count (`5/12` vs `6/12`) do not establish an
efficiency gain.

The registered execution was internally paired in one Python 3.11.15,
Torch 2.13.0+cpu, Transformers 5.15.0 process. The current lock and later B-D
screens use Torch 2.12.1+cu126 and Transformers 5.14.1. The endpoint result has
not been reproduced under that locked runtime, so cross-version timing, state,
and quality comparisons are not admissible.

## Count-control accounting

`halo_count_causal_controls_v1` completed 88 cells: eight count transforms over
11 selected clusters. Every condition used the same 128-entry retained-history
budget and identical query-access schedule. True count state was preserved for
audit even when a transformed denominator drove ranking, so the controls do
not represent storage-optimized implementations.

Age-derived and head-averaged transforms exactly matched the true-C selection
and resident trajectories under this fixed schedule. That engineering result
suggests an opportunity to omit or compress explicit C **only after** the same
equivalence is verified under the intended irregular schedule; it is not an
observed byte saving here. Constant, shuffled, delayed, noisy, and wrong-token
conditions are controls, not candidate efficiency optimizations. Shuffled and
wrong-token are further confounded because their global permutations cross
protected sink/recent slots and alter the eligible-heavy-pool C multiset; no
deployment inference should be drawn from them.

The run persisted 264 boundary-evidence summaries plus full boundary
diagnostics. Their disk footprint and analysis time are instrumentation costs,
not runtime state required by a deployment implementation.

Eligible-pool count v2 completed 33 additional cells under the same B128/C128
logical contract. Both clean transforms preserve the same C storage as true C;
they are causal controls, not byte-saving variants. The within-process true-
count identity is exact. Separate v1/v2 processes were not bitwise trajectory-
reproducible, so no resource or outcome delta may be formed by subtracting
across those runs.

## Held-constant schedule accounting

`halo_schedule_control_v2` completed 110 cells and 55 matched B3-B2 pairs. Each
cell used 128 retained-history entries and a fixed 64-token terminal active
suffix. The matching audit reports equal final pre-decode prompt-cache size and equal query
access within every B2/B3 schedule pair at the end of prompt prefill. Five
prefix chunk sizes changed the number of scoring/compression events without
changing that pre-decode budget. Generated-answer lengths can differ afterward;
terminal B2/B3 persistent state differs by as much as 0.385 MiB and must not be
called byte matched.

Observed local CPU group means ranged from roughly 1.75 to 4.04 seconds per
cell, and mean query-token evaluations rose with realized rounds (about 13.8 at
one round to 108 at seven rounds). These are local CPU measurements with
different schedule geometries and unequal group sizes; they are not deployment
latency, throughput, GPU speedup, or a causal efficiency estimate. Final
persistent tensor means were about 22.25–24.96 MiB across grouped schedule
cells; allocator/object overhead and peak RSS are excluded.

The positive frozen-endpoint interaction (`+0.00936074` per round) and null
answer interaction (`p=0.4375`) are quality analyses, not efficiency wins. A
quality-versus-cost claim requires a registered workload and synchronized GPU
measurements.

## Adaptive-budget screen accounting

`halo_adaptive_budget_screen_v1` conserves exactly 28,672 retained entries
across 28 layers x 8 KV heads—an average of 128 per layer/head—at every replayed
boundary. Per-layer/head allocations range from 36 to 220. Uniform allocation
matches the recorded runtime in 33/33 boundary checks.

This is logical entry-budget accounting only. The screen assumes sparse
per-head/layer physical storage and does not measure padding, indexing,
fragmentation, gather kernels, RSS, or latency. A rectangular implementation
allocated at each group's maximum could erase the nominal budget advantage.
No model was loaded, and no answer-quality or end-to-end efficiency result was
measured. The failed specificity gate also means there is no basis for
promoting entropy allocation as an efficient Cornerstone-specific controller.

The closed-loop padded adapter makes the missing physical cost explicit. Its
uniform path is bitwise identical to the rectangular path. In the 36-cell
quality screen, every condition matched mean prefill logical payload at
23,732,651 bytes. Uniform physical bytes were identical to logical. Entropy and
shuffled physical bytes averaged 26,134,358 and 26,852,148, with mean post-
decode padding overhead 2,403,854 and 3,121,756 bytes—about 9.7% and 12.5% of
their logical payload. This padded implementation therefore supplies no memory-
saving claim despite exact logical-budget conservation. Timings are exploratory
CPU measurements under 2.12.1/5.14.1, not comparable to the earlier registered
2.13.0/5.15.0 run.

## Experiments C-F accounting

Experiment C retains the same B128 ordinary cache and adds no persistent prior
tensor: bias is constructed transiently from existing S/C and masks. All 60
cells scored `5/12`; no quality-per-byte gain exists. The bounded bias changes
SDPA work and disables fused GQA when a head-specific mask is needed, so CPU
phase timings are not deployment throughput.

Experiment D provides the strongest total-capacity ledger. One ordinary Halo
position across all layers/heads costs 120,064 bytes including K/V/S/C and
alignment metadata. Four memory slots/head cost exactly 491,904 bytes. M0 B128
is 15,368,192 bytes; M4 B123 plus slots is 15,259,776 bytes, leaving 108,416
bytes unavoidable integer slack. All five M4 arms used that same capacity and
all six arms scored `5/12`. The fixed post-EOS append reserve equalizes
persistent state but intentionally invalidates a latency claim.

Experiment E's open-loop cost curve reuses compressed chunk-128 and full-cache
chunk-256 timings. Recover-all is recorded as `1.3696x` wall time and 58,949,632
bytes sequential peak persistent state, but this is an accounting illustration,
not a schedule-matched benchmark. The live one-case recovery added 1.3832 s,
58,720,256 bytes of full-cache context K/V, and 4,096 bytes of input IDs.

Experiment F loads no backbone. The seven-parameter logistic fit took
`0.00159 s`; sampled peak process RSS was 49,217,536 bytes, analytically
transient optimizer state was 504 bytes, and persisted optimizer state was
zero. Its held-out two-recovery path has a recorded `1.172x` counterfactual wall
multiplier, again from non-schedule-matched source timings. With only three
evaluation clusters, neither controller timing nor apparent quality is a
deployment estimate.

## Tier-A formula artifact under review

The old model-free run wrote the following deterministic formula table for the
same 28L/8KV/head-dimension geometry:

| Comparison | Entries/head | KV bytes | Auxiliary bytes | Total bytes |
|---|---:|---:|---:|---:|
| Slot matched, no S/C | 256 | 29,360,128 | 1,433,600 | 30,793,728 |
| Slot matched, with S/C | 256 | 29,360,128 | 1,892,352 | 31,252,480 |
| Byte matched, no S/C | 256 | 29,360,128 | 1,433,600 | 30,793,728 |
| Byte matched, with S/C | 252 | 28,901,376 | 1,862,784 | 30,764,160 |

The arithmetic may be audited independently, but the run's methodology and
interpretation remain under review. The table is not a measured Halo allocation
or evidence that normalization is efficient.

## Model-quality and deployment comparisons

| Variant/run | Retained history | Active chunk | Transient query | Total persistent bytes | Quality | Status |
|---|---:|---:|---:|---:|---:|---|
| mini B0 full capability reference | full | chunks 256 | up to 64 | case-specific | 1/3 exploratory | completed, unmatched |
| mini B2 raw | 256 | chunks 256 | up to 64 | case-specific | 1/3; 1/1 eligible | completed exploratory |
| mini B3 normalized | 256 | chunks 256 | up to 64 | case-specific | 1/3; 1/1 eligible | completed exploratory |
| pilot B2 raw | 128 | chunks 128 | up to 64 | mean 25,344,419 | 5/11 selected | completed exploratory |
| pilot B3 normalized | 128 | chunks 128 | up to 64 | mean 25,409,908 | 6/11 selected | completed exploratory |
| pilot B4 positional | 128 | chunks 128 | up to 64 | mean 26,195,782 | 1/11 selected | completed exploratory |
| locally registered B2/B3/B4 | 128 | chunks 128 | up to 64 | means 24.89/24.99/25.74 MB | answers 6/5/1; B3 retained-state contrast positive | local 12-cluster validation; not cross-runtime reproduced |
| count controls | 128 | chunks 128 | up to 64 | condition/case-specific | frozen endpoint varies; answers 5–7/11 | 88 cells, exploratory |
| eligible count v2 | 128 | chunks 128 | up to 64 | condition/case-specific | clean controls Holm-null | 33 cells, process-local exploratory |
| schedule B2/B3 | 128 | terminal active 64 | schedule-specific probes | grouped means 22.25–24.96 MiB | retention interaction only | 110 cells, exploratory |
| adaptive allocation replay | 28,672 global layer/head entries | boundary-local only | recorded query state | sparse physical bytes unmeasured | no answers; specificity failed | 132 rows, open loop |
| heterogeneous allocation | 28,672 logical entries | chunks 128 | matched probes | adaptive padding +2.40–3.12 MB mean | 5/8/5 answers; specificity failed | 36 cells, padded closed loop |
| attention prior | 128 | chunks 128 | matched probes | no new persistent prior tensor | all arms 5/12 | 60 cells, specificity failed |
| memory M0/M4 | 128 / 123 + slots | chunks 128 | matched probes | 15,368,192 / 15,259,776 capacity bytes | all arms 5/12 | 72 cells, negative/neutral |
| recovery E | compressed + conditional full cache | source schedules differ | none after decision | recover-all peak 58,949,632 B | risk AUC 0.50 | mechanism only |
| tiny controller F | no backbone fit | n/a | pre-answer features | 7 parameters; 504 B transient optimizer | held-out n=3 AUC/BA 1.0 | underpowered/overfit-risk screen only |

No row is a confirmatory quality-versus-bytes result. B0 capability calibration
was 18/32 overall but failed the per-family gate, and the retention-pilot rows
exclude the incapable R2 benchmark family by B0 selection.

CPU and GPU timings must be reported separately. CUDA measurements require
synchronization and both allocated/reserved VRAM. Trace/write overhead is timed
separately from the runtime policy.

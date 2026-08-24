# Halo showcase-64 Cornerstone-1 evaluation

Run: [`halo_showcase64_eval_v1`](results/runs/halo_showcase64_eval_v1)
Classification: `BOUNDED_EXPLORATORY`, unregistered.
Scope: Rineya Halo 1.2.0-devfreeze only.

This is the first Cornerstone-1 evaluation to cover the **complete** 64-case
showcase split of the Rineya Long Context Demonstration Suite. The earlier
[`halo_mini_eval_20260823_v2`](results/runs/halo_mini_eval_20260823_v2) covered
three base-variant cases from the same split; that run and its hard-locked
evaluator are unchanged.

## What ran

64 showcase cases (16 scenario packs x base / paraphrase / counterfactual /
unanswerable) under three conditions:

| Condition | Policy | Exponent |
|---|---|---|
| `B0` | instrumented full cache | not applicable |
| `B2` | significance, raw accumulation | `e=0` |
| `B3` | significance, observation normalization | `e=1` |

192 case-conditions, 21.2 minutes total CPU inference. Prompts are 1,130-1,504
tokens, giving four to five chunks and four to five compression boundaries per
case at `chunk=256`, `budget=256`, `sinks=4`, `recent=64`.

Compression semantics are imported unchanged from
`run_published_core_eval.execute_condition`, so this run and the recorded mini
evaluation share one audited execution path. The answer key was opened only
after all 192 generations completed and the answer-blind artifact was durably
written. Runtime parity with the suite baselines: seed 17, greedy decoding,
`enable_thinking=false`, `max_new_tokens=24`.

## Results

Scored twice with independent normalizers — the lab's own scorer and the
demonstration suite's `evaluate_predictions.py`. Both agree exactly.

| Condition | Strict exact | base | paraphrase | counterfactual | unanswerable |
|---|---:|---:|---:|---:|---:|
| `B0` full cache | **30/64 = 0.469** | 8/16 | 11/16 | 9/16 | 2/16 |
| `B3` `e=1` | 19/64 = 0.297 | 5/16 | 9/16 | 5/16 | 0/16 |
| `B2` `e=0` | 17/64 = 0.266 | 4/16 | 8/16 | 4/16 | 1/16 |

Against the recorded same-suite baselines:

| Run | Strict exact |
|---|---:|
| qwen3_8b_showcase | 0.828 |
| qwen3_4b_showcase | 0.672 |
| **Halo B0 (full cache)** | **0.469** |
| **Halo B3 (`e=1`)** | **0.297** |
| **Halo B2 (`e=0`)** | **0.266** |
| qwen3_06b_showcase | 0.219 |
| aegis_showcase | 0.062 |
| greedy_soup_v2_showcase | 0.000 |

## Finding 1 — B3 over B2 is null at the correct analysis unit

Case-level counts favour B3 by two answers, but that evaporates under
cluster-aware analysis:

```text
case level:     4 wins, 2 losses, exact two-sided sign p = 0.6875
cluster level:  2 packs better, 2 packs worse, 12 tied, exact p = 1.0000
```

The four case-level wins fall in only two scenario packs (`RLC-042` supplies
two, `RLC-064` supplies two). Scenario-pack variants are correlated by
construction, so the 64 rows are **not** 64 independent observations and the
16-pack contrast is the honest unit. At that unit the effect is exactly null.

This is consistent with, and does not upgrade, existing ledger rows `N-006` and
`N-031`: observation normalization still supports no answer-quality claim.

## Finding 2 — compression costs 17.2 points at ~1/3 the storage

B3 retains a median 32.0% of B0's cache entries (range 22.5%-36.2%) and 33.7%
of B0's KV bytes including significance-matrix and alignment overhead. Median
persistent footprint is 48.0 MB against 145.0 MB.

```text
accuracy   0.469 -> 0.297   (-17.2 points)
storage    145.0 MB -> 48.0 MB   (-66.9%)
```

Whether that trade is acceptable is a deployment question this run does not
answer. It reports the exchange rate on one 0.6B core, one suite, and one fixed
schedule; nothing here establishes a frontier or a deployment recommendation.

## Finding 3 — Halo's full-cache score more than doubles the stock base

Halo B0 scored 0.469 against 0.219 for `qwen3_06b_showcase` on the identical 64
cases, key, and scorer, at the same parameter count. Compressed Halo (0.297)
still exceeds the uncompressed stock baseline.

Two limits on that comparison. The baselines ran through
`run_local_hf.py` with HuggingFace `generate`, whereas B0 runs the lab's
chunked path with rolled-back query probes; chunked-versus-single-shot logit
identity is verified exactly (`N-002`) but only on a 192-token case, not at
1,130-1,504 tokens. And this is a single greedy pass on one suite, so it
supports a descriptive statement about this benchmark, not a general capability
claim.

## Finding 4 — abstention is weak before compression, not because of it

The unanswerable variant went `2/16` at B0, `1/16` at B2, `0/16` at B3. The
headline reading — "compression destroys abstention" — is not supported: with
only two B0-eligible cases the compressed delta is uninterpretable. What the
data does show is that Halo largely fails to abstain **at full cache**,
confidently answering questions the suite marks unanswerable:

```text
RLC-001-U  accepted "UNKNOWN"   B0 -> "Mosaic"          B3 -> "Mosaic"
RLC-019-U  accepted "UNKNOWN"   B0 -> "Tuesday."        B3 -> "Tuesday."
```

That is a base-capability finding about Halo, not a Cornerstone-1 finding.

## Sanity checks

- Coverage 64/64 in all three conditions; no duplicate or unexpected IDs.
- Zero cases where a compressed condition was correct while B0 was wrong,
  which is the expected pattern if compression only ever removes information.
- Resident-entry accounting and per-layer query-probe counts were asserted
  against the predicted schedule on every case-condition.
- Both scorers returned identical strict-exact values for all three conditions.

## What this run does not support

- No answer-quality claim for observation normalization. The cluster-level
  contrast is null (`p=1.0`).
- No confirmatory status. Nothing was preregistered, no endpoint was frozen,
  and no advance gate was declared before outcomes.
- No published-family claim. Halo Uncensored, Lyra, Nimbus, and Pulse were not
  run; ledger row `N-009` stays `NOT_RUN`.
- No deployment, latency, or GPU claim. CPU timings are local accounting.
- No retention-endpoint result. `layer-fact-retention-v1` was not computed
  here; this run reports answers and storage only.

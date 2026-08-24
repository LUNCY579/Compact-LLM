# Architecture Experiment C — Soft Significance-Aware Attention

## Status

`RUNTIME_COMPATIBILITY_PASS / QUALITY_NOT_EVALUATED`

This bounded Experiment-C slice adds a modular, closed-loop attention prior to
the published Rineya Qwen3-0.6B path. It does **not** establish Finding B, answer
quality, long-context benefit, or feedback safety.

## Mechanism

For layer `l` and an already cached key `j`, the SDPA logit receives

```text
L'[i,j] = L[i,j] + lambda_l * clip((R[j] - mean(R)) / (std(R) + epsilon), -a, a)
```

The default source is `R = S / max(C, 1)`. The implementation also exposes raw
`S`, deterministically shuffled normalized significance, and deterministic age
(larger age gives older historical keys the larger control source).
It supports one global lambda or per-layer overrides. Every lambda is checked
against `max_abs_lambda`, and the source transform is finite-checked and clipped.

The prior is restricted to the historical prefix `key_length - query_length`.
For significance modes, a key must also have positive realized observation count.
Current-forward keys, padding, masked keys, and causal-future keys receive literal
zero prior. A nonzero prior is merged into an explicit bottom-right causal mask;
this avoids the SDPA failure mode in which supplying a new additive mask silently
disables implicit causality. With `lambda=0`, the mask and attention call take a
literal unmodified bypass.

The implementation is in
[`runtime/attention_prior.py`](src/rineya_significance_lab/runtime/attention_prior.py)
and the Qwen/Transformers bridge is in
[`runtime/transformers_runtime.py`](src/rineya_significance_lab/runtime/transformers_runtime.py).

## Condition coverage

| Condition | Implementation | Execution in this slice |
|---|---|---|
| C0 `lambda=0` | Global/per-layer literal bypass | Tiny Qwen tests and real Halo CPU smoke |
| C1 fixed global lambda | Implemented | Halo `lambda=0.01` compatibility smoke |
| C2 per-layer lambdas | Implemented API and unit-tested fallback/override semantics | Not quality-tested |
| C3 CPU-trained lambdas | Per-layer inference API is ready | Not trained; no quality tuning authorized |
| C4 shuffled significance | Deterministic within eligible cached keys | Unit-tested only |
| C5 raw `S` | Implemented separately from `S/C` | Unit-tested only |
| C6 age-only | Deterministic cache-position age | Unit-tested only |

## Leakage declaration

The canonical smoke is `QUERY_AGNOSTIC`:

```text
query: none
answer: none
oracle evidence: none
future tokens: none
future hidden states: none
```

The API permits `QUERY_CONDITIONED` state only when this is declared explicitly.
It describes prior query-probe contributions already carried by persistent S/C;
the prior itself never receives question tokens. Answer/oracle/future access is
rejected by construction.

## Real Halo CPU smoke

The frozen configuration is
[`configs/halo_attention_prior_smoke.yaml`](configs/halo_attention_prior_smoke.yaml).
It verified the published Halo weights, tokenizer, and model config hashes, then
ran one 24-token, three-chunk, CPU-bfloat16 prompt with no answer decode and no KV
eviction.

Results:

- C0 logits: bit-exact to the schedule-matched plain unrecorded chunked B0
  path; maximum delta `0.0`, argmax agreement `1.0`.
- C0 `(K,V,position,entry_id)` structure: bit-exact to the plain, unrecorded B0
  full-cache path. The additional S/C observation state is intentionally absent
  from B0 and therefore is not part of that identity comparison.
- C1 `lambda=0.01`: finite logits and finite persistent state.
- C1 nonzero prior applications: `56` across 28 layers and two historical
  boundaries.
- Maximum observed prior magnitude: `0.01999999955`, within the declared `0.02`
  bound.
- Maximum current-key prior: `0.0`.
- Maximum invalid/masked-key prior: `0.0`.
- Same full-cache storage in every condition: 24 resident entries and zero
  evictions.

The descriptive C1-vs-baseline logit delta (`max 0.34375`, argmax agreement
`0.95833`) only proves that the nonzero intervention was active. It is not a
quality metric. The mean layer diagnostic showed source-entropy change of
`+0.09061` from the first to second biased boundary, but two boundaries on one
prompt cannot diagnose rich-get-richer collapse or safety.

Machine-readable evidence is under
[`results/runs/halo_attention_prior_smoke_v1`](results/runs/halo_attention_prior_smoke_v1/).

## Verification and decision

Focused tests cover:

- exact C0 logits, K/V, S/C, and metadata identity on tiny Qwen;
- GQA head mapping;
- boolean/additive padding and causal masks;
- literal zero bias on current, invalid, and future keys;
- normalized S/C, raw S, shuffled significance, and age controls;
- global and per-layer lambdas;
- bounds, finite guards, access declarations, and feedback diagnostics.
- closed-loop feedback: later S differs from the no-prior path once the biased
  attention distribution is used for the next significance observation.
- declared query-conditioned state after exact temporary-probe rollback, while
  undeclared query conditioning is rejected before model execution.

Decision: `NEEDS_CONTROLLED_QUALITY_EVALUATION`.

That evaluation has now run. See
[`ATTENTION_PRIOR_QUALITY_REPORT.md`](ATTENTION_PRIOR_QUALITY_REPORT.md). The
result is `NO_MEASURABLE_ANSWER_GAIN / SPECIFICITY_NOT_SUPPORTED`: normalized
`S/C`, raw `S`, shuffled significance, age-only, and no-prior all scored 5/12,
and normalized `S/C` did not beat raw or shuffled significance on endpoint v1.

The C0 gate is intentionally schedule-matched. It does not claim that changing
Qwen SDPA from this three-chunk execution to a single-shot execution is bit-exact
in CPU bfloat16; that is a separate chunk/kernel numerical question, not an
attention-prior identity test.

That controlled quality run has now completed; the authoritative interpretation
is in [`ATTENTION_PRIOR_QUALITY_REPORT.md`](ATTENTION_PRIOR_QUALITY_REPORT.md).
All five conditions scored `5/12`, normalized `S/C` did not beat raw or
shuffled significance on endpoint v1, and exact false-positive persistence
could not be scored because the frozen evaluation view lacks exact tokenizer
spans for distractor/stale segments. The next step, if pursued, requires a new
prospectively registered design rather than another retrospective reuse of the
same 12 clusters.

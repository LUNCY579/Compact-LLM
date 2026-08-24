# Closed-loop heterogeneous budget runtime

Status: **prototype validated for runtime compatibility, not architecture quality**.

## Why padding is required

Transformers 5.14 Qwen3 attention receives K/V as rectangular tensors with
shape `[batch, KV heads, slots, head_dim]`. SDPA has no tensor contract for a
different physical sequence dimension per KV head. Qwen3 also constructs one
model-level causal-mask mapping before iterating through its decoder layers, so
unequal physical widths across layers cannot consume that shared mask without
an adapter.

The implemented representation is exact at the attention-semantics level:

- each head retains its declared logical entry count;
- selected entries are packed left and shorter heads are zero padded to the
  longest head in that layer;
- a boolean `[batch, KV heads, slots]` validity tensor travels with K, V, S, C,
  absolute position, and semantic entry ID;
- padded S/C and K/V values are zero, padded metadata is `-1`, and all
  invariants fail loudly if those values diverge;
- the fixed global budget is the sum of valid entries across layer/head units,
  per sequence.

This is not a truly ragged storage kernel. Physical bytes include padding and
the validity tensor, and the byte ledger reports logical payload and physical
storage separately.

## Mask and GQA semantics

At each forward, Qwen supplies one mask sized for the widest layer. The adapter
maps it to every layer as:

`this layer's old prefix + the shared new-token suffix`

Naively taking the first `layer_width` columns would drop new-token columns
when a layer is shorter. After remapping, head validity is repeated across the
contiguous query-head group belonging to each KV head. The combined mask is
bottom-right causal, and explicit `repeat_kv` is used because a head-specific
mask disables fused GQA. The same adjusted mask is used by model SDPA and the
post-RoPE auxiliary significance calculation.

Temporary query probes snapshot each layer's own physical width. Rollback
removes its layer-specific suffix while retaining only S/C updates to the
pre-probe valid entries. Ordinary decode append uses the same mask adapter and
adds one valid semantic entry per layer/head.

## Gates passed

The deliberately unequal tiny-Qwen schedule `[[2, 3], [3, 4]]` passes:

- exact global-budget conservation;
- unequal head and layer physical widths;
- padding exclusion under 4Q/2KV grouped-query attention;
- bottom-right causal masking after layer-width remapping;
- exact query rollback;
- decode append;
- exact tensor-payload byte accounting.

A uniform 3-entry allocation in every layer/head is exactly identical to the
existing rectangular runtime: logits, K, V, S, C, positions, and entry IDs are
bitwise equal.

## One published-Halo smoke

`halo_heterogeneous_budget_smoke_v2` ran one 48-token Halo case on CPU bf16,
with one 24-token compression boundary and an 8-token query probe.

- the uniform padded path was bitwise identical to the rectangular path
  (`max_abs_logit_delta = 0`, exact complete cache state);
- the live entropy controller conserved exactly 3,584 logical entries across
  28 layers × 8 KV heads;
- allocations genuinely differed, spanning 14–18 entries per unit;
- the following chunk and one decode append completed with finite logits;
- after decode, logical payload was 4,922,624 bytes while physical padded
  storage was 5,129,424 bytes, including a 9,552-byte validity tensor;
- representation overhead was 206,800 bytes (about 4.2% of logical payload).

This smoke establishes mechanism compatibility only. It does not show answer
improvement, an optimal allocation statistic, latency improvement, or memory
savings versus a truly ragged kernel.

## Current limitations

- The adapter targets inference-only Transformers SDPA and Qwen3's no-position-
  bias attention path.
- Live adaptive allocation currently requires batch size one. A single global
  matrix would otherwise mix sequence-specific diagnostics.
- Head-level semantic sparsity is exact, but physical storage remains padded to
  each layer's longest head.
- The controller cannot resurrect previously evicted entries; future budgets
  are capped by each unit's current retained entries plus newly appended ones.
- The prototype is not yet a quality experiment or a registered confirmatory
  endpoint.

Implementation entry points:

- `runtime/heterogeneous_cache.py`
- `runtime/heterogeneous_attention.py`
- `runtime/heterogeneous_budget.py`
- `runtime/heterogeneous_runtime.py`

## Completed exploratory follow-up

The 36-cell closed-loop quality screen is now complete; see
[HETEROGENEOUS_BUDGET_QUALITY_SCREEN.md](HETEROGENEOUS_BUDGET_QUALITY_SCREEN.md).
Entropy allocation scored 8/12 versus uniform 5/12, but the paired answer test
was `p=0.25`, shuffled allocation had higher mean retention, and padded physical
storage exceeded logical payload. The runtime result remains valid, while the
quality-specificity decision is negative.

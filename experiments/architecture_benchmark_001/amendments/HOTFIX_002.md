# HOTFIX-002 — scoped high-precision accumulation for the CED shared-memory read

Date: September 11, 2026 (America/Chicago). This is a **diagnostic instrument**, not a
relaxation of any gate and not a new acceptance rule. It adds an explicitly registered,
default-inert option and the means to measure it. It does not by itself change what any
arm is qualified to claim.

## Why this exists

Measured at implementation `eab80e9f` on the official Supa 1.7.0-devfreeze weights, with
the matched ordinary controls run through the identical `compare_paths` grid:

| arm | precision | max abs err | total elementwise violations |
|---|---|---:|---:|
| D40 ordinary | FP32 | 8.1e-5 | **0** |
| B28 official | FP32 | 1.7e-4 | **4** |
| C28 CED | FP32 | 8.2e-4 | **401,863** |
| C40 CED | FP32 | 5.9e-3 | **726,228** |
| B28 official | FP64 | 3.8e-13 | 0 |
| C28 CED | FP64 | 4.1e-12 | 0 |

`causal_suffix_isolation_exact` and `decode_choices_exact` are True for every arm, row and
precision. So the CED transplant is mathematically correct, the strict gate is achievable
because the matched ordinary control passes it outright, and the FP32 failure is
CED-specific rather than generic accumulation noise.

## Structural hypothesis being tested

`DenseReference` calls `dense_attention` at exactly two sites: `SelfBlock` (encoder
self-attention, present in every arm) and `ReadBlock` (the decoder read over the shared
encoder-final K/V bank, present only in CED). In CED, **every** decoder layer contracts
over the **same** bank. A chunk-shape-dependent perturbation therefore strikes all decoder
layers identically and compounds with depth, instead of being independently re-derived per
layer as in an ordinary model. That is a hypothesis consistent with the data, not a proven
mechanism; the receipts continue to record `cause: UNDETERMINED`.

This amendment makes the hypothesis **falsifiable** by raising the accumulation precision of
the shared-memory read alone and measuring the result:

- If the violations collapse, the amplification is created in the read contraction.
- If they persist, the error is inherited from the FP32 encoder and merely amplified by
  sharing, and a read-side fix is the wrong remedy.

Both outcomes are reportable results. Neither authorizes optimization.

## What is added

`dense_attention` gains an optional `accumulate_dtype`. When `None` — the default — every
computation is bit-identical to `eab80e9f`; when set, the score matmul, the softmax and the
probability/value contraction are all performed in that dtype and only the result is cast
back. `ReadBlock.forward` forwards it. `DenseReference` gains the runtime-only attribute
`memory_accumulation_dtype`, alongside the existing `gradient_checkpointing`, and passes it
to the decoder blocks only. `SelfBlock` is untouched, so the encoder is unchanged.

`diagnose-numerics` gains `--memory-precision {inherit,float64}` and records the resolved
setting in `NUMERICS.json`.

## What is explicitly NOT changed

- No tolerance, no acceptance rule, no check name, no gate threshold.
- No default behaviour. `memory_accumulation_dtype` defaults to `None`, so the requalified
  ladder must reproduce the previous violation counts **exactly**. That equality is the
  test that this amendment is inert, and it is asserted in the test suite.
- Not a training or deployment permission. A higher-precision inference lane is not
  evidence about quality, speed, memory or 8-GB fit, and FP64 agreement has never qualified
  FP32 — `HOTFIX_001.md` already forbids that reading and it still holds.
- The attribute is runtime-only and is **not** part of `Config`, so parameter counts,
  `config` equality against `presets()`, saved `config.json` and every budget check are
  unaffected.

## Standing under the existing contract

A CED arm whose strict FP32 parity fails remains unqualified and its optimizer lane remains
closed. If the diagnostic shows that high-precision shared-memory accumulation removes the
violations, that is a candidate **implementation change**, and adopting it as the registered
default would require its own prospective amendment plus a full requalification — it would
not retroactively qualify anything measured here. Adopting it would also change what the
architecture is, and therefore what a later comparison against an ordinary control can
claim, since the ordinary arms need no such lane.

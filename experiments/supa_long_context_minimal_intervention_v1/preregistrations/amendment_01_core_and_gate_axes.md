# Amendment 01 — checkpoint and abstention-gate axes for Phase 0

**Status: follow-up, not preregistered.** Phase 0's primary result was already in
hand when this was requested. It is reported as a robustness replication, and no
result here may be presented as a confirmatory test of a prior hypothesis.

## Correction to the premise

The request was to repeat Phase 0 on "Supa A1.0, the model without the baked
gate". **Supa-A1 is not an ungated model.** Verified from
`Rineya-Supa-A1-1.0.0-devfreeze/config.json`:

| | Supa-A1 (1.0.0) | Supa-A1.1 (1.1.0) |
|---|---|---|
| `architectures` | `Qwen3AbstainForCausalLM` | `Qwen3AbstainForCausalLM` |
| `abstain_token_id` | 51378 | 51378 |
| `abstain_logit_bias` | 5.156249985098839 | 5.156249985098839 |
| `dtype` | **bfloat16** | **float16** |
| `auto_map` module | `modeling_supa_a1` | `modeling_supa_a11` |

The two modeling files are byte-identical (md5 `a3d795d5c827614329554eb7ad66d895`).
A1 is the **bfloat16 store of the same recipe**, not a gate-free variant. Per its
own `CORE.json`, folding the scale-0.8125 LoRA into bfloat16 discards 17.25% of
the delta as rounding noise against float16's 2.19%, which is the entire reason
A1.1 supersedes it.

So "no gate" and "A1" are two different axes, and both are now run.

## The two axes

* **core** — `a11` (float16) or `a1` (bfloat16). Each core is loaded in **its
  own published dtype**. Loading A1 in float16 would re-round bfloat16 weights
  and confound a checkpoint comparison with a precision conversion.
* **gate** — `on`, or `off` via `abstain_logit_bias = 0.0` on the **in-memory
  config only**. The offset is applied in `forward` and is not in the weights,
  so this is the same ungated control the core's own
  `evidence/no_gate_control_*.jsonl` files use. Nothing is written to disk;
  every checkpoint file is hashed before and after each run.

Verified on CPU before running, for both cores: toggling the gate changes the
logit of token 51378 by +5.164 (float16) / +5.125 (bfloat16) — 5.15625 rounded
in each format — and changes **every other logit by exactly 0.0**.

## Why the ungated arm is the more interesting of the two

Phase 0's headline mechanism is an **abstention shift**: on answerable rows the
abstention rate rose 1.0% → 10.9% → 26.0% across full / query-selected / random
context. The gate is a constant +5.15625 bias on the `UNKNOWN` token.

This gives a sharp, falsifiable prediction. If the abstention shift is
substantially *caused* by the gate, then removing it should shrink the shift and
move B − A toward zero. If the shift is a property of the underlying model
reacting to reduced context, removing the gate should leave B − A roughly where
it was.

Either outcome is informative, and the prediction was written down before the
arms were run.

## Arms

Run on the primary demo suite only. The cached conditions from
`results/phase0_conditions/` are reused unchanged, so A/B/C/D inputs are
byte-identical to the primary run; only the model and gate state differ.

| arm | core | gate | published expected (Showcase / Stability / all256) | stage directory |
|---|---|---|---|---|
| baseline (already run) | a11 | on | 45 / 125 / 170 | `phase0_eval` |
| ungated | a11 | off | **41 / 121 / 162** | `phase0_eval_a11_ungated` |
| bfloat16 store | a1 | on | **43 / 125 / 168** | `phase0_eval_a1_gated` |
| both | a1 | off | *none published* | `phase0_eval_a1_ungated` |

Three of the four cells carry a published validity gate; condition A must
reproduce its cell's counts exactly or that arm's contrasts are not interpreted.
The fourth cell has no published figure and is reported without a gate.

## Analysis

Identical to Phase 0: B − A, B − C, D − A, D − B with paired bootstrap CIs and
exact McNemar, plus the abstention-rate table split by answerability.

The added quantity of interest is the **interaction**: does the sign or
magnitude of B − A depend on the gate, or on the storage dtype?

## Claims that will not be made

No significance claims. With four cells on the same 256 rows this is a
multiple-comparison surface, and the corpus is underpowered by its own published
analysis. Agreement across cells will be read as *robustness*, not as
independent confirmation — the rows, the selector and the cached contexts are
shared, so the arms are not independent samples.

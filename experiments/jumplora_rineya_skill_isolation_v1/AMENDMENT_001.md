# Amendment 001 — paper-fidelity correction before native Rineya execution

Status: **mechanical correction; no efficacy run; no sealed access**.

## Why this amendment exists

A post-qualification audit compared the implementation against JumpLoRA v3 Algorithm 1 line-by-line before any Rineya checkpoint was loaded.

The paper defines the low-rank update as `Delta W = A B`, applies JumpReLU and learns `tau` in those **unscaled Delta-W units**, and only then applies the LoRA scale `beta = alpha/r` in the forward pass. The initial implementation's `dense_delta()` helper already included `alpha/r`, and J3 thresholded that scaled tensor. A positive scalar does not change magnitude ordering, so the selected support at a fixed cardinality can look correct, but it **does change the numerical units of tau and epsilon and therefore changes threshold-learning dynamics**. That is not a faithful implementation of Algorithm 1.

The same audit found that the integration helper initialized one global threshold across all injected q/v matrices. The paper motivates each layer learning its own sparsity level. A global cross-layer threshold permits support budget to migrate between layers and therefore confounds support learning with layer-budget allocation.

## Correction

1. `LoRALinear.raw_delta()` now exposes the unscaled `B @ A`.
2. Existing `dense_delta()` semantics are preserved for compatibility: it remains the ordinary LoRA update including `alpha/r`.
3. J3 threshold initialization, JumpReLU gating, and support reporting use `raw_delta()`.
4. `alpha/r` is applied only after dense/sparse interpolation.
5. J3 thresholds are initialized per adapted q/v module.
6. The registered J1/J2 matched-density controls now bind **the same final active count per module** as J3. A global-top-k helper remains only as an explicitly exploratory utility.
7. New paper-fidelity tests establish scale placement, a non-zero hand-checkable STE gradient, per-module threshold initialization, and per-module J1 matching.

## Evidence boundary

The original `QUALIFICATION_RECEIPT.json` remains historical evidence for the pre-amendment code commit recorded inside it. It must not be treated as proving this amendment by itself.

The correction is validated separately by `PAPER_FIDELITY_VALIDATION.json` and the added `tests/test_paper_fidelity.py`. Both are **MECHANICAL_ONLY**. No pretrained Rineya model, tokenizer, training split, development split, held-out split, sealed/ANY_READ asset, or native GPU run was used for the correction.

A native exact-host qualification remains required before any efficacy spend.

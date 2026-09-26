# Protocol — Supa-A1.1 Single-Layer Importance Fast Screen

Protocol version: `15.0.0-exploratory`

Parent: Cross-Scale lab; does **not** amend v1/v2/v3 transport protocols.

## Objective

Fast leave-one-layer-out screen on **Supa-A1.1 only**: for each decoder
layer `0..27`, identity-skip that block (residual passthrough) and measure
whether generation collapses into garbage / repetition / incoherent text.

## Ablation operator

**Identity residual skip.** A forward hook on `model.model.layers[i]` returns
the block input as output, so attention and MLP for that layer contribute
nothing. No checkpoint rewriting. One model load.

## Model

- Path: `Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze`
- `trust_remote_code=True`, CPU, float32, greedy, thinking disabled
- Intact Supa is the sole baseline

## Smoke bank

- `data/evaluation/supa_layer_ablation_smoke.jsonl` (8 stratified calibration
  prompts, seed-ordered family pick, `max_new_tokens=24`)
- SHA-256: `404c866f64f36422e5e9e5d52170a358d6c2dd28d276492a38f348ce18f2050f`

## Critical-layer heuristics (screening only)

A layer is **critical** if any of:

1. `weird_char_rate ≥ 0.25` on ≥2 smoke prompts
2. `invalid_generation` on ≥50% of smoke prompts
3. mean final-token logit cosine vs intact `< 0.5`

## Escalation

- Sealed: **CLOSED**
- MERGEDEV-fast: only if ≥1 critical layer (follow-up; not required to close this screen)
- No donors, merges, or transport reopen

## Hardware

CPU-only; one model resident; `OMP_NUM_THREADS=4`; refuse GPU if busy.

# R2 cross-depth soup scoreboard

CPU-only lab. Scaffold is R2 (28L, vocab 32768). Same Phase-2B kept-layer stitch as published Qwen3 cores.

- Generated: `2026-08-21T05:13:22Z`
- Protocol: greedy, `24` new tokens, context `4096`, seed `17`, thinking disabled, device CPU
- Qwen3 published numbers below are **cross-scaffold references**, not paired scores.

## Qwen3 published family (reference only)

| Model | Hits / 64 | Dual (if recorded) |
|---|---:|---:|
| Qwen3-0.6B | 14/64 | — |
| Nimbus (Qwen3 × Aegis α=0.5) | 24/64 | 0.7814 |
| Lyra | 28/64 | 0.7445 |
| Halo | 30/64 | 0.7381 |
| Halo Uncensored | 31/64 | 0.6823 |
| late_rineya / Pulse 1.6 | 31/64 | 0.7561 |

## R2 lab runs

| Name | Round | Kind | Hits / 64 | Strict exact | Unanswerable | Counterfactual-and-changed | ≤12-word rate |
|---|---|---|---:|---:|---:|---:|---:|
| `r2_baseline` | baseline | scaffold | 11/64 | 0.1719 | 0.0625 | 0.1250 | 0.7344 |
| `r2_x_aegis` | cross_depth | vault | 21/64 | 0.3281 | 0.0000 | 0.2500 | 0.8750 |
| `r2_x_v1_5e4_1e5` | cross_depth | soup | 25/64 | 0.3906 | 0.0000 | 0.3125 | 0.8750 |
| `r2_x_v1_5e4_v2` | cross_depth | soup | 25/64 | 0.3906 | 0.0000 | 0.2500 | 0.8750 |
| `r2_x_v1_5e4_v2_a1` | replace_kept | soup | 27/64 | 0.4219 | 0.0000 | 0.3125 | 1.0000 |
| `r2_late_rineya` | alpha_recipe | soup | 26/64 | 0.4062 | 0.0000 | 0.3125 | 0.8750 |

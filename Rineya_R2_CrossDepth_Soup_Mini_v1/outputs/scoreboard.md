# R2 CrossDepth — CPU showcase + dual

- Generated: `2026-08-21T07:27:38Z`
- Device: **CPU** (`CUDA_VISIBLE_DEVICES` empty).
- Scaffold: local copy of R2 (`rineya-core-en32k-28L`, vocab 32768).
- Showcase: greedy, 24 tok, 4096, seed 17, thinking off, bf16.
- Dual: Phase 1 chat DEV (510) + Phase 9 RAG DEV (752×2).
- Qwen3 family rows are **cross-scaffold references**, not paired.

## Showcase

| Name | Hits / 64 | Strict exact | Unanswerable | CF pair |
|---|---:|---:|---:|---:|
| `r2_baseline` | 11/64 | 0.1719 | 0.0625 | 0.1250 |
| `r2_x_aegis` | 21/64 | 0.3281 | 0.0000 | 0.2500 |
| `r2_x_v1_5e4_1e5` | 25/64 | 0.3906 | 0.0000 | 0.3125 |
| `r2_x_v1_5e4_v2` | 25/64 | 0.3906 | 0.0000 | 0.2500 |
| `r2_x_v1_5e4_v2_a1` | 27/64 | 0.4219 | 0.0000 | 0.3125 |
| `r2_late_rineya` | 26/64 | 0.4062 | 0.0000 | 0.3125 |

## Dual composite

| Model | Dual | Chat Q | Task adequacy | Oracle supp | Real supp | Oracle abstain | Real proc | Showcase |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3 Nimbus (α=0.5 Aegis) (Qwen3 ref) | 0.7814 | — | — | — | — | — | — | 24/64 |
| Qwen3 Lyra (Qwen3 ref) | 0.7445 | — | — | — | — | — | — | 28/64 |
| Qwen3 Halo (Qwen3 ref) | 0.7381 | — | — | — | — | — | — | 30/64 |
| Qwen3 Halo Uncensored (Qwen3 ref) | 0.6823 | — | — | — | — | — | — | 31/64 |
| Qwen3 late_rineya / Pulse 1.6 (Qwen3 ref) | 0.7561 | — | — | — | — | — | — | 31/64 |
| `r2_baseline` | 0.6564 | 0.8316 | 0.4049 | 0.7679 | 0.3868 | 0.9928 | 0.3533 | 11/64 |
| `r2_x_aegis` | 0.7729 | 0.8521 | 0.4744 | 0.8962 | 0.6679 | 0.8273 | 0.8400 | 21/64 |
| `r2_x_v1_5e4_1e5` | 0.7421 | 0.8608 | 0.5146 | 0.8415 | 0.6283 | 0.7410 | 0.7000 | 25/64 |
| `r2_x_v1_5e4_v2` | 0.7353 | 0.8616 | 0.5131 | 0.8472 | 0.6283 | 0.6619 | 0.7000 | 25/64 |
| `r2_x_v1_5e4_v2_a1` | 0.6841 | 0.8622 | 0.5079 | 0.8679 | 0.6509 | 0.0000 | 0.7800 | 27/64 |
| `r2_late_rineya` | 0.7550 | 0.8600 | 0.5026 | 0.8509 | 0.6151 | 0.8777 | 0.7133 | 26/64 |

Lab copies only; not vaulted. Qwen3 published-core numbers are cross-scaffold references (vocab 151936 vs R2 32768).

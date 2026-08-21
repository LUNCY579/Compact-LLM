# late_rineya + global_best ? CPU dual + showcase fill-in

- Generated: `2026-08-21T01:00:04Z`
- Device: **CPU** (`CUDA_VISIBLE_DEVICES` empty) so the Halo-pair soup-lab can keep the GPU.
- Showcase: greedy, 24 tok, 4096, seed 17, thinking off, bf16.
- Dual: Phase 1 chat DEV (510) + Phase 9 RAG DEV (752?2), same weights as family scoreboard.
- global_best note: soup-lab alias global_best = cand_d1b08a8faf299ce4

## Showcase

| Name | Hits / 64 | Strict exact | Unanswerable | CF pair | Source |
|---|---:|---:|---:|---:|---|
| `global_best` | 29/64 | 0.4531 | 0.1250 | 0.3125 | soup-lab alias global_best = cand_d1b08a8faf299ce4 |
| `late_rineya` | 31/64 | 0.4844 | 0.1250 | 0.4375 | prior alpha_recipes sweep |

## Dual composite

| Model | Dual | Chat Q | Task adequacy | Oracle supp | Real supp | Oracle abstain | Real proc | Showcase |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Aegis 19L` | 0.8382 | 0.8594 | 0.4978 | 1.0000 | 0.7358 | 1.0000 | 0.9533 | 4/64 |
| `Nimbus (qwen3_x_aegis)` | 0.7814 | 0.8602 | 0.5032 | 0.8906 | 0.6736 | 0.8273 | 0.8600 | 24/64 |
| `Lyra (qwen3_x_v1_5e4_1e5)` | 0.7445 | 0.8667 | 0.5289 | 0.8415 | 0.6283 | 0.7266 | 0.7000 | 28/64 |
| `Halo (qwen3_x_v1_5e4_v2)` | 0.7381 | 0.8619 | 0.5117 | 0.8472 | 0.6283 | 0.6906 | 0.7000 | 30/64 |
| `Pulse` | ? | 0.7984 | ? | ? | ? | ? | ? | 0/64 |
| `global_best` | 0.7444 | 0.8649 | 0.5204 | 0.8434 | 0.6075 | 0.7770 | 0.6933 | 29/64 |
| `late_rineya` | 0.7561 | 0.8612 | 0.5067 | 0.8509 | 0.6208 | 0.8705 | 0.7133 | 31/64 |

Lab fill-in only. global_best is the soup-lab uniform global α winner so far; layer_influence probes can score higher on chat but do not replace the alias until layer_search/freeze. late_rineya/global_best are not vault cores.

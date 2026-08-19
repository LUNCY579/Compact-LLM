# Teacher soup mini-lab scoreboard

Curiosity-only. Frozen 64-row showcase. Not a recorded core.

- Generated: `2026-08-19T18:49:47Z`
- Protocol: greedy, `24` new tokens, context `4096`, seed `17`

## References

| Model | Hits / 64 | Strict exact | Unanswerable | Counterfactual-and-changed | ≤12-word rate |
|---|---:|---:|---:|---:|---:|
| aegis_showcase | 4/64 | 0.0625 | 0.0000 | 0.0000 | 1.0000 |
| qwen3_06b_showcase | 14/64 | 0.2188 | 0.1875 | 0.1250 | 0.8906 |

## Lab runs

| Name | Round | Kind | Hits / 64 | Strict exact | Unanswerable | Counterfactual-and-changed | ≤12-word rate |
|---|---|---|---:|---:|---:|---:|---:|
| `v1_lr1e5` | A_singles | peft | 4/64 | 0.0625 | 0.0000 | 0.0000 | 0.7500 |
| `v1_lr2e5` | A_singles | peft | 3/64 | 0.0469 | 0.0000 | 0.0625 | 0.6719 |
| `v1_lr5e5` | A_singles | peft | 2/64 | 0.0312 | 0.0000 | 0.0000 | 0.8125 |
| `v1_lr1e4` | A_singles | peft | 2/64 | 0.0312 | 0.0000 | 0.0000 | 0.9531 |
| `v1_lr2e4` | A_singles | peft | 3/64 | 0.0469 | 0.0000 | 0.0000 | 0.8125 |
| `v1_lr3e4` | A_singles | peft | 4/64 | 0.0625 | 0.0000 | 0.0000 | 0.9062 |
| `v1_lr5e4` | A_singles | peft | 4/64 | 0.0625 | 0.0000 | 0.0000 | 1.0000 |
| `v2_lr1e4` | A_singles | peft | 2/64 | 0.0312 | 0.0000 | 0.0000 | 0.8594 |
| `v1_5e4_1e5` | A_pairwise_5e4 | soup | 5/64 | 0.0781 | 0.0000 | 0.0000 | 1.0000 |
| `v1_5e4_2e5` | A_pairwise_5e4 | soup | 4/64 | 0.0625 | 0.0000 | 0.0000 | 1.0000 |
| `v1_5e4_5e5` | A_pairwise_5e4 | soup | 3/64 | 0.0469 | 0.0000 | 0.0000 | 0.9844 |
| `v1_5e4_1e4` | A_pairwise_5e4 | soup | 3/64 | 0.0469 | 0.0000 | 0.0000 | 0.9688 |
| `v1_5e4_2e4` | A_pairwise_5e4 | soup | 2/64 | 0.0312 | 0.0000 | 0.0000 | 0.9531 |
| `v1_5e4_3e4` | A_pairwise_5e4 | soup | 4/64 | 0.0625 | 0.0000 | 0.0000 | 0.9688 |
| `v1_5e4_v2` | B_cross_mix | soup | 5/64 | 0.0781 | 0.0000 | 0.0625 | 1.0000 |
| `v2_lr5e4` | C_v2_5e4 | peft | 0/64 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `v2_5e4_v1_5e4` | C_v2_5e4 | soup | 1/64 | 0.0156 | 0.0000 | 0.0000 | 0.9062 |
| `v2_5e4_v1_1e5` | C_v2_5e4 | soup | 1/64 | 0.0156 | 0.0000 | 0.0000 | 0.9219 |
| `v2_5e4_v2_1e4` | C_v2_5e4 | soup | 0/64 | 0.0000 | 0.0000 | 0.0000 | 0.9062 |

# Qwen3 cross-depth soup scoreboard

Curiosity-only. Frozen 64-row showcase. 28-layer Qwen3 geometry. Not a recorded core.

- Generated: `2026-08-19T20:46:49Z`
- Protocol: greedy, `24` new tokens, context `4096`, seed `17`, thinking disabled

## References

| Model | Hits / 64 | Strict exact | Unanswerable | Counterfactual-and-changed | ≤12-word rate |
|---|---:|---:|---:|---:|---:|
| aegis_showcase | 4/64 | 0.0625 | 0.0000 | 0.0000 | 1.0000 |
| qwen3_06b_showcase | 14/64 | 0.2188 | 0.1875 | 0.1250 | 0.8906 |

## Lab runs

| Name | Round | Kind | Hits / 64 | Strict exact | Unanswerable | Counterfactual-and-changed | ≤12-word rate |
|---|---|---|---:|---:|---:|---:|---:|
| `qwen3_x_aegis` | cross_depth | vault | 24/64 | 0.3750 | 0.1250 | 0.1250 | 0.9062 |
| `qwen3_x_v1_5e4_1e5` | cross_depth | soup | 28/64 | 0.4375 | 0.1250 | 0.3125 | 0.9062 |
| `qwen3_x_v1_5e4_v2` | cross_depth | soup | 30/64 | 0.4688 | 0.1250 | 0.3750 | 0.9062 |
| `qwen3_x_v2_lr5e4` | v2_5e4 | soup | 26/64 | 0.4062 | 0.0000 | 0.3750 | 0.8750 |
| `qwen3_x_v2_5e4_v1_5e4` | v2_5e4 | soup | 26/64 | 0.4062 | 0.0000 | 0.3750 | 0.8906 |
| `qwen3_replace_aegis` | replace_kept | vault | 23/64 | 0.3594 | 0.0000 | 0.3125 | 0.9531 |
| `qwen3_replace_v1_5e4_1e5` | replace_kept | soup | 30/64 | 0.4688 | 0.0000 | 0.3125 | 0.9375 |
| `qwen3_replace_v1_5e4_v2` | replace_kept | soup | 31/64 | 0.4844 | 0.0000 | 0.3750 | 0.9844 |
| `qwen3_x_v2_1e4_5e4` | v2_5e4_pairs | soup | 26/64 | 0.4062 | 0.0000 | 0.3125 | 0.8906 |
| `qwen3_x_v1_1e4_v2_5e4` | v2_5e4_pairs | soup | 24/64 | 0.3750 | 0.0000 | 0.2500 | 0.8750 |

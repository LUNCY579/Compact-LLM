# Halo-donor kept-layer α recipes

Curiosity-only. Same 19 kept layers / 9 dropped Qwen3 layers as Halo. Donor is Halo’s v1@5e-4 + v2@1e-4 soup. Frozen 64-example showcase.

- Generated: `2026-08-20T06:44:50Z`
- Protocol: greedy, `24` new tokens, context `4096`, seed `17`, thinking off, bf16
- Halo reference: `30/64`
- Keep weights if hits ≥ Halo. ±1 hit is noise; +2 hits is discuss-only, not a vault core.

| Name | Family | Mean α | Hits / 64 | Strict exact | Unanswerable | CF pair | Source |
|---|---|---:|---:|---:|---:|---:|---|
| `uniform_0.25` | uniform_dose | 0.2500 | 25/64 | 0.3906 | 0.1250 | 0.3125 | this_sweep |
| `uniform_0.375` | uniform_dose | 0.3750 | 29/64 | 0.4531 | 0.1250 | 0.3750 | this_sweep |
| `uniform_0.5` | uniform_dose | 0.5000 | 30/64 | 0.4688 | 0.1250 | 0.3750 | outputs/qwen3_x_v1_5e4_v2_report.json |
| `uniform_0.625` | uniform_dose | 0.6250 | 25/64 | 0.3906 | 0.0000 | 0.3125 | this_sweep |
| `uniform_0.75` | uniform_dose | 0.7500 | 26/64 | 0.4062 | 0.0000 | 0.3125 | this_sweep |
| `uniform_1.0` | uniform_dose | 1.0000 | 31/64 | 0.4844 | 0.0000 | 0.3750 | outputs/qwen3_replace_v1_5e4_v2_report.json |
| `late_rineya` | shape_mean_0.5 | 0.5000 | 31/64 | 0.4844 | 0.1250 | 0.4375 | this_sweep |
| `early_rineya` | shape_mean_0.5 | 0.5000 | 26/64 | 0.4062 | 0.0000 | 0.3125 | this_sweep |
| `qwen_index_ramp` | shape_mean_0.5 | 0.5000 | 30/64 | 0.4688 | 0.1250 | 0.3750 | this_sweep |
| `keep_importance` | shape_mean_0.5 | 0.4989 | 28/64 | 0.4375 | 0.1250 | 0.3125 | this_sweep |

**Best so far:** `late_rineya` at 31/64.

## Reading

- Halo (`uniform_0.5`) 30/64 and Halo Uncensored (`uniform_1.0`) 31/64 were **reused**, not restitched.
- New uniform blends: 0.25 → 25, 0.375 → 29, 0.625 → 25, 0.75 → 26. Uniform 0.5 still peaks among blend (α<1) doses.
- Mean-0.5 shapes: `late_rineya` 31/64 (+1 vs Halo), `qwen_index_ramp` 30/64, `keep_importance` 28/64, `early_rineya` 26/64.
- `late_rineya` also has the better CF-pair score (0.4375 vs Halo 0.3750) and keeps Halo’s unanswerable 0.1250. That is still **+1 hit**, below the +2 discuss floor. **Not vaulted.** No ±0.1 local bump.
- Lab weights kept for `late_rineya` and `qwen_index_ramp` (hits ≥ 30). Other recipe safetensors dropped.

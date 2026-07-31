# PHASE 2B LAYER-COMBINATION PLAYGROUND REPORT

## Experiment boundary

This is an independent exploratory experiment. It does **not** modify or approve
Phase 2B, does not overwrite either frontier, and does not create a downstream
profile. Raw combination variants are rebuilt from the same full SFT reference
without recovery so that layer-set effects are easier to interpret.

- Suite: `all`
- Probe: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl`
- Cases requested: 150
- Response control: `eos_only`
- Quick drops: `[6, 10, 13, 14, 19, 26]`
- Thorough drops: `[3, 5, 10, 13, 14, 19, 21, 22, 23, 24]`
- Shared drops: `[10, 13, 14, 19]`
- Quick-only drops: `[6, 26]`
- Thorough-only drops: `[3, 5, 21, 22, 23, 24]`
- Full union: `[3, 5, 6, 10, 13, 14, 19, 21, 22, 23, 24, 26]`

## Results

| Variant | Remaining layers | Total original layers dropped | Weight reduction | Heuristic score | Score loss | Semantic pass | Complete utterance | Target NLL | Fuzzy similarity to reference | Critical | Major | Measured latency reduction |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 28 | `[]` | 0.00% | 1.000 | 0.00% | 1.000 | 1.000 | 4.8624 | n/a | 0 | 0 | +0.00% |
| common_only | 24 | `[10, 13, 14, 19]` | 10.56% | 0.993 | 0.67% | 1.000 | 0.993 | 4.6970 | 0.456 | 0 | 0 | +4.58% |
| quick_unique_only | 26 | `[6, 26]` | 5.28% | 1.000 | 0.00% | 1.000 | 1.000 | 4.9768 | 0.522 | 0 | 0 | +4.45% |
| thorough_unique_only | 22 | `[3, 5, 21, 22, 23, 24]` | 15.84% | 0.931 | 6.89% | 1.000 | 1.000 | 5.9880 | 0.359 | 0 | 0 | -39.03% |
| nonoverlap_all | 20 | `[3, 5, 6, 21, 22, 23, 24, 26]` | 21.11% | 0.693 | 30.67% | 1.000 | 0.980 | 7.2155 | 0.217 | 0 | 0 | -193.30% |
| quick_set_raw | 22 | `[6, 10, 13, 14, 19, 26]` | 15.84% | 0.980 | 2.00% | 1.000 | 1.000 | 5.3580 | 0.363 | 0 | 0 | -17.80% |
| thorough_set_raw | 18 | `[3, 5, 10, 13, 14, 19, 21, 22, 23, 24]` | 26.39% | 0.704 | 29.56% | 1.000 | 0.960 | 7.6787 | 0.254 | 0 | 0 | -164.18% |
| union_all | 16 | `[3, 5, 6, 10, 13, 14, 19, 21, 22, 23, 24, 26]` | 31.67% | 0.704 | 29.56% | 1.000 | 0.993 | 10.0903 | 0.234 | 0 | 0 | -144.58% |
| thorough_plus_drop_6 | 17 | `[3, 5, 6, 10, 13, 14, 19, 21, 22, 23, 24]` | 29.03% | 0.704 | 29.56% | 1.000 | 0.993 | 8.2390 | 0.244 | 0 | 0 | -166.91% |
| thorough_plus_drop_26 | 17 | `[3, 5, 10, 13, 14, 19, 21, 22, 23, 24, 26]` | 29.03% | 0.700 | 30.00% | 1.000 | 0.980 | 9.4234 | 0.221 | 0 | 0 | -189.03% |
| union_leave_3_in | 17 | `[5, 6, 10, 13, 14, 19, 21, 22, 23, 24, 26]` | 29.03% | 0.698 | 30.22% | 1.000 | 0.987 | 9.3301 | 0.223 | 0 | 0 | -168.45% |
| union_leave_5_in | 17 | `[3, 6, 10, 13, 14, 19, 21, 22, 23, 24, 26]` | 29.03% | 0.727 | 27.33% | 1.000 | 0.987 | 9.2607 | 0.236 | 0 | 0 | -153.34% |
| union_leave_6_in | 17 | `[3, 5, 10, 13, 14, 19, 21, 22, 23, 24, 26]` | 29.03% | 0.700 | 30.00% | 1.000 | 0.980 | 9.4234 | 0.221 | 0 | 0 | -156.38% |
| union_leave_10_in | 17 | `[3, 5, 6, 13, 14, 19, 21, 22, 23, 24, 26]` | 29.03% | 0.704 | 29.56% | 1.000 | 0.987 | 8.9629 | 0.229 | 0 | 0 | -171.02% |
| union_leave_13_in | 17 | `[3, 5, 6, 10, 14, 19, 21, 22, 23, 24, 26]` | 29.03% | 0.704 | 29.56% | 1.000 | 0.993 | 9.9180 | 0.200 | 0 | 0 | -164.10% |
| union_leave_14_in | 17 | `[3, 5, 6, 10, 13, 19, 21, 22, 23, 24, 26]` | 29.03% | 0.689 | 31.11% | 1.000 | 0.993 | 9.2590 | 0.213 | 0 | 0 | -169.14% |
| union_leave_19_in | 17 | `[3, 5, 6, 10, 13, 14, 21, 22, 23, 24, 26]` | 29.03% | 0.698 | 30.22% | 1.000 | 0.993 | 8.6871 | 0.234 | 0 | 0 | -164.62% |
| union_leave_21_in | 17 | `[3, 5, 6, 10, 13, 14, 19, 22, 23, 24, 26]` | 29.03% | 0.684 | 31.56% | 1.000 | 0.947 | 9.0536 | 0.203 | 0 | 0 | -161.42% |
| union_leave_22_in | 17 | `[3, 5, 6, 10, 13, 14, 19, 21, 23, 24, 26]` | 29.03% | 0.704 | 29.56% | 1.000 | 0.993 | 8.5693 | 0.232 | 0 | 0 | -161.36% |
| union_leave_23_in | 17 | `[3, 5, 6, 10, 13, 14, 19, 21, 22, 24, 26]` | 29.03% | 0.696 | 30.44% | 1.000 | 0.987 | 8.8519 | 0.243 | 0 | 0 | -181.61% |
| union_leave_24_in | 17 | `[3, 5, 6, 10, 13, 14, 19, 21, 22, 23, 26]` | 29.03% | 0.689 | 31.11% | 1.000 | 0.927 | 8.2318 | 0.204 | 0 | 0 | -161.52% |
| union_leave_26_in | 17 | `[3, 5, 6, 10, 13, 14, 19, 21, 22, 23, 24]` | 29.03% | 0.704 | 29.56% | 1.000 | 0.993 | 8.2390 | 0.244 | 0 | 0 | -167.44% |
| quick_actual_endpoint | 22 | `[6, 10, 13, 14, 19, 26]` | 15.84% | 0.998 | 0.22% | 1.000 | 1.000 | 3.5858 | 0.434 | 0 | 0 | +0.81% |
| thorough_actual_stage06 | 22 | `[3, 13, 14, 19, 23, 24]` | 15.84% | 0.996 | 0.44% | 1.000 | 1.000 | 3.5186 | 0.456 | 0 | 0 | -7.28% |
| thorough_actual_endpoint | 18 | `[3, 5, 10, 13, 14, 19, 21, 22, 23, 24]` | 26.39% | 1.000 | 0.00% | 1.000 | 1.000 | 3.9416 | 0.424 | 0 | 0 | +6.91% |
| thorough_actual_plus_drop_6 | 17 | `[3, 5, 6, 10, 13, 14, 19, 21, 22, 23, 24]` | 29.03% | 0.993 | 0.67% | 1.000 | 1.000 | 4.1630 | 0.378 | 0 | 0 | +32.63% |
| thorough_actual_plus_drop_26 | 17 | `[3, 5, 10, 13, 14, 19, 21, 22, 23, 24, 26]` | 29.03% | 0.958 | 4.22% | 1.000 | 1.000 | 4.9661 | 0.357 | 0 | 0 | -17.55% |
| thorough_actual_plus_drop_6_26 | 16 | `[3, 5, 6, 10, 13, 14, 19, 21, 22, 23, 24, 26]` | 31.67% | 0.953 | 4.67% | 1.000 | 1.000 | 5.1401 | 0.339 | 0 | 0 | -4.91% |

## How to interpret the key controls

- `common_only` tests the four layers that both searches repeatedly selected.
- `nonoverlap_all` tests only the disagreement set.
- `quick_set_raw` and `thorough_set_raw` remove recovery as a confounder.
- `union_all` asks whether every layer selected by either greedy path can be
  removed simultaneously. Failure here would show interaction, not that either
  original path was invalid.
- `thorough_plus_drop_26`, when included, isolates the marginal effect of adding
  layer 26 to the thorough structural set.
- Variants whose names contain `actual` preserve recovery updates from the saved
  endpoint and should not be treated as raw structural controls.

Measured latency is secondary and noisy. The playground does not use latency to
rank or accept candidates. Inspect each variant's `OUTPUT_DEVIATION.md` and
case-level `generation/results.jsonl` before drawing a quality conclusion.

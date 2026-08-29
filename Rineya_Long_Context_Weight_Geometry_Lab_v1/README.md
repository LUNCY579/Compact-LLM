# Rineya Long-Context Weight-Geometry Lab v1

This lab records a bounded search for higher exact-match performance on the frozen `Rineya_Long_Context_Demo_v1` suite. It does not modify published model checkpoints or train on Demo rows.

## Outcome

The best single model is published **Nimbus 1.6** plus the disjoint long-context Stage-A checkpoint-200 LoRA:

- scale `0.8125`: **42/64** Showcase, **121/192** Stability, **163/256** overall;
- scale `1.0625`: **42/64** Showcase, emphasizing abstention over paired paraphrase robustness.

The original best family recipe runs Nimbus + adapter at scales `0.8125` and `1.25`, returning the high-scale result only when it normalizes to `UNKNOWN`. The expanded published-family search improved this twice:

- **Baya 2.0 System46**, a three-arm Pulse/Nimbus cascade: 46/64 Showcase, 125/192 Stability, 171/256 overall;
- **Supa 2.0 System48**, a nine-arm ordered consensus cascade: **48/64**, **128/192**, **176/256**.

Every selector consumes predictions only and never reads answer keys at inference time.

| Split | Exact | BP both correct | CF pair | UNKNOWN | All-four packs |
|---|---:|---:|---:|---:|---:|
| Showcase 64 (selection) | **48/64 (75.0%)** | 13/16 | 11/16 | 9/16 | 7/16 |
| Stability 192 (frozen validation) | **128/192 (66.7%)** | 36/48 | 27/48 | 19/48 | 14/48 |
| Complete 256 | **176/256 (68.8%)** | 49/64 | 38/64 | 28/64 | 21/64 |

Under the identical runner, unmodified Nimbus scores 30/64 Showcase, 83/192 Stability, and 113/256 overall, with zero complete packs. The search ceiling adds 18, 45, and 63 exact hits respectively and reaches 21/64 complete packs.

The four frozen tiers are now backed up as Supa System48, Baya System46,
Nimbus DualScale PEFT Gate45, and Supa-Baya Dense Gate42. Only the last uses
the two dense published cores; the others require exact PEFT/task-vector arms.

The Showcase figure is post-selection and must not be described as an untouched test result. The Stability recipe was frozen before its first run and improved there as well.

## What was tested

Across the original and expanded family searches, 147 complete 64-row reports with unique run names were scored under one protocol. The search covered:

- published Base/Halo/Halo-Uncensored/Lyra/Nimbus/Pulse baselines and a retained layer-wise soup;
- long-context LoRA checkpoints 200, 225, and 250;
- global LoRA-delta scales from `0.5` through `1.25`, including fine local refinement;
- attention/MLP and layer-depth scale profiles;
- exact checkpoint SWA, 200/250 weighted soups, and a negative late-training vector;
- structured top-component LoRA rank pruning at ranks 12 and 8;
- transfer of the learned delta to Halo Uncensored, Pulse, Lyra, Halo, and Nimbus;
- native 19-layer Aegis/Pulse dense interpolation;
- answer-blind UNKNOWN gates, modal votes, and ordered consensus cascades.

The main empirical finding is that the long-context delta is highly anchor-dependent. Applying checkpoint 200 to Halo at its best scale reached 39/64; applying the same delta near Nimbus reached 42/64. Convex checkpoint soups, continuation checkpoints, and rank pruning did not improve the frontier. A specific Pulse negative vector, `1.25*delta200 - 0.25*delta250`, did improve Pulse to 41/64 and exposed complementary abstention behavior.

See [`PUBLISHED_FAMILY_SEARCH_REPORT.md`](PUBLISHED_FAMILY_SEARCH_REPORT.md) for the eight-core baselines, publication parity, stop rules, and final cascade definitions.

## Frozen protocol

- BF16 model load
- greedy decoding, seed 17
- Qwen thinking disabled
- 24 generated tokens
- 4096-token context limit
- no truncation
- canonical normalized exact scorer

## Reproduce the selected arms

From the repository root:

```powershell
$python = ".\.venv\Scripts\python.exe"
$runner = ".\Rineya_Long_Context_Demo_v1\scripts\run_local_hf.py"
$model = ".\Rineya_Published_Cores\Rineya-Nimbus-1.6.0-devfreeze"
$adapter = ".\artifacts\rineya_long_context_weight_geometry_v1\selected\halo_stage_a_step200_adapter"

& $python $runner --model $model --adapter $adapter --adapter-scale 0.8125 `
  --input .\Rineya_Long_Context_Demo_v1\data\all_prompts.jsonl `
  --output .\artifacts\rineya_long_context_weight_geometry_v1\reproduction\low.jsonl `
  --max-new-tokens 24 --max-context 4096 --disable-thinking `
  --torch-dtype bfloat16 --max-gpu-memory 2GiB --max-cpu-memory 32GiB

& $python $runner --model $model --adapter $adapter --adapter-scale 1.25 `
  --input .\Rineya_Long_Context_Demo_v1\data\all_prompts.jsonl `
  --output .\artifacts\rineya_long_context_weight_geometry_v1\reproduction\high.jsonl `
  --max-new-tokens 24 --max-context 4096 --disable-thinking `
  --torch-dtype bfloat16 --max-gpu-memory 2GiB --max-cpu-memory 32GiB

& $python .\Rineya_Long_Context_Weight_Geometry_Lab_v1\scripts\build_abstention_scale_ensemble.py `
  --low .\artifacts\rineya_long_context_weight_geometry_v1\reproduction\low.jsonl `
  --high .\artifacts\rineya_long_context_weight_geometry_v1\reproduction\high.jsonl `
  --output .\artifacts\rineya_long_context_weight_geometry_v1\reproduction\ensemble.jsonl
```

The two standalone selected adapters encode those scales directly as LoRA alphas 26 and 40, so they can also be loaded without `--adapter-scale`.

## Tooling

- `build_exact_adapter_soup.py` constructs exact weighted LoRA-delta sums, including negative coefficients, without dense model materialization.
- `build_abstention_scale_ensemble.py` applies the answer-blind two-scale rule and writes a provenance manifest.
- `build_consensus_cascade.py` applies ordered UNKNOWN and non-UNKNOWN consensus rules with per-row decision traces.
- `run_adapter_scale_sweep.py` evaluates many scales from one model load under the frozen protocol.
- `build_dense_lerp.py` provides a guarded streaming dense LERP wrapper for shape-compatible checkpoints.
- `aggregate_showcase_results.py` refuses to rank partial-coverage runs.
- `combine_prediction_splits.py` combines disjoint prediction files with duplicate-ID protection.
- `run_local_hf.py` now supports finite global adapter scaling and validated, fingerprinted per-layer/per-projection scale profiles.

Generated predictions, reports, selected adapters, manifests, and the full scoreboard are under `artifacts/rineya_long_context_weight_geometry_v1/`.

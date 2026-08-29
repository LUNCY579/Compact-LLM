# Rineya TGSA retrofit — operator and agent runbook

Every command fails closed. All commands are development-only: none reads the
Phase-1 confirmatory bank.

```powershell
$ErrorActionPreference = "Stop"
Set-Location C:\Users\clclc\BusinessProject\Chat-Bubble-rineya-retrofit\Rineya_TGSA_Retrofit_v1
python -m pip install -e ".[dev]"

$SourceConfig = ".\tests\fixtures\rineya_supa_a1_config.json"
$Evidence     = ".\evidence"
```

## 1. Doctor — correctness gate

```powershell
python -m pytest .\tests -q --tb=short --durations=15
```

44 checks: bitwise exactness against an independent parallel GQA reference,
RoPE rotation and relativity, GQA grouping, KV-cache layout, cached-vs-uncached
agreement, prefill/decode agreement, causal-prefix and padding invariance,
batch-size consistency, mixed precision, save/reload identity, gradient
connectivity, conversion reproducibility, and manifest tamper detection.

## 2. Conversion

```powershell
python .\scripts\convert_rineya.py `
  --source-checkpoint <PATH_TO_SAFETENSORS_OR_DIR> `
  --source-config $SourceConfig `
  --convert-layers "0,7,14,21" `
  --trajectory-rank 32 `
  --master-seed 20260829 `
  --output (Join-Path $Evidence "converted") `
  --manifest (Join-Path $Evidence "conversion_manifest.json")
```

Reuses `q_proj`, `k_proj`, `v_proj`, `o_proj` verbatim. Refuses a shape
mismatch, a silently dropped pretrained bias, an out-of-range layer, or a config
whose RoPE convention it cannot reproduce.

## 3. Conversion validation

```powershell
python .\scripts\verify_conversion.py `
  --manifest (Join-Path $Evidence "conversion_manifest.json") `
  --source-checkpoint <PATH> `
  --converted (Join-Path $Evidence "converted")
```

Re-derives every hash and fails on any drift in a reused tensor.

## 4. Conversion-damage report — run BEFORE training

```powershell
python .\scripts\run_conversion_damage.py `
  --source-config $SourceConfig `
  --output (Join-Path $Evidence "conversion_damage.json") `
  --layers 6 --convert-every 2
```

Reports token-level KL, NLL change, top-1/top-5 agreement, output and per-layer
drift, and norm ratio at float64/float32/bfloat16. No pass threshold is
asserted; the full distribution is produced.

## 5. Layer-placement development study

```powershell
python .\scripts\run_placement_study.py `
  --source-config $SourceConfig `
  --output (Join-Path $Evidence "layer_placement.json") `
  --steps 400 --seeds "1,2,3"
```

Runs the registered ladder: `none` → five single layers sampled across depth →
distributed → contiguous early/middle/late blocks → half → all.

## 6. Warm-start / freeze-unfreeze development study

```powershell
python .\scripts\run_warm_start_study.py `
  --source-config $SourceConfig `
  --output (Join-Path $Evidence "warm_start.json") `
  --steps 400 --seeds "1,2,3"
```

Compares the progressive schedule (`trajectory` → `attention` →
`converted_layers` → `all`) against `immediate_full`. Records learning curves and
the trainable-parameter count at every stage transition.

## 7. Training and resume

```powershell
python .\scripts\train_retrofit.py `
  --config .\configs\rineya_supa_a1_bake.json `
  --output-dir <RUN_DIR> --schedule progressive --steps 4000

python .\scripts\train_retrofit.py `
  --config .\configs\rineya_supa_a1_bake.json `
  --output-dir <RUN_DIR> --resume
```

## 8. Generation test

```powershell
python .\scripts\generation_test.py `
  --converted (Join-Path $Evidence "converted") `
  --source-config $SourceConfig `
  --prompt "Explain why the sky is blue." --max-new-tokens 64 --greedy `
  --compare-cached-vs-uncached
```

The published core requires greedy decoding and `trust_remote_code=True`.

## 9. Trajectory tracing

```powershell
python .\scripts\trace_retrofit.py `
  --converted (Join-Path $Evidence "converted") `
  --source-config $SourceConfig `
  --prompt "..." --layer 14 --report (Join-Path $Evidence "trace_l14.json")
```

Per layer and per sequential head: carrier summary, velocity, acceleration
(where present), encoded control, control/state/predictor/correction gates,
attention entropy, significance score, and top-token evolution. Summaries by
default; `--dump-tensors` opts into full tensors.

## 10. Evidence verification

```powershell
python .\scripts\verify_conversion.py --manifest ... --strict
python -m pytest .\tests -q
```

## Forensic recovery from one failed model

| question | artifact |
|---|---|
| source model, config, core id | `conversion_manifest.json` → `source_*` |
| source commit / branch | run manifest → `git` |
| converted layer indices | `conversion_manifest.json` → `converted_layers` |
| trajectory architecture | `conversion_manifest.json` → `retrofit_config` |
| which tensors were reused vs new | `layers[].reused_tensors`, `layers[].new_tensor_groups` |
| dataset hashes | training config → `dataset` |
| seed, step, optimizer state | latest checkpoint payload |
| resume ancestry | run manifest → `training.resumed_from_step`, `train_log_repair` |
| gradient / trajectory / gate statistics | `trace_retrofit.py` output |
| prediction traces | `generation_test.py` output |

## Status of these commands

Steps 1, 4, and the placement/warm-start ladders are implemented and exercised.
Steps 2, 3, 7, 8, 9 depend on `scripts/convert_rineya.py`,
`train_retrofit.py`, `generation_test.py`, and `trace_retrofit.py`, which are
**not yet written** — the Rineya-Supa-A1 weights are not cached locally, so
there is nothing for them to load. The library functions they will call
(`convert_checkpoint`, `verify_conversion_manifest`, `apply_stage`,
`measure_conversion_damage`) are implemented and tested.

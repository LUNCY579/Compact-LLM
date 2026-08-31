# Rineya TGSA retrofit — operator and agent runbook

Every command fails closed. All commands are development-only: none reads the
Phase-1 confirmatory bank.

## Two-ledger production flow

The historical harvest and shipping decisions are deliberately separate:

```text
historical empirical positive
  -> PRODUCTION_TRIAL_QUEUE (worth one bounded transfer; never promotable)
  -> TRANSFER_POSITIVE against the exact modern parent
  -> candidate_ledger.json ten-gate production validation
  -> PROMOTE only when every shipping gate is true
```

Validate both ledgers before any trial:

```powershell
python .\scripts\verify_production_trial_queue.py
python .\scripts\verify_candidate_ledger.py
```

The completed historical trial registrations are:

- `configs/historical_soup_a11_baya_midpoint_trial.json`
- `configs/historical_dose_a11_1p0625_trial.json`
- `configs/historical_pulse_vector_a11_trial.json`

Their coefficients, parents, artifacts, evaluation hashes, and no-tie gates
must never be changed to rescue the recorded outcomes.

```powershell
$ErrorActionPreference = "Stop"
Set-Location C:\Users\clclc\BusinessProject\Chat-Bubble-rineya-retrofit\Rineya_TGSA_Retrofit_v1
python -m pip install -e ".[dev]"

$SourceConfig = ".\tests\fixtures\rineya_supa_a1_config.json"
$Evidence     = ".\evidence"
$ModelDir    = "<PATH_TO_EXACT_PARENT_MODEL_DIRECTORY>"
```

## Historical transfer reproduction

The streaming builders refuse an existing destination and fail closed on every
registered source hash. The dose/vector builder materializes its control first
and refuses the candidate unless all 310 parent tensors match bitwise.

```powershell
# Rank 1: 50/50 A1.1/Baya soup
python .\scripts\historical_soup_trial.py preflight
python .\scripts\historical_soup_trial.py build
python .\scripts\historical_soup_trial.py verify
python .\scripts\historical_soup_trial.py smoke-load
python .\scripts\summarize_historical_soup_trial.py

# Rank 2: exact checkpoint-200 dose 1.0625
python .\scripts\historical_dose_trial.py preflight `
  --config .\configs\historical_dose_a11_1p0625_trial.json
python .\scripts\historical_dose_trial.py build `
  --config .\configs\historical_dose_a11_1p0625_trial.json
python .\scripts\historical_dose_trial.py smoke-load `
  --config .\configs\historical_dose_a11_1p0625_trial.json
python .\scripts\summarize_historical_dose_trial.py `
  --config .\configs\historical_dose_a11_1p0625_trial.json

# Rank 3: exact selected Pulse negative vector
python .\scripts\historical_dose_trial.py preflight `
  --config .\configs\historical_pulse_vector_a11_trial.json
python .\scripts\historical_dose_trial.py build `
  --config .\configs\historical_pulse_vector_a11_trial.json
python .\scripts\historical_dose_trial.py smoke-load `
  --config .\configs\historical_pulse_vector_a11_trial.json
python .\scripts\summarize_historical_dose_trial.py `
  --config .\configs\historical_pulse_vector_a11_trial.json
```

The inference invocations themselves are pinned in each registration. They use
the same exact A1.1 parent outputs, greedy seed 17, float16, thinking disabled,
4096 context, 24 generated tokens, `logits_to_keep=1`, and the fixture-verified
portable S4 logit bias. Raw predictions and large trial checkpoints remain under
ignored `artifacts/`; compact outcome/provenance summaries are committed under
`evidence/`.

## 1. Doctor — correctness gate

```powershell
python -m pytest .\tests -q --tb=short --durations=15
```

The suite covers strict numerical exactness against an independent parallel GQA reference,
RoPE rotation and relativity, GQA grouping, KV-cache layout, cached-vs-uncached
agreement, prefill/decode agreement, causal-prefix and padding invariance,
batch-size consistency, mixed precision, save/reload identity, gradient
connectivity, conversion reproducibility, manifest tamper detection, and the
fail-closed production promotion predicate. The real Qwen adapter separately
checks bitwise equality against transformers' native eager implementation.

## 2. Conversion

```powershell
python .\scripts\convert_rineya.py `
  --model-dir $ModelDir `
  --output-dir (Join-Path $Evidence "converted") `
  --layers "0,7,14,21,27" `
  --trajectory-rank 32 `
  --master-seed 20260829 `
  --dtype float32 `
  --attn-impl eager `
  --save-weights
```

Reuses `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, and `k_norm` verbatim.
It refuses an out-of-range layer, an unknown conditioning target, a config whose
RoPE convention it cannot reproduce, or non-neutral conditioning. The current
output is a manifest plus optional converted-attention state, not a standalone
Hugging Face checkpoint; that is a documented production gap, not a baked model.

## 3. Conversion validation

There is no `scripts/verify_conversion.py` in this branch. Conversion performs
native-tensor completeness and neutrality checks in-process, while
`build_source_manifest.py`, `real_layer_exactness.py`, and
`real_conversion_damage.py` provide the separately persisted source/fidelity
evidence. Do not claim independent artifact reload verification until a
self-contained artifact writer and verifier exist.

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

The candidate generator and unit tests exist in `rineya_tgsa.schedule`, but no
`scripts/run_placement_study.py` driver or real-model result exists. The intended
ladder is `none` → depth-sampled single layers → distributed → contiguous
early/middle/late blocks → half → all. Research evidence must select a placement
before a production bake; this branch must not launch an exploratory sweep.

## 6. Warm-start / freeze-unfreeze development study

`FreezeSchedule` and `apply_stage` are implemented and unit-tested, but no
`scripts/run_warm_start_study.py` driver or real-model result exists. The
development-only `train_retrofit.py` exposes bounded R0/R1/R2 stages; it is not
an evidence-bearing bake or a substitute for a positive source configuration.

## 7. Training and resume

```powershell
python .\scripts\train_retrofit.py `
  --model-dir $ModelDir `
  --output-dir <RUN_DIR> --stage r0 `
  --layers "0,14,27" --steps 40

python .\scripts\train_retrofit.py `
  --model-dir $ModelDir `
  --output-dir <RUN_DIR> --resume
```

## 8. Generation test

```powershell
python .\scripts\generation_test.py `
  --model-dir $ModelDir `
  --output (Join-Path $Evidence "generation_test.json") `
  --layers "0,7,14,21,27" `
  --max-new-tokens 12 --dtype float32 --attn-impl eager
```

The published core requires greedy decoding and `trust_remote_code=True`.

## 9. Trajectory tracing

```powershell
python .\scripts\trace_retrofit.py `
  --model-dir $ModelDir `
  --prompt "The capital of France is" `
  --layers "14" --trace-layers "14" `
  --report (Join-Path $Evidence "trace_l14.json")
```

Per selected layer and sequential head this records carrier, velocity, encoded
control, conditioning magnitude, attention entropy, and top-token evolution.
Summaries are the default; `--dump-tensors` is opt-in.

## 10. Evidence verification

```powershell
python .\scripts\verify_candidate_ledger.py
python -m pytest .\tests -q
```

## Forensic recovery from one failed model

| question | artifact |
|---|---|
| source model and config | `conversion_manifest.json` → `source` |
| source commit / branch | `candidate_ledger.json` → `source_branch`, `source_commit` |
| converted layer indices | `conversion_manifest.json` → `conversion.converted_layers` |
| trajectory architecture | `conversion_manifest.json` → `retrofit_config` |
| which tensors were reused vs new | `conversion.layers[].reused_tensors`, trajectory hashes |
| research evidence | `candidate_ledger.json` → `research_evidence` |
| seed, step, optimizer state | latest checkpoint payload |
| resume ancestry | R0/R1/R2 report and `trajectory_state.pt` |
| gradient / trajectory / gate statistics | `trace_retrofit.py` output |
| prediction traces | `generation_test.py` output |

## Status of these commands

Steps 1, 2, 4, 7, 8, 9, candidate-ledger verification, and historical trial-
queue verification are implemented.
Real Supa-A1 conversion evidence exists for the older BF16 A1 parent. Steps 3,
5, and 6 lack standalone drivers/artifacts. No command currently writes and
fresh-process reloads a self-contained trained retrofit checkpoint, benchmarks
serving, packages a release, or smoke-tests the exact artifact in a production
demo. Supa-A1.1 and Baya weights are absent from this worktree but have been
located and hash-verified in the sibling canonical ChatBubble checkout. The
strict shipping ledger contains no `ELIGIBLE` model; the separate historical
queue now authorizes only the frozen bounded midpoint-soup transfer trial.

# Halo Long-Context QLoRA Fine-Tune Lab v1

Resume-safe **QLoRA SFT** of published **Halo 1.2.0** on the long-context CC0 corpus (Stage A: 2K+4K plus short-response replay). Published Halo weights are **never overwritten**. Adapters land under `artifacts/halo_long_context_sft_v1/`.

## Babysit — Stage A

From the Chat-Bubble repo root:

```powershell
$env:PYTHONUTF8 = "1"

# 1. Build the 2K/4K + ~20% short-replay mix (or all available short rows)
.\rineya_halo_long_context_sft_v1\scripts\run_stage_a_train.ps1

# After Ctrl+C, OOM, or a crash:
.\rineya_halo_long_context_sft_v1\scripts\resume_stage_a.ps1

# Checkpoint / adapter status
.\.venv\Scripts\python.exe rineya_halo_long_context_sft_v1\scripts\train_halo_qlora.py --status
```

Resume uses Hugging Face `checkpoint-*` directories (`save_steps=50`, keep last 3) via `--resume-from-checkpoint latest`.

## Evaluate long-context understanding

After an adapter exists:

```powershell
.\rineya_halo_long_context_sft_v1\scripts\run_stage_a_eval.ps1
```

That writes Halo baseline vs Halo+adapter scores on Stage A validation and ≤4K main test (exact + trim match, by stratum). Predictions are `{id, response}` JSONL and are resumable with `--resume`.

Sealed scoring runs only when `rineya_long_context_cc0_v1_main/sealed_eval/prompts/prompts.jsonl` and `.../answer_key/answers.jsonl` are present. If they are missing, the eval prints `SKIPPED_SEALED_BANK_MISSING`. To generate the bank locally:

```powershell
.\rineya_halo_long_context_sft_v1\scripts\run_stage_a_eval.ps1 -BuildSealed
```

Do not train on anything under `sealed_eval/`.

## Layout

| Path | Role |
|---|---|
| `configs/stage_a_qlora.yaml` | Locked Stage A hyperparams |
| `configs/stage_b_qlora.yaml` | 8K stub (not run in v1) |
| `configs/stage_c_qlora.yaml` | 16K stub (VRAM-limited) |
| `scripts/build_stage_mix.py` | Shard load, 2k/4k filter, short replay |
| `scripts/train_halo_qlora.py` | Babysit wrapper (status / resume / dry-run) |
| `scripts/train_halo_engine.py` | Lazy-tokenizing QLoRA Trainer (long-context safe) |
| `scripts/predict_long_context.py` | Greedy decode, resume by id |
| `scripts/run_stage_a_eval.py` | Baseline vs adapter + sealed scorer |
| `artifacts/halo_long_context_sft_v1/stage_a/` | Mix, checkpoints, adapter (gitignored) |

Base model: `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze`.

## Stage A settings

- QLoRA r=16, alpha=32, all layers, attention+MLP, gradient checkpointing
- `max_sequence_length=4096`, batch 1 × accum 8, 1 epoch, seed 11
- Train: CC0 `data/sft/train` strata `{2k,4k}` + short replay from `short-response-model-phase-0-1-2/data/train/train.jsonl` targeting 20% of the final mix (capped by the short pool)
- Val for Trainer: 64-row subsample so babysitting stays tractable
- Decode: greedy, 32 new tokens, thinking disabled

## Later stages (config only)

- **Stage B:** 8192 context, add 8k stratum
- **Stage C:** 16k if VRAM allows; 32k is a tiny ablation
- Optional DPO on `data/dpo/train` after the SFT gate

## Smoke

```powershell
.\.venv\Scripts\python.exe rineya_halo_long_context_sft_v1\scripts\smoke_babysit.py
```

Runs unit tests, a tiny mix, a dry-run train command, adapter-directory validation, and (if CUDA + Halo weights are present) a 2-step train plus resume to step 3.

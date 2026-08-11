# v2.2 Model-Soup Micro-Experiment

Standalone side project next to `short-response-model-phase-0-1-2`.

**Question:** does uniform weight-averaging of compatible v2.2 checkpoints help (Wortsman-style model soups), without touching the official v2.3 runner?

## Soups built

1. `anchor_greedy_uniform` — average of `frozen_anchor` + `greedy_soup` full checkpoints
2. `specialists_lora_uniform` — `W_anchor + mean(Delta_specialist)` over Track A/B + Track C specialists
3. `kitchen_sink_full_uniform` — average of full checkpoints: anchor, greedy, and each specialist materialized onto the anchor

All merges run on **CPU only** so they do not crowd a live GPU Phase job.

## Relation to official Phase 4 soup

Phase 4 (`merge_delta_soup.py`) averages **reconstructed FP32 adapter deltas**:
`W = W_base + Σ c_i Δ_i` with `Σ c_i = 1`.
That is algebraically the same as averaging the materialized full models when they share one base.
`specialists_lora_uniform` targets that same endpoint via the official v2.2 LoRA composer (`weight=1/N` each).
`anchor_greedy_uniform` / kitchen-sink are Wortsman full-checkpoint averages; greedy is not an independent same-base adapter ingredient in the Phase 4 sense.
See sibling project `../v22-specialist-mean-dose-sweep` for an explicit Phase-4 merger + dose sweep.

## Recorded commands

Run from this folder. Prefer the repo `.venv`. Keep CUDA hidden so these stay CPU-only.

```powershell
cd C:\Users\clclc\BusinessProject\Chat-Bubble\v22-model-soup-micro-experiment
$env:CUDA_VISIBLE_DEVICES=""
$py = "C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe"
```

### 1) Build the three baseline soups

```powershell
& $py .\scripts\build_v22_soups.py
# rebuild from scratch:
& $py .\scripts\build_v22_soups.py --force
# skip the large kitchen-sink average:
& $py .\scripts\build_v22_soups.py --skip-kitchen-sink
```

### 2) Smoke eval (truncated DEV slices)

```powershell
& $py .\scripts\smoke_eval_cpu.py --limit 48
```

### 3) Full official DEV eval (4050 + 4860 + 1320 rows)

```powershell
& $py .\scripts\smoke_eval_cpu.py --full --resume
# optional filters:
& $py .\scripts\smoke_eval_cpu.py --full --resume --benchmark multidomain_dev
& $py .\scripts\smoke_eval_cpu.py --full --resume --candidate specialists_lora_uniform --candidate greedy_soup
```

Outputs: `artifacts/full_eval/FULL_COMPARISON.md`

### 4) 8-mode toxic-exclusion matrix

Configured in `configs/exclusion_matrix.json`:

- Always keep: Track B, abstention, grounding, social
- Toxic trio (from v2.2 negative margins): `layer_separated`, `track_a_broad`, `specialist_general`
- 4 drop modes: drop exactly one of the trio, or drop all three
- × procedure toggle (`specialist_procedure` yes/no) → **8 soups**
- Live report: `artifacts/exclusion_matrix/PROGRESS_REPORT.md`
- Densified adapters are shared once under `artifacts/exclusion_matrix/_materialized/` (not copied per mode)

```powershell
# smoke (48 rows/bench), resumable
.\scripts\run_exclusion_matrix.ps1 -ReuseExisting

# full DEV, resumable
.\scripts\run_exclusion_matrix.ps1 -ReuseExisting -Full

# one mode only
.\scripts\run_exclusion_matrix.ps1 -ReuseExisting -Mode drop_all_toxic__proc_no

# equivalent python entrypoint
& $py .\scripts\run_exclusion_matrix.py --ReuseExisting
& $py .\scripts\run_exclusion_matrix.py --ReuseExisting --full
```

Artifacts land under `artifacts/`. Official Phase datasets are referenced read-only from `../short-response-model-phase-0-1-2`.

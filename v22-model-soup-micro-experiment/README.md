# v2.2 Model-Soup Micro-Experiment

Standalone side project next to `short-response-model-phase-0-1-2`.

**Question:** does uniform weight-averaging of compatible v2.2 checkpoints help (Wortsman-style model soups), without touching the official v2.3 runner?

## Soups built

1. `anchor_greedy_uniform` — average of `frozen_anchor` + `greedy_soup` full checkpoints
2. `specialists_lora_uniform` — `W_anchor + mean(Delta_specialist)` over Track A/B + Track C specialists
3. `kitchen_sink_full_uniform` — average of full checkpoints: anchor, greedy, and each specialist materialized onto the anchor

All merges run on **CPU only** so they do not crowd the live v2.3 GPU job.

## Relation to official Phase 4 soup

Phase 4 (`merge_delta_soup.py`) averages **reconstructed FP32 adapter deltas**:
`W = W_base + Σ c_i Δ_i` with `Σ c_i = 1`.
That is algebraically the same as averaging the materialized full models when they share one base.
`specialists_lora_uniform` targets that same endpoint via the official v2.2 LoRA composer (`weight=1/N` each).
`anchor_greedy_uniform` / kitchen-sink are Wortsman full-checkpoint averages; greedy is not an independent same-base adapter ingredient in the Phase 4 sense.
See sibling project `../v22-specialist-mean-dose-sweep` for an explicit Phase-4 merger + dose sweep.

## Quick start

From this folder, with the same `.venv` as the official package:

```powershell
..\..\.venv\Scripts\Activate.ps1
$env:CUDA_VISIBLE_DEVICES=""
python .\scripts\build_v22_soups.py
python .\scripts\smoke_eval_cpu.py --limit 48
```

Artifacts land under `artifacts/`. Official Phase datasets are referenced read-only from `../short-response-model-phase-0-1-2`.

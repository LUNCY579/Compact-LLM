# v2.2 Specialist-Mean Dose Sweep

Parallel micro-experiment next to `v22-model-soup-micro-experiment`.

## Research question

Does a **Phase-4-style mean of v2.2 specialist deltas**, applied at capability doses `0.1 … 1.0`, beat the plain frozen anchor / greedy baselines?

## Phase 4 equivalence (important)

Official Phase 4 soup (`scripts/phase_4_model_soups/merge_delta_soup.py`) does **not** average LoRA A/B factors. It:

1. materializes each adapter on the shared base in FP32
2. reconstructs `Δ_i = W_i − W_base`
3. forms `W = W_base + Σ c_i Δ_i` with `Σ c_i = 1`

When every ingredient is `base + Δ_i`, that is algebraically identical to Wortsman uniform averaging of the full checkpoints:

`Σ c_i W_i = W_base + Σ c_i Δ_i`

This project therefore:

- builds the specialist mean with the **official Phase 4 merger**
- applies dose `d` as `W(d) = (1−d)·W_anchor + d·W_mean`  
  (= `W_anchor + d · mean(Δ)`)

The earlier `v22-model-soup-micro-experiment` specialists soup used the official v2.2 LoRA composer with weights `1/N` each, which targets the same `W_anchor + mean(Δ)` endpoint. Full-checkpoint averaging of already-materialized models is the Wortsman form of the same math. Averaging `frozen_anchor` with `greedy_soup` is a different experiment (diluting the abstention dose), not a Phase-4 multi-adapter delta soup.

## Recorded commands

Run from this folder. Prefer the repo `.venv`. Keep CUDA hidden so these stay CPU-only.

```powershell
cd C:\Users\clclc\BusinessProject\Chat-Bubble\v22-specialist-mean-dose-sweep
$env:CUDA_VISIBLE_DEVICES=""
$py = "C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe"
```

### 1) Build Phase-4 specialist mean + dose checkpoints

Doses: `0.0, 0.1, 0.25, 0.5, 0.75, 1.0`  
Also compares the prior LoRA composer soup against the Phase-4 merger (`artifacts/PHASE4_EQUIVALENCE_CHECK.json`).

```powershell
& $py .\scripts\build_phase4_dose_models.py
# rebuild from scratch:
& $py .\scripts\build_phase4_dose_models.py --force
# skip prior-soup equivalence check:
& $py .\scripts\build_phase4_dose_models.py --skip-verify-prior
```

### 2) Smoke eval (truncated DEV slices)

```powershell
& $py .\scripts\smoke_eval_cpu.py --limit 24
# or one-shot build + smoke:
& $py .\scripts\run_dose_sweep.py --limit 24
```

### 3) Full official DEV eval (4050 + 4860 + 1320 rows)

```powershell
& $py .\scripts\smoke_eval_cpu.py --full --resume
# optional filters:
& $py .\scripts\smoke_eval_cpu.py --full --resume --benchmark multidomain_dev
& $py .\scripts\smoke_eval_cpu.py --full --resume --candidate dose_1p00 --candidate dose_0p00 --candidate greedy_soup
# one-shot build + full eval (resumable):
& $py .\scripts\run_dose_sweep.py --full --resume --skip-build
& $py .\scripts\run_dose_sweep.py --full --resume
```

Outputs:

- `artifacts/DOSE_BUILD_SUMMARY.json`
- `artifacts/smoke_eval/SMOKE_COMPARISON.md`
- `artifacts/full_eval/FULL_COMPARISON.md`

Official Phase datasets are referenced read-only from `../short-response-model-phase-0-1-2`.

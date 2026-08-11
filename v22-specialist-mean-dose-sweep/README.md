# v2.2 Specialist-Mean Dose Sweep

Parallel micro-experiment next to `v22-model-soup-micro-experiment`.

## Research question

Does a **Phase-4-style mean of v2.2 specialist deltas**, applied at capability doses `0.1 … 1.0`, beat the plain frozen anchor / greedy baselines on tiny smoke probes?

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

## Run (CPU only)

```powershell
$env:CUDA_VISIBLE_DEVICES=""
python .\scripts\run_dose_sweep.py --limit 24
```

# Phase 2B verified execution manual

“Verified” means the scripts contain explicit provenance, reload checks, common evaluation, and stop conditions. It does not claim a GPU run has already succeeded.

## Why this is inside Phase 2

Phase 2A found no selective one-, two-, or four-layer adapter that recovered the all-layer SFT result. Phase 2B changes the question from **where to train** to **what can be removed and then recovered**. Running it before DPO gives Phase 3 a chance to align the reduced architecture instead of deleting layers only after all behavior training is complete.

## Preflight

1. Keep `artifacts/dense_sft/adapter` and `artifacts/dense_sft_evaluation` unchanged.
2. Confirm `reports/PHASE_1_STOP_GATE.md` contains an explicit human PASS.
3. Confirm adequate disk space. One Qwen3-0.6B full checkpoint can be roughly gigabyte-scale depending on precision.
4. Activate the CUDA environment used by Phase 1.
5. Close other GPU workloads before comparing latency or memory.

## Recommended first run

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -MaxDrops 4 `
  -RecoveryTopK 3 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

Increase `MaxDrops` only after reviewing the first frontier. Use `-Resume` only when code, evaluator, probe, model, and data have not changed.

## Review order

1. Read `PHASE_2B_PROGRESSIVE_REDUCTION_REPORT.md`.
2. Inspect every accepted stage's `OUTPUT_DEVIATION.md`.
3. Inspect the lowest-score rejected neighbors to confirm the stopping point is meaningful.
4. Check critical and major semantic flags manually.
5. Confirm target NLL did not sharply worsen even when generated score remained 1.0.
6. Compare checkpoint weight bytes and parameters, not only adapter size.
7. Choose approved profiles in `PHASE_2B_STOP_GATE.md`, for example:

```text
- Human decision: `PASS`
- Approved profiles: `full_quality,balanced`
```

Do not approve `min_memory` merely because its reduction/score ratio is positive.

## Continuing Phase 3

Full quality, preserving the existing behavior:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 `
  -Phase2Profile full_quality
```

Balanced reduced branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 `
  -Phase2Profile balanced
```

Minimum-memory branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 `
  -Phase2Profile min_memory
```

Profile-specific Phase 3 outputs are written side by side so the original accepted Phase 3 artifacts are not overwritten.

## Failure and rollback

A failed structural sweep is still useful evidence. Keep the reports, leave reduced profiles unapproved, and continue with `full_quality`. The original base model, SFT adapter, full materialized reference, and every accepted frontier model are never modified in place.

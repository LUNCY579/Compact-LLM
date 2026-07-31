# Phase 2C — Importance-Guided Targeted Recovery

Phase 2C starts from a **human-approved Phase 2B reduced checkpoint** and asks:

> Which surviving layers can receive a small LoRA/QLoRA adapter and recover nearly all of the quality of all-surviving-layer tuning with materially less trainable state?

It preserves the physical weight-memory reduction already achieved by Phase 2B. It does not reinterpret the negative Phase 2A result: Phase 2A trained tiny subsets on the original architecture, while Phase 2C trains selected surviving layers after concise behavior was learned and redundant depth was removed.

## Default command

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -QLoRA `
  -GradientCheckpointing
```

Resume an interrupted run with the **identical settings and unchanged inputs**:

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -QLoRA `
  -GradientCheckpointing `
  -Resume
```

`run_config.json` records SHA-256 fingerprints of the reduced checkpoint, Phase 2B catalog/registry, datasets, benchmark, seeds, thresholds, LoRA settings, and ranking weights. Resume fails rather than mixing evidence when any recorded input or setting changes.

## Data

The Phase 2C mixture retains the approved Phase 1 train/dev records and adds deterministic, mechanically verified concise-reasoning cases for implication, ordering, comparison, arithmetic, contradiction detection, set membership, temporal reasoning, and clarification. Added train/dev entities are disjoint, exact prompt-target leakage is rejected, and sealed data is never read.

The development file deterministically interleaves original and Phase 2C cases. Therefore, the default 64-case importance calibration and 180-case candidate evaluation contain both conversational controls and concise-reasoning examples instead of accidentally evaluating only the first original-data records.

## Importance signals

1. First-order Taylor sensitivity of assistant-target loss to each surviving layer output.
2. Target-token NLL increase when one surviving layer is temporarily bypassed.
3. Final-stage Phase 2B raw-deletion damage when the matching registry is available.

The combined ranking chooses **where limited updates may be useful**. It is not evidence that a layer can be deleted.

## Required controls

- `all_remaining`
- `top1`, `top2`, `top4`, `top8`
- `complementary4`
- `random4`, `low4`
- `top4_attention`, `top4_mlp`

Candidate gates compare against both the independently trained `all_remaining` control and the frozen Phase 2B model. They use heuristic quality, semantic pass rate, lexical reference F1, complete utterances, category-level composite quality, target NLL, critical/major output-deviation flags, trainable parameters, training time, peak training VRAM, and adapter size.

Every recommended profile receives repeated-seed evidence. Selective repeats are compared with a separately trained `all_remaining` control using the same seed. If a failed finalist exposes a new recommendation, that new finalist is repeated before publication.

## Outputs

- `artifacts/phase_2c_targeted_recovery/run_config.json`
- `artifacts/phase_2c_targeted_recovery/importance/combined_layer_ranking.json`
- `artifacts/phase_2c_targeted_recovery/candidate_registry.json`
- `artifacts/phase_2c_targeted_recovery/model_catalog.json`
- `reports/PHASE_2C_TARGETED_RECOVERY_REPORT.md`
- `reports/PHASE_2C_STOP_GATE.md`

No profile enters Phase 3 until the stop gate says `Human decision: PASS` and the exact profile is listed in `Approved profiles`.

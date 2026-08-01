# Phase 2 — Layer Selection and Progressive Structural Reduction

Phase 2 now contains three related but independent studies:

- **Phase 2A — selective-layer tuning:** asks where concise behavior can be learned with fewer trainable layers. The current project result remains a valid negative finding: no one-, two-, or four-layer adapter matched the all-layer reference within the configured gate.
- **Phase 2B — progressive structural reduction:** asks which complete inference layers can be physically removed, optionally recovery-tuned, and removed in combination while preserving a useful quality/weight-memory tradeoff.
- **Phase 2C — targeted recovery:** starts from an approved reduced model, ranks surviving layers, and tests the smallest selected-layer adapters that match all-surviving-layer recovery within strict quality floors.

Training importance and inference-removal tolerance are not the same measurement. A failed Phase 2A gate does not imply that every layer is equally irreplaceable, and a successful Phase 2B deletion does not retroactively make selective tuning successful.

## Commands

Run Phase 2A and Phase 2B in order (Phase 2C remains blocked until Phase 2B human approval):

```powershell
.\scripts\phase_2_layer_selection\run_phase_2.ps1 -Mode All -Resume
```

Run only the new structural study after the existing Phase 2A result:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2.ps1 `
  -Mode StructuralOnly `
  -MaxDrops 6 `
  -RecoveryTopK 3 `
  -RecoverySteps 50 `
  -Resume
```


Run Phase 2C after approving a reduced Phase 2B profile:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2.ps1 `
  -Mode TargetedRecoveryOnly `
  -Phase2CSourceProfile balanced `
  -Resume
```

The dedicated runner exposes the complete settings:

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -QLoRA `
  -GradientCheckpointing
```

Direct Phase 2B command using the synchronized Quality Evaluation v2.0 defaults:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -MaxDrops 6 `
  -RecoveryTopK 3 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

Thorough 150-case run with isolated artifacts and reports:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2_progressive_reduction_v1_2_thorough `
  -Report .\reports\PHASE_2B_V1_2_THOROUGH_REPORT.md `
  -GateReport .\reports\PHASE_2B_V1_2_THOROUGH_STOP_GATE.md `
  -MaxDrops 10 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

The dedicated wrapper now exposes and forwards every Phase 2B quality-v2 threshold, `ProbeLimit`, probe/recovery seeds, phase benchmark, response length, and report paths. Its defaults match `progressive_reduction.py` and `configs/phase_2_progressive_reduction.yaml`: mean quality loss `0.05`, balanced-profile loss `0.02`, semantic loss `0.02`, category loss `0.08`, target-NLL increase `0.15`, completion `0.99`, quality-pass `0.95`, intelligibility `1.00`, alignment loss `0.08`, and zero new major issues. A custom `OutputRoot` automatically receives unique report names unless `Report` and `GateReport` are supplied explicitly.

Set `-RecoveryTopK 28` for a 28-layer model to recovery-train every raw deletion branch. The default recovers only the strongest three per stage because training every branch at every progressive step is extremely expensive.

## Phase 2B procedure

1. Materialize the accepted Phase 1 all-layer SFT adapter as an immutable full checkpoint.
2. Build a deterministic, category-balanced 50-case probe from development data; sealed data is never used.
3. Delete each current layer separately and physically save a temporary candidate.
4. Generate and retain all 50 replies, common evaluator scores, category metrics, semantic flags, teacher-forced target NLL, output-deviation reports, latency, and model footprint.
5. Recovery-train the strongest raw candidates with short all-remaining-layer LoRA/QLoRA SFT.
6. Select the least damaging valid deletion and materialize it as the next preserved frontier checkpoint.
7. Repeat the every-current-layer deletion study from that checkpoint.
8. Stop when no next deletion keeps weight-memory reduction greater than relative heuristic-score loss while passing hard floors.
9. Publish three retained profiles: `full_quality`, `balanced`, and `min_memory`.

The 50-case probe detects trends and exposes score saturation; it is not a release benchmark. Retained profiles are separately re-evaluated on ShortChatBench Pilot for Phase 3 handoff.

## Retention behavior

Every accepted frontier checkpoint is preserved. Every rejected candidate's generated outputs, metrics, manifests, and recovery adapter are preserved. Large rejected full checkpoints are deleted by default because they can consume hundreds of gigabytes across a progressive sweep and can be regenerated from the preserved parent plus the recorded layer ID. Use `-KeepCandidateCheckpoints` only when sufficient disk space is available.

## Main files

- `run_static_layer_study.py` — Phase 2A coarse-to-fine selected-layer tuning study.
- `build_phase_2_human_review.py` — repairs the existing runner reference and builds the Phase 2A candidate/control review packet from raw outputs.
- `progressive_reduction.py` — Phase 2B every-layer scan, recovery, progressive deletion, frontier selection, catalog, and reports.
- `create_depth_candidate.py` — physically removes one current decoder block while preserving original layer lineage.
- `build_structural_probe.py` — deterministic development-only 50-case probe.
- `evaluate_teacher_forcing.py` — continuous target-NLL contrast when generated heuristic scores saturate.
- `compare_outputs.py` — side-by-side reply changes, similarity, score deltas, and new semantic flags.
- `resolve_phase_2_profile.py` — safely hands an approved profile to Phase 3.
- `run_phase_2_structural_reduction.ps1` — Windows orchestration for Phase 2B.
- `STRUCTURAL_REDUCTION_MANUAL.md` — execution and review details.
- `FILE_RESPONSIBILITIES.md` — responsibility of every new file.

## Phase 2C handoff

Phase 2C writes its own catalog and stop gate. It preserves the Phase 2B structural memory reduction while comparing all-surviving-layer recovery with top-k, random, low-ranked, complementary, attention-only, and MLP-only adapters. See `scripts/phase_2c_targeted_recovery/README.md`.

## Gates

- `reports/PHASE_2_STOP_GATE.md` remains the Phase 2A selective-tuning result.
- `reports/PHASE_2B_STOP_GATE.md` governs reduced profile approval.
- `reports/PHASE_2C_STOP_GATE.md` governs targeted-recovery adapter approval.
- `full_quality` remains available even when Phase 2B is rejected.
- `balanced` or `min_memory` may enter Phase 3 only after the Phase 2B report contains `Human decision: PASS` and explicitly lists the approved profile names.

## Quality evaluator v2.0

This phase consumes or produces evaluator schema `2.0`. The main score is not a length/completion average: hard intelligibility collapse receives zero, and phase gates separately enforce tail quality, category behavior, semantic and completion floors, reference alignment, and critical/major issue limits. Evidence created by the legacy evaluator must be regenerated before this phase can automatically pass. See the repository-root `QUALITY_EVALUATION_V2_AUDIT.md` and `QUALITY_EVALUATION_V2_MIGRATION.md`.


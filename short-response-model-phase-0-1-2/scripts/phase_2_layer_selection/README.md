# Phase 2 — Layer Selection and Progressive Structural Reduction

Phase 2 now contains three related but independent studies:

- **Phase 2A — selective-layer tuning:** asks where concise behavior can be learned with fewer trainable layers. The current project result remains a valid negative finding: no one-, two-, or four-layer adapter matched the all-layer reference within the configured gate.
- **Phase 2B — progressive structural reduction:** asks which complete inference layers can be physically removed, optionally recovery-tuned, and removed in combination while preserving a useful quality/weight-memory tradeoff.
- **Phase 2C — targeted recovery:** starts from an approved reduced model, ranks surviving layers, and tests selected-layer adapters using proportional training-resource benefit versus proportional quality opportunity loss.

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

Direct Phase 2B command using the shared Evaluator v4.5 relative-opportunity policy:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -MaxDrops 6 `
  -RecoveryTopK 3 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

The normal runner writes durable example evidence and live reports throughout the command. To refresh the human-readable view in another PowerShell window without relying on a file handle that may be replaced atomically:

```powershell
while ($true) {
  Clear-Host
  Get-Content .\artifacts\phase_2_progressive_reduction\LIVE_PROGRESS.md -Raw
  Start-Sleep -Seconds 5
}
```

Resume an interrupted run with the same experiment settings and output root:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -OutputRoot .\artifacts\phase_2_progressive_reduction `
  -MaxDrops 6 `
  -RecoveryTopK 3 `
  -RecoverySteps 50 `
  -Resume
```

Rebuild candidate, stage, and partial reports without model inference:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -OutputRoot .\artifacts\phase_2_progressive_reduction `
  -RebuildReport `
  -AllowPartialReport
```

The direct Python equivalent is:

```powershell
python .\scripts\phase_2_layer_selection\build_phase_2b_report.py `
  --run-root .\artifacts\phase_2_progressive_reduction `
  --allow-partial `
  --rebuild
```

Thorough 150-case run with isolated artifacts and reports:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2_progressive_reduction_v1_3_relative `
  -Report .\reports\PHASE_2B_V1_2_THOROUGH_REPORT.md `
  -GateReport .\reports\PHASE_2B_V1_2_THOROUGH_STOP_GATE.md `
  -MaxDrops 10 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```

Evaluator v4.5 requires every Phase 2B probe row to contain an explicit `response_contract`; legacy probes are rejected rather than treated as silently open-ended. Use `probe_50_v4_5.jsonl` or `probe_150_thorough_v4_5.jsonl`.

The dedicated wrapper exposes search, reproducibility, output, and quality-profile settings. It does not expose absolute promotion thresholds. A custom `OutputRoot` automatically receives unique report names unless `Report` and `GateReport` are supplied explicitly.

Set `-RecoveryTopK 28` for a 28-layer model to recovery-train every raw deletion branch. The default recovers only the strongest three per stage because training every branch at every progressive step is extremely expensive.

## Phase 2B procedure

1. Materialize the accepted Phase 1 all-layer SFT adapter as an immutable full checkpoint.
2. Build a deterministic, category-balanced 50-case probe from development data; sealed data is never used.
3. Delete each current layer separately and physically save a temporary candidate.
4. Generate and retain all 50 replies, common evaluator scores, category metrics, concrete defect labels, contract results, teacher-forced target NLL, output-deviation reports, latency, and model footprint.
5. Recovery-train the strongest raw candidates with short all-remaining-layer LoRA/QLoRA SFT.
6. Select the least damaging valid deletion and materialize it as the next preserved frontier checkpoint.
7. Repeat the every-current-layer deletion study from that checkpoint.
8. Stop when no next deletion has loaded weight-memory gain strictly greater than combined proportional quality opportunity loss.
9. Publish three retained profiles: `full_quality`, `balanced`, and `min_memory`.

The 50-case probe detects trends and exposes score saturation; it is not a release benchmark. Retained profiles are separately re-evaluated on ShortChatBench Pilot for Phase 3 handoff.

## Retention behavior

Every accepted frontier checkpoint is preserved. Every rejected candidate's generated outputs, metrics, manifests, and recovery adapter are preserved. Large rejected full checkpoints are deleted by default because they can consume hundreds of gigabytes across a progressive sweep and can be regenerated from the preserved parent plus the recorded layer ID. Use `-KeepCandidateCheckpoints` only when sufficient disk space is available.

## Incremental artifact contract

```text
<OutputRoot>/
├── LIVE_PROGRESS.json
├── LIVE_PROGRESS.md
├── RUN_STATE.json
├── PARTIAL_RUN_REPORT.md
├── model_catalog.json
├── candidate_registry.json
└── evidence/
    └── stage_06/
        ├── stage_status.json
        ├── STAGE_06_REPORT.md
        └── stage_06_drop_original_17_raw/
            ├── candidate_status.json
            ├── candidate_row.json
            ├── CANDIDATE_REPORT.md
            ├── combined_metrics.json
            ├── progress_context.json
            └── generation/
                ├── evaluation_manifest.json
                ├── evaluation_progress.json
                ├── results.jsonl
                └── summary.json
```

`results.jsonl` is append-only while evaluation is running. Each completed example is flushed and fsynced immediately. Resume loads valid rows, repairs only a malformed final line, deduplicates by stable benchmark `id`, and skips IDs already present. Once all examples are complete, the file is atomically rewritten into deterministic benchmark order.

`LIVE_PROGRESS.json` is updated after each example. `LIVE_PROGRESS.md` defaults to every five examples and can be changed with `-ProgressUpdateInterval`; use `-DisableLiveMarkdown` when only machine-readable progress is needed. Partial metrics are explicitly provisional and never authorize promotion.

Candidate states are `pending`, `preparing`, `evaluating`, `evaluation_complete`, `recovering`, `recovery_complete`, `gating`, `accepted`, `rejected`, `interrupted`, and `failed`. A gate-accepted candidate is still not the next parent until its status records `promoted=true` and a preserved frontier checkpoint path.

The offline builder supports older folders containing only `results.jsonl`, `summary.json`, stage evidence, and old reports. It marks inferred legacy fields, reconstructs only what the artifacts prove, and keeps promotion disabled when no saved gate decision exists.

## Main files

- `run_static_layer_study.py` — Phase 2A coarse-to-fine selected-layer tuning study.
- `build_phase_2_human_review.py` — repairs the existing runner reference and builds the Phase 2A candidate/control review packet from raw outputs.
- `progressive_reduction.py` — Phase 2B every-layer scan, recovery, progressive deletion, frontier selection, catalog, and reports.
- `incremental_reporting.py` — model-free live status, candidate/stage/partial reports, legacy inference, and offline artifact reconstruction.
- `build_phase_2b_report.py` — runnable no-inference report rebuild command.
- `create_depth_candidate.py` — physically removes one current decoder block while preserving original layer lineage.
- `build_structural_probe.py` — deterministic development-only 50-case probe.
- `evaluate_teacher_forcing.py` — continuous target-NLL contrast when generated heuristic scores saturate.
- `compare_outputs.py` — side-by-side reply changes, similarity, score deltas, and new concrete defect/contract labels.
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

## Quality evaluator v4.5

This phase uses Quality Evaluator `4.5` and the shared relative-opportunity engine. Informal or mildly ungrammatical language is permitted when understandable; references are examples rather than unique answers; and behavioral restrictions are enforced only when explicitly present in the fresh-message response contract. Promotion never depends on an absolute score, pass-rate, completion, category, NLL, or issue-count floor. The declared proportional resource benefit must be strictly greater than combined proportional quality opportunity loss. See `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` and `EVALUATION_V4_5_MIGRATION.md`.


## Phase 2B continuation hotfix — direct Stage 10+ resume

A completed `model_catalog.json` no longer causes `-Resume` to exit. The runner now:

1. validates the persisted model, adapter, probe, benchmark, evaluator, quality profile, and every available run setting;
2. loads the existing `candidate_registry.json` and accepted frontier;
3. restores the last accepted frontier checkpoint as the parent model;
4. starts from the next stage rather than rebuilding Stages 1 through 10;
5. audits legacy accepted frontier stages under the new parent-relative rule;
6. requires both cumulative and incremental opportunity margins to be positive;
7. runs focused recovery first and recovery-tests every remaining branch only when the stage would otherwise stop;
8. records an explicit terminal reason in the catalog, report, and stop gate;
9. keys downstream profile-validation folders by model path so a later frontier cannot reuse an older profile summary.

The pre-hotfix catalog did not persist every recovery and fingerprint field. During its first continuation, the runner verifies all fields the catalog contains and writes the unavailable legacy fields to `resume_validation.json` as explicit assumptions. After that migration, future resumes require exact matching settings and fingerprints except that `MaxDrops` may be increased.

Continue the established August 1 thorough run with:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl `
  -ProbeLimit 150 `
  -OutputRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Report .\reports\PHASE_2B_phase_2b_quality_v3_thorough_20260801_REPORT.md `
  -GateReport .\reports\PHASE_2B_phase_2b_quality_v3_thorough_20260801_STOP_GATE.md `
  -MaxDrops 27 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing `
  -Resume
```

`MaxDrops 27` is only an upper bound. The two-layer safety guard means a 28-layer source can accept at most 26 deletions, and the run normally stops earlier when no remaining deletion is worthwhile under both opportunity comparisons.

## Evaluator v4.5 rescoring and corrected-frontier replay

Pause an active v3.0 run before migration. To re-score all preserved Phase 2B replies without model inference and prepare the corrected frontier replay:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4_5.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl
```

Then rerun the exact original structural-reduction command with `-Resume`. Stored generation, NLL, and footprint evidence is reused while the corrected frontier follows the same parent lineage. If v4.5 changes a stage winner, only the downstream branch created from the superseded parent is regenerated. Timestamped pre-v4.5 backups and `PHASE2B_QUALITY_V4_5_MIGRATION.json` preserve the migration audit trail.

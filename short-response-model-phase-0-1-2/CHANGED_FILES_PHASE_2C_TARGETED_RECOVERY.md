# Changed Files — Phase 2C Targeted Recovery

## New files

| Path | Purpose |
|---|---|
| `scripts/phase_2c_targeted_recovery/build_phase_2c_dataset.py` | Builds and verifies the deterministic concise-reasoning mixture. |
| `scripts/phase_2c_targeted_recovery/rank_surviving_layers.py` | Ranks surviving reduced-model layers using Taylor, bypass-NLL, and optional Phase 2B evidence. |
| `scripts/phase_2c_targeted_recovery/phase2c_common.py` | Pure candidate, category-quality, gate, efficiency, profile, and approval helpers. |
| `scripts/phase_2c_targeted_recovery/targeted_recovery.py` | Full Phase 2C orchestration, fingerprints, controls, repeated seeds, reports, gate, and catalog. |
| `scripts/phase_2c_targeted_recovery/run_phase_2c.ps1` | Runnable Windows entry point with configurable training/ranking/gate settings. |
| `scripts/phase_2c_targeted_recovery/check_phase_2c_gate.py` | Verifies automated eligibility and exact repeated-seed evidence. |
| `scripts/phase_2c_targeted_recovery/resolve_phase_2c_profile.py` | Resolves only human-approved, eligible, seed-stable profiles for Phase 3. |
| `scripts/phase_2c_targeted_recovery/audit_phase_2c_artifacts.py` | Read-only completed-artifact consistency auditor. |
| `scripts/phase_2c_targeted_recovery/README.md` | Experiment overview and evidence boundaries. |
| `scripts/phase_2c_targeted_recovery/EXECUTION_MANUAL.md` | Run, isolated-run, resume, restart, review, approval, audit, and handoff guide. |
| `scripts/phase_2c_targeted_recovery/FILE_RESPONSIBILITIES.md` | Responsibility and non-responsibility of every Phase 2C file. |
| `data/phase_2c_targeted_recovery/train.jsonl` | Phase 1 training data plus deterministic Phase 2C reasoning records. |
| `data/phase_2c_targeted_recovery/dev.jsonl` | Interleaved original and Phase 2C development records. |
| `data/phase_2c_targeted_recovery/dataset_manifest.json` | Counts, provenance, ordering, leakage result, and SHA-256 hashes. |
| `data/phase_2c_targeted_recovery/README.md` | Dataset provenance and regeneration instructions. |
| `configs/phase_2c_targeted_recovery.yaml` | Human-readable default research configuration and gates. |
| `reports/PHASE_2C_STOP_GATE.md` | Initial NOT_RUN human-gate placeholder replaced by the runner. |
| `tests/test_phase_2c_targeted_recovery.py` | Phase 2C regression tests. |
| `PHASE_2C_IMPLEMENTATION_CHANGELOG.md` | Implementation summary and evidence boundary. |
| `PHASE_2C_MANUAL_UPDATE_CHECKLIST.md` | Hotfix installation and manual workflow checklist. |
| `PHASE_2C_SOURCE_VERIFICATION.md` | Source-level verification results and unexecuted GPU boundary. |
| `CHANGED_FILES_PHASE_2C_TARGETED_RECOVERY.md` | This file list. |

## Modified files

| Path | Change |
|---|---|
| `scripts/phase_1_dense_sft/train_dense_sft.py` | Records trainable/total parameters, adapter bytes, LoRA/training settings, and run labels in training manifests. |
| `scripts/phase_1_dense_sft/evaluate_model.py` | Adds category semantic-pass, reference-F1, completion, critical-case, and major-case metrics while preserving prior fields. |
| `scripts/phase_2_layer_selection/run_phase_2.ps1` | Adds `TargetedRecoveryOnly` orchestration and Phase 2C parameters. |
| `scripts/phase_2_layer_selection/README.md` | Documents Phase 2A/2B/2C boundaries and commands. |
| `scripts/phase_2_layer_selection/FILE_RESPONSIBILITIES.md` | Adds Phase 2C handoff responsibilities. |
| `scripts/phase_3_preference_alignment/run_phase_3.ps1` | Accepts mutually exclusive Phase 2B or Phase 2C profiles and isolates branch artifacts. |
| `scripts/phase_3_preference_alignment/train_dpo.py` | Continues a selected Phase 2C SFT adapter on the exact reduced checkpoint. |
| `scripts/phase_3_preference_alignment/README.md` | Documents Phase 2C profile branches and compatibility. |
| `scripts/phase_3_preference_alignment/EXECUTION_MANUAL.md` | Adds Phase 2C handoff/run instructions. |
| `scripts/phase_4_model_soups/README.md` | Documents Phase 2C soup branch names and compatibility rules. |
| `scripts/phase_4_model_soups/EXECUTION_MANUAL.md` | Adds exact Phase 2C branch workflow. |
| `scripts/phase_4b_structural_reduction/README.md` | Clarifies named Phase 2C post-alignment revalidation. |
| `README.md` | Adds Phase 2C to the mature pipeline overview. |
| `Phases.md` | Adds Phase 2C responsibility and handoff boundaries. |
| `verify_post_phase_3.ps1` | Includes Phase 2C in compile/test/runner-reference verification. |

# Changed Files — v1.4.7 Phase 2C Reliability and Correctness

## Core reliability

| Path | Change |
|---|---|
| `scripts/common/experiment_utils.py` | Bounded Windows atomic-replace retry, orphan-temp recovery, best-effort progress writes, and verified content-only dataset migration helper. |
| `scripts/phase_1_dense_sft/evaluate_model.py` | Resumable durable generation, normalized and script-fingerprinted manifests, manifestless-evidence rejection, explicit mismatch diagnostics, contract-only rescoring migration, progress isolation, and `None`-safe reporting. |
| `scripts/phase_2_layer_selection/evaluate_teacher_forcing.py` | Durable stable-ID NLL rows, script-fingerprinted manifests, partial-tail recovery, resume, strict manifestless-evidence handling, and authorized legacy contract-only input migration. |
| `scripts/phase_2_layer_selection/compare_outputs.py` | Exact stable-ID coverage, duplicate rejection, technical-validity counts, and result-file fingerprints. |
| `scripts/phase_1_dense_sft/train_dense_sft.py` | Argument validation, free-space guard, selected-layer validation, default checkpoint suppression, staged adapter publication, and final-adapter verification. |
| `scripts/phase_2_layer_selection/progressive_reduction.py` | Forwards resume into teacher-forcing evidence generation. |

## Phase 2C correctness and orchestration

| Path | Change |
|---|---|
| `scripts/phase_2c_targeted_recovery/build_phase_2c_dataset.py` | Explicit correctness contracts for all deterministic objective reasoning rows and versioned v1.4.6 migration metadata. |
| `data/phase_2c_targeted_recovery/train.jsonl` | Same IDs/messages/assistant targets with explicit evaluation contracts added. |
| `data/phase_2c_targeted_recovery/dev.jsonl` | Same IDs/messages/assistant targets with explicit evaluation contracts added. |
| `data/phase_2c_targeted_recovery/dataset_manifest.json` | Manifest v2 hashes, semantic digests, and approved predecessor hashes. |
| `scripts/phase_2c_targeted_recovery/targeted_recovery.py` | Exact run-config diffs, approved contract migration, child resume propagation, sentinel quarantine, adapter validation, disk/checkpoint controls, stale-comparison rebuilding, and atomic final artifacts. |
| `scripts/phase_2c_targeted_recovery/run_phase_2c.ps1` | Dataset-directory validation, custom dataset-directory forwarding, resume/checkpoint/disk options, and Evaluator v4.5 documentation. |
| `scripts/phase_2c_targeted_recovery/rank_surviving_layers.py` | Ranking argument validation and current-policy terminology. |
| `scripts/phase_2c_targeted_recovery/audit_phase_2c_artifacts.py` | Comprehensive candidate, evidence, adapter, seed, profile, catalog, and registry audit. |
| `scripts/phase_2c_targeted_recovery/check_phase_2c_gate.py` | Stronger automated eligibility, repeated-seed, profile-validation, and adapter checks. |

## Documentation and tests

| Path | Change |
|---|---|
| `README.md` | v1.4.7 migration and reliability guidance. |
| `scripts/phase_2c_targeted_recovery/README.md` | Correctness defect, migration, resume, and checkpoint policy. |
| `scripts/phase_2c_targeted_recovery/EXECUTION_MANUAL.md` | Exact run/resume/migration/audit instructions. |
| `PHASE_2C_RELIABILITY_AND_CORRECTNESS_HOTFIX_v1.4.7.md` | Release rationale and installation guide. |
| `RELEASE_VERIFICATION_v1.4.7.md` | Verification record and residual evidence boundary. |
| `tests/test_phase_2c_targeted_recovery.py` | Objective-contract correctness regressions. |
| `tests/test_reliability_v147.py` | Atomic retry, manifests, resume, sentinels, migration, rescoring, output coverage, and wrapper regressions. |
| `HOTFIX_MANIFEST_v1.4.7.json` | File hashes and verification metadata for the hotfix package. |

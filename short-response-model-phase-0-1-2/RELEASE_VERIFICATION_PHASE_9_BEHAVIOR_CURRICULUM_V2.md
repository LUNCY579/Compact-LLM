# Release verification — Phase 9 Behavior Curriculum v2

Version 1.6.1 | August 2026

## Verification scope

This report verifies repository structure, deterministic data preparation, split/calibration behavior, evaluation arithmetic, and regression compatibility. It does not claim that CUDA training completed or that any student achieved a quality target.

## Automated verification

| Check | Result |
|---|---:|
| Python source compilation | PASS |
| Complete pytest suite | PASS — 169 tests |
| Evaluator v4.5 calibration | PASS — 639/639 |
| Real-collapse regression coverage | PASS — 150/150 |
| Relative-opportunity policy audit | PASS |
| Phase 9 config JSON validity | PASS |
| Phase 9 PowerShell structural balance | PASS |
| Phase 9 runner Python references | PASS — 14/14 exist |
| Foundation materialization | PASS — 336 rows |
| Foundation curriculum expansion smoke | PASS — 1,456 rows |
| Foundation decision coverage | PASS — answer/social/abstain/conflict |
| Research-ready canary split | PASS — 33/6/5 |
| Canary connected-group leakage check | PASS |
| Canary evidence-gate calibration | PASS — 4 answerable, 2 unanswerable |
| Near-zero comparison stability tests | PASS |
| Social/unsupported classification tests | PASS |

## Environment boundary

Available locally:

- Python source execution;
- CPU PyTorch 2.10;
- SQLite FTS5;
- pytest and repository evaluators.

Unavailable locally:

- Transformers;
- PEFT;
- TRL;
- CUDA/GPU training;
- PowerShell runtime.

Accordingly, the following remain target-machine validations:

- model and tokenizer loading under the user's pinned environment;
- SFT optimizer updates;
- completion-position online KL;
- 4-bit teacher loading;
- LoRA continuation from SFT into KD;
- real VRAM/latency measurements;
- actual student output quality.

## Recommended clean run

Do not reuse the earlier v1 Phase 9 training/evaluation artifacts for the first Behavior Curriculum v2 measurement. Use a fresh output root or omit `-ReuseExisting`:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -OutputRoot .\artifacts\phase_9_behavior_curriculum_v2_20260806 `
  -UseLora `
  -Teacher4Bit `
  -MaxSteps 600
```

The default research-ready run requires more than the earlier six-row packet. `-SmokeTest` is available to verify execution, but its result is intentionally not eligible for production recommendation.

## Release conclusion

The source package is suitable for a fresh Phase 9 experiment. It is not evidence that a student is accepted. Human review remains required after inspecting raw closed-book, oracle, real, filtered-adversarial, and passthrough-adversarial outputs.

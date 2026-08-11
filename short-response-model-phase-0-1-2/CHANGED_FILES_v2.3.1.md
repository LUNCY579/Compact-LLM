# v2.3.1 Changed Files

v2.3.1 is built on `shortresponsev2.3revised.zip`. It preserves the source repository and expands the already-present v2.3.1 diagnostic-completion helpers into a full Grokking Dynamics Observatory.

## Added

- `configs/phase_9_grokking_dynamics_v231.json`
- `scripts/phase_9_student_distillation/grokking_utils_v231.py`
- `scripts/phase_9_student_distillation/build_grokking_variants_v231.py`
- `scripts/phase_9_student_distillation/evaluate_grokking_checkpoint_v231.py`
- `scripts/phase_9_student_distillation/evaluate_grokking_time_curve_v231.py`
- `scripts/phase_9_student_distillation/evaluate_grokking_dose_surface_v231.py`
- `scripts/phase_9_student_distillation/evaluate_grokking_procedure_curve_v231.py`
- `scripts/phase_9_student_distillation/analyze_lora_dynamics_v231.py`
- `scripts/phase_9_student_distillation/analyze_representation_dynamics_v231.py`
- `scripts/phase_9_student_distillation/run_grokking_factor_sweep_v231.py`
- `scripts/phase_9_student_distillation/summarize_grokking_factors_v231.py`
- `scripts/phase_9_student_distillation/detect_grokking_transitions_v231.py`
- `scripts/phase_9_student_distillation/compile_grokking_report_v231.py`
- `scripts/phase_9_student_distillation/validate_grokking_dynamics_v231.py`
- `scripts/phase_9_student_distillation/run_phase_9_grokking_dynamics_v231.ps1`
- `tests/test_phase_9_grokking_dynamics_v231.py`
- `GROKKING_DYNAMICS_V2_3_1.md`
- `RELEASE_VERIFICATION_v2.3.1.md`
- `CHANGED_FILES_v2.3.1.md`
- `PACKAGE_MANIFEST_v2.3.1.json`

## Modified

- `scripts/phase_9_student_distillation/train_student.py`
  - opt-in optimizer/gradient dynamics telemetry;
  - opt-in lightweight PEFT snapshots at Trainer save events;
  - explicit control over model-only versus resumable optimizer checkpoints;
  - normal Phase 9 behavior remains unchanged when the new flags are omitted.
- `README.md`
- `Phases.md`
- `scripts/phase_9_student_distillation/README.md`
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`
- `scripts/phase_9_student_distillation/FILE_RESPONSIBILITIES.md`

## Intentionally unchanged

The v2.3 direct/procedural corpora, consumed-sealed registry, v2.2 model-composition implementation, historical Phase 9 evaluation policy, and Phase 10/11 handoffs are not silently rewritten by the dynamics release.

# Changed Files — v1.7.7 Phase 9 SFT Selection Gate Hotfix

## Updated

- `scripts/phase_9_student_distillation/select_grounded_sft_candidate.py`
  - separates Stage-B integrity warnings from blocking initialization coverage defects;
  - Stage-B prompt/decode integrity no longer poisons trained candidates;
  - preserves strict candidate-owned integrity/coverage/non-degeneracy/behavior-success checks;
  - records `selection_policy_revision=phase9_sft_selection_gate_v1_7_7`;
  - makes the warning/blocking distinction explicit in JSON and Markdown reports.
- `scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1`
  - under `-ReuseExisting`, recomputes Stage-D and final DEV selections from existing summaries;
  - avoids requiring users to delete stale `selection.json` manually;
  - continues reusing expensive training and evaluation artifacts.
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`
- `scripts/phase_9_student_distillation/README.md`
- `README.md`

## Added

- `scripts/phase_9_student_distillation/reselect_existing_grounded_sft.py` — training-free refresh utility for an existing run root.
- `tests/test_phase_9_sft_selection_gate_v177.py`
- `PHASE_9_SFT_SELECTION_GATE_HOTFIX_v1.7.7.md`
- `RELEASE_VERIFICATION_v1.7.7.md`
- `PACKAGE_MANIFEST_v1.7.7.json`

## Not changed

No dataset, LoRA training hyperparameter, Stage-D sweep definition, evaluator metric, sealed-test split, candidate-quality criterion, Phase 9B logic, Phase 10/11 logic, or playground dataset was changed.

# v2.2 Exact Change Record

Source repository: `Output(20260810-063134).zip`

Source SHA-256: `3c5623ec807d9c48aa9728a8691248afca46450ae09eccab200cabbdee7661f8`

Design source: `Chat-Bubble LLM - Procedural Skill Recovery.pdf`

Design-source SHA-256: `b92838141e10efe471a7e2f0df008591c273bb7499de0d379d1e3bae0fdfe280`

## Existing source files intentionally modified

1. `Phases.md`
2. `README.md`
3. `STUDENT_DISTILLATION_PIPELINE.md`
4. `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`
5. `scripts/phase_9_student_distillation/FILE_RESPONSIBILITIES.md`
6. `scripts/phase_9_student_distillation/README.md`
7. `scripts/phase_9_student_distillation/select_general_reasoning_v212.py`
8. `scripts/phase_9_student_distillation/train_student.py`
9. `tests/test_phase_9_general_reasoning_v212.py`

The `select_general_reasoning_v212.py` change is a deliberate backport of the v2.2 evidence rule: missing protected preservation metrics are technical invalidity, never zero preservation loss.

## New v2.2 implementation files

1. `CAPABILITY_COMPOSITION_V2_2.md`
2. `CHANGED_FILES_v2.2.md`
3. `RELEASE_VERIFICATION_v2.2.md`
4. `configs/phase_9_capability_composition_v22.json`
5. `research_sources/v2.2/Chat-Bubble LLM - Procedural Skill Recovery.pdf`
6. `research_sources/v2.2/SOURCE_TRUTH_MANIFEST.json`
7. `scripts/phase_9_student_distillation/build_capability_probe_v22.py`
8. `scripts/phase_9_student_distillation/compose_lora_adapters_v22.py`
9. `scripts/phase_9_student_distillation/derive_layer_plan_v22.py`
10. `scripts/phase_9_student_distillation/evaluate_candidate_v22.py`
11. `scripts/phase_9_student_distillation/evaluate_multidomain_reasoning_v22.py`
12. `scripts/phase_9_student_distillation/generate_multidomain_reasoning_v22.py`
13. `scripts/phase_9_student_distillation/materialize_student_anchor_v22.py`
14. `scripts/phase_9_student_distillation/run_greedy_adapter_soup_v22.py`
15. `scripts/phase_9_student_distillation/run_phase_9_capability_composition_v22.ps1`
16. `scripts/phase_9_student_distillation/select_capability_composition_v22.py`
17. `scripts/phase_9_student_distillation/validate_multidomain_reasoning_v22.py`
18. `tests/test_phase_9_capability_composition_v22.py`

## New v2.2 data assets

1. `data/phase_9_student_distillation/multidomain_reasoning_v22/DATASET_AUDIT.md`
2. `data/phase_9_student_distillation/multidomain_reasoning_v22/README.md`
3. `data/phase_9_student_distillation/multidomain_reasoning_v22/canonical.jsonl`
4. `data/phase_9_student_distillation/multidomain_reasoning_v22/curriculum_sources/social_synthetic_train.jsonl`
5. `data/phase_9_student_distillation/multidomain_reasoning_v22/curriculum_views/specialist_abstention.reviewed.jsonl`
6. `data/phase_9_student_distillation/multidomain_reasoning_v22/curriculum_views/specialist_general.reviewed.jsonl`
7. `data/phase_9_student_distillation/multidomain_reasoning_v22/curriculum_views/specialist_grounding.reviewed.jsonl`
8. `data/phase_9_student_distillation/multidomain_reasoning_v22/curriculum_views/specialist_procedure.reviewed.jsonl`
9. `data/phase_9_student_distillation/multidomain_reasoning_v22/curriculum_views/specialist_social.reviewed.jsonl`
10. `data/phase_9_student_distillation/multidomain_reasoning_v22/curriculum_views/track_a_broad_general.reviewed.jsonl`
11. `data/phase_9_student_distillation/multidomain_reasoning_v22/curriculum_views/track_b_anchored_mixed.reviewed.jsonl`
12. `data/phase_9_student_distillation/multidomain_reasoning_v22/manifests/dataset_manifest.json`
13. `data/phase_9_student_distillation/multidomain_reasoning_v22/manifests/file_hashes.json`
14. `data/phase_9_student_distillation/multidomain_reasoning_v22/reviewed_splits/dev.reviewed.jsonl`
15. `data/phase_9_student_distillation/multidomain_reasoning_v22/reviewed_splits/test.reviewed.jsonl`
16. `data/phase_9_student_distillation/multidomain_reasoning_v22/reviewed_splits/train.reviewed.jsonl`

## Source files intentionally omitted from the cleaned release

Only disposable generated caches from the source ZIP are omitted:

- 14 `__pycache__/*.pyc` files under `scripts/` and `scripts/phase_9_student_distillation/`;
- 1 pytest-generated `.pyc` under `tests/__pycache__/`.

No source code, config, dataset, report, documentation, or research artifact from the input repository is removed.

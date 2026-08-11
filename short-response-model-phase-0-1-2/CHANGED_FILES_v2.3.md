# v2.3 Exact Changed-Files Record

Source: `Output(20260811-003531).zip`

- Source files: **777**
- Final files: **809**
- Common paths: **764**
- Byte-identical common files: **751**
- Modified existing files: **13**
- New files: **45**
- Removed source files: **13**

The removed paths are disposable Python bytecode caches only; no source/config/data/documentation file is intentionally removed.

## Modified existing files

- `Phases.md`
- `README.md`
- `scripts/phase_11_production_rag/benchmark_production_runtime.py`
- `scripts/phase_11_production_rag/calibrate_answerability.py`
- `scripts/phase_11_production_rag/serve_production_rag.py`
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`
- `scripts/phase_9_student_distillation/FILE_RESPONSIBILITIES.md`
- `scripts/phase_9_student_distillation/README.md`
- `scripts/phase_9_student_distillation/audit_rag_retrieval_coverage.py`
- `scripts/phase_9_student_distillation/calibrate_evidence_gate.py`
- `scripts/phase_9_student_distillation/evaluate_rag.py`
- `scripts/phase_9_student_distillation/phase9_common.py`
- `scripts/phase_9_student_distillation/train_student.py`

## New v2.3 files

- `CHANGED_FILES_v2.3.md`
- `PACKAGE_MANIFEST_v2.3.json`
- `RELEASE_VERIFICATION_v2.3.md`
- `RESIDUAL_CAPABILITY_V2_3.md`
- `configs/phase_9_residual_capability_v23.json`
- `data/phase_9_student_distillation/residual_capability_v23/README.md`
- `data/phase_9_student_distillation/residual_capability_v23/SEALED_REGISTRY.json`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/README.md`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/SEALED_DATA_POLICY.md`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/canonical.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/manifests/dataset_manifest.json`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/manifests/file_hashes.json`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/probes/dev_probe.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/probes/ood_probe.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/splits/dev.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/splits/discovery.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/splits/fresh_sealed.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/splits/ood_dev.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/README.md`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/calibration/discovery_calibration.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/canonical.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/knowledge.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/manifests/dataset_manifest.json`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/manifests/file_hashes.json`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/procedures.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/splits/dev.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/splits/discovery.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/splits/fresh_sealed.reviewed.jsonl`
- `data/phase_9_student_distillation/residual_capability_v23/procedural_rag/splits/ood_dev.reviewed.jsonl`
- `scripts/phase_9_student_distillation/analyze_adapter_interference_v23.py`
- `scripts/phase_9_student_distillation/build_preservation_mining_subset_v23.py`
- `scripts/phase_9_student_distillation/build_residual_training_mix_v23.py`
- `scripts/phase_9_student_distillation/compose_residual_adapters_v23.py`
- `scripts/phase_9_student_distillation/evaluate_candidate_v23.py`
- `scripts/phase_9_student_distillation/evaluate_residual_reasoning_v23.py`
- `scripts/phase_9_student_distillation/generate_residual_discovery_v23.py`
- `scripts/phase_9_student_distillation/generate_residual_procedural_rag_v23.py`
- `scripts/phase_9_student_distillation/mine_residual_capabilities_v23.py`
- `scripts/phase_9_student_distillation/oracle_routing_upper_bound_v23.py`
- `scripts/phase_9_student_distillation/run_phase_9_residual_capability_v23.ps1`
- `scripts/phase_9_student_distillation/run_residual_dose_sweep_v23.py`
- `scripts/phase_9_student_distillation/select_residual_capability_v23.py`
- `scripts/phase_9_student_distillation/select_residual_dose_v23.py`
- `scripts/phase_9_student_distillation/validate_residual_capability_v23.py`
- `tests/test_phase_9_residual_capability_v23.py`

## Removed disposable source artifacts

- `scripts/__pycache__/__init__.cpython-311.pyc`
- `scripts/common/__pycache__/__init__.cpython-311.pyc`
- `scripts/common/__pycache__/experiment_utils.cpython-311.pyc`
- `scripts/common/__pycache__/observable_quality.cpython-311.pyc`
- `scripts/common/__pycache__/quality_evaluation.cpython-311.pyc`
- `scripts/common/__pycache__/response_control.cpython-311.pyc`
- `scripts/phase_1_dense_sft/__pycache__/train_dense_sft.cpython-311.pyc`
- `scripts/phase_9_student_distillation/__pycache__/__init__.cpython-311.pyc`
- `scripts/phase_9_student_distillation/__pycache__/phase9_common.cpython-311.pyc`
- `scripts/phase_9_student_distillation/__pycache__/phase9_metrics.cpython-311.pyc`
- `scripts/phase_9_student_distillation/__pycache__/phase9_prompting.cpython-311.pyc`
- `scripts/phase_9_student_distillation/__pycache__/phase9_training_weights.cpython-311.pyc`
- `scripts/phase_9_student_distillation/__pycache__/run_greedy_adapter_soup_v22.cpython-311.pyc`

## Modification summary

- `phase9_common.py`, RAG calibration/evaluation/audit CLIs, and Phase 11 consumers accept `query_aware_v23`; historical defaults remain unchanged.
- `train_student.py` adds row-specific v2.3 anchor-KL multipliers while preserving the existing global anchor objective.
- README/phase/manual/responsibility documentation records the v2.3 branch and one-command workflow.
- All other v2.3 implementation scripts, fresh corpora, configuration, tests, and governing/release documents are additive.

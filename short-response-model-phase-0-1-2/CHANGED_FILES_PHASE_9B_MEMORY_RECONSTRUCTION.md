# Changed files — Phase 9B memory reconstruction v1.7.0

## Added

- `configs/phase_9b_memory_reconstruction.json`
- `data/phase_9b_memory_reconstruction/README.md`
- `data/phase_9b_memory_reconstruction/memories.sample.jsonl`
- `data/phase_9b_memory_reconstruction/reviewed.sample.jsonl`
- `scripts/phase_9b_memory_reconstruction/__init__.py`
- `scripts/phase_9b_memory_reconstruction/phase9b_common.py`
- `scripts/phase_9b_memory_reconstruction/build_reconstruction_candidates.py`
- `scripts/phase_9b_memory_reconstruction/generate_teacher_reconstructions.py`
- `scripts/phase_9b_memory_reconstruction/build_reconstruction_dataset.py`
- `scripts/phase_9b_memory_reconstruction/evaluate_memory_reconstruction.py`
- `scripts/phase_9b_memory_reconstruction/check_phase_9b_gate.py`
- `scripts/phase_9b_memory_reconstruction/finalize_phase_9b_selection.py`
- `scripts/phase_9b_memory_reconstruction/run_phase_9b.ps1`
- `scripts/phase_9b_memory_reconstruction/README.md`
- `scripts/phase_9b_memory_reconstruction/EXECUTION_MANUAL.md`
- `scripts/phase_9b_memory_reconstruction/FILE_RESPONSIBILITIES.md`
- `MEMORY_RECONSTRUCTION_PIPELINE.md`
- `tests/test_phase_9b_memory_reconstruction.py`
- `RELEASE_VERIFICATION_PHASE_9B_MEMORY_RECONSTRUCTION.md`
- `PACKAGE_MANIFEST_PHASE_9B_MEMORY_RECONSTRUCTION.json`

## Updated

- `README.md` and `Phases.md` — place Phase 9B between student distillation and structural reduction.
- `POST_PHASE_3_EXECUTION_GUIDE.md` and `STUDENT_DISTILLATION_PIPELINE.md` — add the reviewed Phase 9B workflow and handoff.
- `PHASE_10_11_WORKFLOW.md` — document the Phase 9B preference and production runtime mode.
- `scripts/common/check_human_gate.py` — recognize the inserted `9B` gate label.
- `scripts/phase_10_student_architecture/run_phase_10.ps1` — prefer an approved Phase 9B reference manifest when present, otherwise use Phase 9.
- `scripts/phase_9_student_distillation/phase9_common.py` — preserve `source_state`, `validity_conditions`, and `memory_type` in existing SQLite metadata.
- `scripts/phase_11_production_rag/serve_production_rag.py` — optional prompted/trained reconstruction pass with auditable decisions and complete cost reporting.
- `configs/phase_11_production_rag.json`, Phase 11 manuals, and production-data README — define the optional reconstruction contract.
- `verify_post_phase_3.ps1` — include the Phase 9B runner and required assets.

## Intentionally unchanged

- Existing Phase 9 artifacts, reports, and selected reference are not rewritten.
- Phase 11 reconstruction defaults to `off`.
- No model checkpoint or human approval is fabricated by this source release.

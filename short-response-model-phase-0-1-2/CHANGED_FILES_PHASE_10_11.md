# Changed files — Phase 10 and Phase 11 full integration

## Added: Phase 10

- `configs/phase_10_student_architecture.json` — pilot/full structural matrices, explicit parameter target bands, recovery policy, compact challenger.
- `scripts/phase_10_student_architecture/__init__.py`
- `scripts/phase_10_student_architecture/README.md`
- `scripts/phase_10_student_architecture/EXECUTION_MANUAL.md`
- `scripts/phase_10_student_architecture/FILE_RESPONSIBILITIES.md`
- `scripts/phase_10_student_architecture/phase10_common.py` — candidate planning and physical tensor slicing.
- `scripts/phase_10_student_architecture/profile_student_structure.py` — all-block residual, MLP, and attention profiling.
- `scripts/phase_10_student_architecture/create_structural_candidate.py` — physical candidate construction, target-band manifest, fresh reload.
- `scripts/phase_10_student_architecture/select_recovery_candidates.py` — bounded recovery search selection.
- `scripts/phase_10_student_architecture/check_phase_10_gate.py` — architecture registry, structural manifest evidence, proportional recommendation, pending gate.
- `scripts/phase_10_student_architecture/finalize_phase_9_selection.py` — Phase 9 model/adapter materialization and immutable handoff.
- `scripts/phase_10_student_architecture/resolve_phase_9_student.py`
- `scripts/phase_10_student_architecture/finalize_phase_10_selection.py` — selected checkpoint materialization, merge, rollback, hash.
- `scripts/phase_10_student_architecture/resolve_phase_10_student.py`
- `scripts/phase_10_student_architecture/run_phase_10.ps1` — complete Pilot/Full/Plan/Evaluate orchestration.
- `data/phase_10_student_architecture/README.md`
- `tests/test_phase_10_student_architecture.py`

## Added: Phase 11

- `configs/phase_11_production_rag.json` — retrieval, answerability, generation, grounding, quantization, and release policy.
- `scripts/phase_11_production_rag/__init__.py`
- `scripts/phase_11_production_rag/README.md`
- `scripts/phase_11_production_rag/EXECUTION_MANUAL.md`
- `scripts/phase_11_production_rag/FILE_RESPONSIBILITIES.md`
- `scripts/phase_11_production_rag/phase11_common.py` — route policy, answerability calibration, grounding checks.
- `scripts/phase_11_production_rag/build_knowledge_snapshot.py` — immutable deduplicated knowledge/procedure snapshot and index.
- `scripts/phase_11_production_rag/calibrate_answerability.py` — question-plus-evidence/procedure gate calibration.
- `scripts/phase_11_production_rag/quantize_student.py` — dense, NF4, INT8, and caller-supplied GGUF artifacts.
- `scripts/phase_11_production_rag/benchmark_production_runtime.py` — complete production-path evaluation, parent/external RSS, VRAM, latency, grounding, and retrieval evidence.
- `scripts/phase_11_production_rag/serve_production_rag.py` — persistent JSONL/interactive local production process.
- `scripts/phase_11_production_rag/check_phase_11_gate.py` — total-process-memory opportunity registry and pending release gate.
- `scripts/phase_11_production_rag/package_phase_11_release.py` — hashable manifest/copy release package with rollback.
- `scripts/phase_11_production_rag/run_phase_11.ps1` — Prepare/Benchmark/Full/Package orchestration.
- `data/phase_11_production_rag/README.md`
- `tests/test_phase_11_production_rag.py`

## Added: repository-wide documentation

- `PHASE_10_11_ARCHITECTURE_REVIEW.md`
- `PHASE_10_11_WORKFLOW.md`
- `CHANGED_FILES_PHASE_10_11.md`
- `RELEASE_VERIFICATION_PHASE_10_11.md`

## Updated

- `README.md` — Phase 10/11 architecture, handoffs, and starting points.
- `Phases.md` — official Phase 10 and Phase 11 responsibilities and artifacts.
- `POST_PHASE_3_EXECUTION_GUIDE.md` — runnable Phase 4–11 sequence.
- `verify_post_phase_3.ps1` — Phase 10/11 runner-reference verification and updated completion label.

## Important preserved boundaries

- Existing Phase 0–9 source behavior was not silently replaced.
- Phase 10 consumes the exact human-approved Phase 9 checkpoint.
- Phase 11 consumes the exact human-approved Phase 10 checkpoint.
- No model result is embedded in this source archive.
- No new private absolute quality floor controls promotion.
- CUDA training, structural recovery quality, quantization quality, and hardware measurements remain target-machine experiments.

## Phase 9 hybrid-suite compatibility synthesis

The Phase 10/11 source was rebased onto the latest Phase 9 hybrid-profile repository rather than overwriting it.

Updated compatibility files:

- `scripts/phase_10_student_architecture/finalize_phase_9_selection.py` — accepts any exact candidate key present in `PHASE_9_ASSESSMENT.json`, including profile-qualified hybrid names; normalizes relative model/adapter paths; records recommendation/control metadata; requires explicit acknowledgement before a research control receives a non-research role.
- `tests/test_phase_10_student_architecture.py` — verifies hybrid candidate handoff and research-control safeguards.
- `PHASE_10_11_WORKFLOW.md` and `PHASE_10_11_ARCHITECTURE_REVIEW.md` — document legacy and hybrid Phase 9 candidate formats.
- `scripts/phase_10_student_architecture/README.md`, `EXECUTION_MANUAL.md`, and `FILE_RESPONSIBILITIES.md` — document the exact Phase 9 assessment-key handoff.
- `README.md`, `Phases.md`, `POST_PHASE_3_EXECUTION_GUIDE.md`, and `verify_post_phase_3.ps1` — synthesize the Phase 9 hybrid suite with official Phase 10/11 orchestration.

Preserved unchanged from the user's latest Phase 9 source:

- hybrid profile definitions in `configs/phase_9_student_distillation.json`;
- hybrid orchestration and candidate naming in `run_phase_9.ps1`;
- multi-candidate assessment logic in `check_phase_9_gate.py`;
- parameter-validation behavior in `create_student.py`;
- Phase 9 hybrid regression tests and existing Phase 9 reports.

# CHANGED FILES — Student Optimization v2.0

## Scope

v2.0 adds a post-distillation optimization program for an already-accepted Phase 9/9B student. It does not renumber or replace official Phases 9, 9B, 10, or 11. Instead it adds a durable meta-orchestrator that reuses Phase 10/11 and fills student-specific efficiency gaps that were not previously covered end to end.

## New files

- `STUDENT_OPTIMIZATION_V2_WORKFLOW.md` — architecture, stage semantics, checkpoint rules, and execution model.
- `configs/student_optimization_v2.json` — cross-stage configuration for sensitivity, context, sparsity, quantization, and release.
- `configs/phase_10_student_architecture_v2.json` — importance-guided Phase 10 search matrix plus matched uniform controls and recovery budget.
- `scripts/student_optimization_v2/state_manager.py` — durable JSON state and incremental live Markdown master report.
- `scripts/student_optimization_v2/scan_student_layer_sensitivity.py` — physically deletes each decoder layer once and ranks DEV sensitivity.
- `scripts/student_optimization_v2/context_budget_sweep.py` — disjoint-DEV retrieval/evidence budget calibration and selection.
- `scripts/student_optimization_v2/sparse_student_pipeline.py` — student contextual-sparsity simulation and measured router-runtime wrapper with dense fallback.
- `scripts/student_optimization_v2/quantization_sweep.py` — dense/NF4/INT8/optional-GGUF DEV runtime comparison with failure-tolerant fallback.
- `scripts/student_optimization_v2/run_student_optimization_v2.ps1` — master cross-phase state machine.
- `scripts/student_optimization_v2/README.md` — operator guide and phase-equivalence map.
- `tests/test_student_optimization_v2.py` — regression coverage for configuration, explicit layer deletion, split isolation, state semantics, orchestration, and human-gate preservation.

## Modified files

- `scripts/phase_10_student_architecture/create_structural_candidate.py` — adds exact `explicit-drop` layer IDs so empirical layer sensitivity can drive physical depth candidates.
- `scripts/phase_10_student_architecture/run_phase_10.ps1` — supports an external structure profile, layer-sensitivity rankings, importance-guided depth removal, and output-root-local registry/report/gate paths while preserving legacy behavior.
- `README.md` — v2.0 orientation and master-runner entry point.
- `Phases.md` — documents v2.0 as a meta-workflow over the unchanged official Phase 9/9B/10/11 boundaries.

## Design decisions

1. **No blind duplication of teacher phases.** Phase 9/9B already owns behavior/evidence/memory specialization; Phase 10 already owns physical architecture reduction and recovery; Phase 11 already owns production RAG/runtime packaging. v2.0 fills the missing student-side layer scan, context/KV selection, sparsity/runtime study, quantization sweep, and orchestration.
2. **No mandatory model soup across physical Phase 10 candidates.** Different depth/width/GQA candidates have incompatible tensor topology, so averaging them is not mathematically valid. Same-topology adapter/soup experiments remain optional research.
3. **Human gates are preserved.** Completed Phase 10/11 registries are resume boundaries. The master runner does not call their `Full` gate builders again after they have completed, preventing a human `PASS` from being overwritten by a regenerated `PENDING`. Intentional fresh evidence cycles require explicit force switches.
4. **Dense fallback is permanent.** Sparse and quantized candidates must demonstrate measured benefit; unsupported or failing backends do not abort the entire workflow.
5. **Completion has a strict meaning.** The master state reaches `COMPLETE` only after Phase 11 human PASS and creation of the approved release ZIP.

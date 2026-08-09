# Student Optimization v2.0 — automated post-distillation optimization

This folder is the high-level orchestration layer that starts **after an exact Phase 9 or Phase 9B student has received human approval and has been materialized**. It does not replace the official lower-level phases. Instead, it closes the efficiency gap between the teacher's Phase 0–8 program and the student path by combining existing Phase 10/11 mechanisms with new student-specific layer sensitivity, context-budget, sparse-runtime, and quantization-selection stages.

Main runner:

```powershell
.\scripts\student_optimization_v2\run_student_optimization_v2.ps1 `
  -TeacherModel <authoritative-teacher-checkpoint>
```

The same command is safe to rerun after a checkpoint. Once a workflow state file exists, the master runner automatically enables reuse/resume behavior for expensive child stages.
Completed Phase 10/11 registries are treated as durable checkpoint evidence, so resume does **not** rerun their `Full` gate builders and cannot silently overwrite a human `PASS` with a new `PENDING`. `-ForcePhase10Rerun` and `-ForcePhase11Rerun` exist only for intentional fresh evidence cycles and therefore require fresh human review.

## Stage map

| v2 stage | Student responsibility | Teacher-pipeline analogue | Implementation |
|---|---|---|---|
| S0 | Freeze exact accepted student lineage | 0/1 handoff discipline | Phase 9/9B manifests + human gate |
| S1 | Empirical one-layer deletion scan | 2A/2B layer study | **new** `scan_student_layer_sensitivity.py` |
| S2 | Physical depth/hidden/MLP/GQA reduction + recovery KD | 2B/2C/4B | Phase 10, upgraded with importance-guided depth candidates |
| S3 | Human structural checkpoint + exact materialization | stop-gate handoff | Phase 10 gate/finalizer |
| S4 | Reduce retrieved prompt/KV budget | 5 | **new** disjoint-DEV context sweep |
| S5 | Simulate contextual sparsity and measure router runtime | 6/7 | **new student wrapper**, reusing Phase 6/7 primitives |
| S6 | Compare dense/NF4/INT8/(optional GGUF) on DEV | 7/deployment quantization | **new** quantization sweep |
| S7 | Freeze settings and run final production RAG comparison | 8 release experiment | Phase 11 Full |
| S8 | Human release checkpoint + package | 8 release package | Phase 11 Package |

## Why not blindly duplicate every teacher phase?

Some teacher techniques are already upstream or mathematically conditional:

- Dense/SFT behavior learning happens in Phase 9; rerunning teacher Phase 1 would throw away the accepted student baseline.
- Preference alignment can occur in Phase 9/9B. Structural Phase 10 recovery uses authoritative same-tokenizer KD to restore quality after pruning.
- Model soups only make sense for checkpoints/adapters with identical tensor topology. The Phase 10 frontier intentionally contains different physical architectures, so averaging them is invalid. Compatible same-architecture adapter experiments remain optional research, not a mandatory compression stage.
- Phase 10 already performs the student equivalents of physical pruning, recovery, DEV selection, and sealed architecture comparison.
- Phase 11 already provides immutable RAG snapshots, answerability calibration, measured total-process memory, quantization, rollback, and release packaging.

## Completion semantics

The master workflow has only four broad end states:

- `RUNNING` — more automated work exists.
- `WAITING_FOR_*` — all automation up to a required human/materialization boundary is done. This is a **clean checkpoint**, not failure.
- `FAILED` — an executable stage failed or required evidence is technically invalid.
- `COMPLETE` — Phase 11 human PASS exists **and** the release package was successfully created. No further automated work remains.

The runner never edits a human gate to PASS.

## Incremental artifacts

```text
artifacts/student_optimization_v2/
  WORKFLOW_STATE.json
  layer_sensitivity/
  context_optimization/
  sparse_runtime/
  quantization_sweep/
  runtime_lab/

reports/
  STUDENT_OPTIMIZATION_V2_MASTER_REPORT.md
```

`WORKFLOW_STATE.json` and the master Markdown report are updated before and after every expensive stage, and immediately on checkpoint/failure.

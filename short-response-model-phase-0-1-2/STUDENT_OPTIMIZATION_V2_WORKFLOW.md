# Student Optimization v2.0 — complete architecture and workflow

## Objective

Take the **already-proven grounded student** from Phase 9/9B and subject it to a coherent efficiency program analogous to the original teacher Phase 0–8 workflow, without duplicating stages that already exist or violating DEV/TEST discipline.

The program optimizes in this order:

```text
accepted behavior
    ↓
layer sensitivity
    ↓
physical depth / width / GQA reduction
    ↓
recovery distillation
    ↓
human structural selection
    ↓
context / KV budget
    ↓
contextual sparsity simulation
    ↓
measured sparse runtime
    ↓
quantization/runtime backend selection
    ↓
frozen final production RAG test
    ↓
human release decision
    ↓
reproducible package
```

This ordering follows the repository's long-standing principle that behavior must be proven before efficiency, and simulated sparsity must not be mistaken for measured speed.

## What already existed before v2.0

### Phase 10 already covered a large part of teacher structural optimization

Phase 10 already:

- evaluates the accepted student baseline;
- profiles residual coordinates, MLP channels, and attention heads;
- physically deletes decoder layers;
- physically slices hidden dimensions;
- physically slices gated-MLP channels;
- physically slices grouped-query attention groups;
- fresh-reloads every candidate;
- evaluates raw candidates on grounded DEV conditions;
- recovery-distills a bounded top set under the authoritative teacher;
- optionally compares compact pretrained challengers;
- freezes one candidate on DEV;
- evaluates only that frozen architecture on sealed TEST;
- writes a human gate and rollback information.

### Phase 11 already covered production RAG and final deployment measurement

Phase 11 already:

- builds an immutable knowledge/procedure snapshot;
- calibrates answerability;
- routes/retrieves/gates/generates/post-grounds;
- measures serialized bytes, index bytes, total process RSS, peak VRAM, latency, and quality;
- supports dense, bitsandbytes NF4, bitsandbytes INT8, and llama.cpp GGUF experiments;
- preserves dense fallback;
- packages the human-approved runtime.

## What v2.0 adds

### 1. Real student layer-sensitivity evidence

The old Phase 10 `uniform-drop` depth search did not reproduce the teacher's every-layer deletion study. v2.0 now physically deletes **each student layer one at a time**, fresh-reloads it, and evaluates it on non-sealed grounded data. It records a ranked deletion order but does **not** assume multi-layer effects are additive.

The v2 Phase 10 config therefore includes both:

- importance-guided depth candidates based on the empirical scan; and
- uniform-drop controls matched for parameter count.

Every multi-layer candidate is still independently rebuilt and evaluated.

### 2. Context/KV budget selection before final runtime

Instead of hardcoding one RAG budget, v2.0 creates group-disjoint DEV calibration/selection subsets and compares multiple retrieval/evidence profiles. Each profile gets its own answerability calibration. Selection uses **actual measured prompt-token reduction** versus proportional quality loss.

The frozen profile is recalibrated on the full DEV set only after selection, inside the final Phase 11 run.

### 3. Student contextual-sparsity pipeline

The master workflow reuses the tested Phase 6/7 primitives directly on the Phase 10 student:

- dense activation traces;
- oracle MLP block masks;
- masked-generation quality frontier;
- per-layer router training when worthwhile;
- dense-vs-routed runtime benchmark;
- automatic `DENSE_RUNTIME` fallback when sparse execution is not worthwhile.

A sparse recommendation remains a feature-flag research artifact in v2.0; it is not silently injected into the Phase 11 package because the production RAG runtime does not yet own a router-aware execution backend.

### 4. DEV-only quantization sweep

NF4, INT8, and optionally GGUF can be tested in parallel against the dense Phase 10 student using the already-frozen context profile. Backend failures are recorded instead of aborting the whole sweep. The selected backend is chosen on DEV total-process memory versus quality opportunity loss.

Only that DEV-selected backend proceeds to the final Phase 11 sealed experiment.

### 5. A durable automation state machine

Every expensive stage writes state before execution and after completion. Human stop gates are first-class checkpoints. The same master command resumes from the boundary; users do not need to manually reconstruct a long sequence of child commands.
A completed Phase 10 or Phase 11 registry is also a resume boundary: the orchestrator does not rerun that phase's `Full` gate builder, because doing so would regenerate a `PENDING` gate and could erase a reviewer-recorded `PASS`. Intentional rebuilds require `-ForcePhase10Rerun` / `-ForcePhase11Rerun` and a new human review cycle.

## Why v2.0 does not force model soup across Phase 10 architectures

Model soup is useful when ingredients share the same parameter topology and fine-tuned solutions are compatible. Phase 10 deliberately changes tensor shapes and layer counts. Averaging `combined_small_target` with `depth_drop_1_importance`, for example, is not a meaningful weight average because their tensors do not align.

Therefore v2.0 keeps model soup conditional rather than manufacturing a misleading analogue. If multiple same-architecture LoRA/DPO variants are later produced, the existing compatibility validation and delta-soup infrastructure can be used as an optional research branch.

## Fresh full command

Run from the repository root in the activated environment:

```powershell
.\scripts\student_optimization_v2\run_student_optimization_v2.ps1 `
  -TeacherModel C:\path\to\authoritative_teacher `
  -ReuseExisting
```

For the current v1.7.5 Qwen grounded-recovery lineage, the default dataset/knowledge/calibration paths point at:

```text
artifacts\phase_9_grounded_recovery_v1_7_5_qwen\
```

The official workflow should begin only after Phase 9/9B human PASS and materialization.

## Human checkpoint behavior

### First execution

The runner progresses automatically until Phase 10 is complete, then exits with:

```text
WAITING_FOR_PHASE10_HUMAN_PASS
```

Review Phase 10. Record the explicit PASS and candidate decision. Re-run the **same command**.

### Second execution

The runner resolves/materializes the Phase 10 choice, then automatically executes context, sparsity, quantization, and final Phase 11 work. It exits at:

```text
WAITING_FOR_PHASE11_HUMAN_PASS
```

Review the final frozen runtime and sealed outputs. Record PASS. Re-run the same command.

### Final execution

The runner packages the exact approved Phase 11 candidate and changes the master workflow status to:

```text
COMPLETE
checkpoint = NO_FURTHER_AUTOMATED_WORK
```

That is the only state considered fully finished.

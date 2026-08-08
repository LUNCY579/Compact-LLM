# Student Distillation Playground v1.0 — package v1.7.6

This is an **exploratory playground**, intentionally separate from all official phase directories. It is designed for exactly the kind of student demonstration that is useful on video: give the model a compact logic problem, supplied evidence, or source-state-aware memory; show the final answer; then show correctness, latency, generated tokens, throughput, and peak CUDA memory.

The playground reuses the repository's official Phase 9 generation path and Phase 9/9B prompt builders instead of inventing a second inference protocol. Its data and reports are nevertheless **not official phase-gate evidence**. Do not tune or select an official student from these showcase cases and then claim the same cases as held-out evidence.

## Recommended point in the official workflow

**Best first use: finish Phase 9 through the selected student's full Oracle/Real/adversarial evaluation and human selection, then run the playground before Phase 10.** At that point the video measures the actual distilled student rather than an intermediate training checkpoint.

- Logic and evidence showcase: run after Phase 9 Stage E / final candidate freeze; human Phase 9 PASS is preferred if you call it the accepted student.
- Memory-reconstruction showcase: run `--memory-mode two_pass` only after Phase 9B reconstruction training/evaluation; use `direct` to test the one-pass state-aware behavior of any Phase 9/9B-compatible candidate.
- Phase 10 is **not required** for the first playground run. Phase 10 is a useful later comparison because it asks how much of the same behavior survives structural reduction.

## Included novel showcase cases

The bundled suite contains **34 newly authored cases** that are not copied from the official Phase 9/9B benchmark:

- 10 compact logic games: ordering, implication, XOR, scheduling, elimination, path reasoning.
- 12 evidence tasks: direct support, multi-document reasoning, distractors, missing evidence, conflict, stale/current precedence, injection resistance, arithmetic, tentative evidence, and procedure application.
- 12 memory tasks: ACCEPT/ADAPT/REJECT/CONFLICT/INSUFFICIENT-style situations using current state, source state, validity conditions, trust, and supersession.

Individual files are also available under `datasets/` so a recording can focus on one capability.

## Fast video command

```powershell
.\Student_Distillation_Playground_v1.0\run_playground.ps1 `
  -Model .\path\to\student_base_or_full_model `
  -Adapter .\path\to\selected_student_adapter `
  -VideoMode `
  -ShowContext `
  -PauseSeconds 0.5
```

For a merged/full checkpoint, omit `-Adapter`.

The default output is:

```text
playground_artifacts/student_distillation_showcase/
  results.jsonl     one durable record per case
  summary.json      aggregate accuracy/latency/throughput/VRAM
  REPORT.md         human-readable report
```

## Evidence-only recording

```powershell
.\Student_Distillation_Playground_v1.0\run_playground.ps1 `
  -Model .\path\to\model `
  -Adapter .\path\to\adapter `
  -TaskType evidence `
  -VideoMode `
  -ShowContext
```

## Memory reconstruction recording

Direct one-pass state-aware answer:

```powershell
.\Student_Distillation_Playground_v1.0\run_playground.ps1 `
  -Model .\path\to\model `
  -Adapter .\path\to\adapter `
  -TaskType memory `
  -MemoryMode direct `
  -VideoMode `
  -ShowContext
```

Auditable two-pass Phase 9B reconstruction then short answer:

```powershell
.\Student_Distillation_Playground_v1.0\run_playground.ps1 `
  -Model .\path\to\model `
  -Adapter .\path\to\phase9b_adapter `
  -TaskType memory `
  -MemoryMode two_pass `
  -VideoMode `
  -ShowContext
```

In two-pass mode the report additionally records reconstruction JSON validity, decision accuracy, and source-ID precision/recall.

## Run a custom dataset

Copy any bundled JSONL and edit it, or follow `schemas/playground_case.schema.json`.

```powershell
.\Student_Distillation_Playground_v1.0\run_playground.ps1 `
  -Model .\path\to\model `
  -Dataset .\my_tests\new_cases.jsonl `
  -OutputDir .\playground_artifacts\my_run
```

Keep custom playground datasets outside official `data/phase_*` trees unless you deliberately promote them through the project's dataset review, split, leakage, and provenance process.

## What the video output means

`PASS/MISS` is case-contract accuracy, not a broad LLM quality grade. The tool also records total case latency, generation latency, tokens, tokens/sec, and CUDA peak allocated memory. Evidence grounding is reported as a diagnostic. In two-pass memory mode, the reconstruction decision is scored separately from the final answer.

The first selected case is run once as an unscored warmup by default so CUDA/kernel initialization does not dominate the latency numbers. Set `--warmup 0` when calling the Python script directly if cold-start latency is what you want to demonstrate.

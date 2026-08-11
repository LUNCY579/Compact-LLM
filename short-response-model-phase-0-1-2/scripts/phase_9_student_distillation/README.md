# v2.3.1 Grokking Dynamics Observatory

Use `run_phase_9_grokking_dynamics_v231.ps1` after a completed v2.3 run. The branch is designed to distinguish delayed generalization from readout thresholds, ordinary feature learning, or memorization. It keeps lightweight LoRA snapshots at every save event while retaining only a small number of resumable optimizer checkpoints, so long temporal curves are practical on consumer hardware.

Primary design: `../../GROKKING_DYNAMICS_V2_3_1.md`.

## v2.3 preservation-constrained residual capability distillation

v2.3 starts from the **materialized v2.2 greedy_soup** rather than continuing broad SFT. It mines only gold-verified cases where the greedy anchor is wrong and at least one v2.2 teacher is correct, trains separate answerable-reasoning and procedure residual LoRAs with preservation replay + greedy-anchor KL, calibrates how strongly each delta should be applied, and evaluates a small R/P composition family on DEV + structurally held-out OOD-DEV. It also introduces `query_aware_v23` for the fresh procedural corpus so an explicitly requested missing locator fails closed instead of borrowing a sibling record.

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_residual_capability_v23.ps1 `
  -V22RunRoot ".\artifacts\phase_9_capability_composition_v22_lr1e4_s1200" `
  -OutputRoot ".\artifacts\phase_9_residual_capability_v23" `
  -RequireCuda
```

The v2.2 sealed TEST is consumed. The first run never uses it and should not pass `-RunFreshSealedTest`. Full design: repository-root `RESIDUAL_CAPABILITY_V2_3.md`.

### v2.3.1 diagnostic-completion (no retrain)

If dose search retained weight 0 and the final matrix/oracle collapsed to greedy-only, complete research diagnostics without retraining:

```powershell
.\scripts\phase_9_student_distillation\run_residual_diagnostics_v231.ps1 `
  -V23Root ".\artifacts\phase_9_residual_capability_v23" `
  -RequireCuda `
  -ReuseExisting
```

This rematerializes rejected residual doses from retained LoRAs, keeps them in the research matrix as `REJECTED_BY_POLICY`, rebuilds a multi-specialist oracle (including answerable-only), and writes teacher-overlap + dose-ladder reports under `diagnostics_v231/`. Production selection remains `greedy_anchor` when policy says so.

---

## v2.1.1 preservation-first procedural recovery

The observed v2.1 DEV run established a crucial distinction: `query_aware_v21` retrieval restored the original student's procedural correctness with **unchanged weights**, while the tested recovery training improved the hard challenge but substantially damaged the strong original path. v2.1.1 therefore keeps the unchanged query-aware student as **candidate zero** and makes every weight-updated checkpoint earn its preservation cost.

Use:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_procedural_recovery_v211.ps1 `
  -StudentModel <warm/base checkpoint> `
  -StudentAdapter <current selected adapter> `
  -RequireCuda
```

The dedicated path uses a 768-row hard Stage A burst, then two 60-step preservation-first consolidation checkpoints over a 4,820-row mix that is 64.7% original train-only replay. It evaluates original Phase 9 DEV and the new challenge after every checkpoint and may intentionally select `query_aware_only`. See the repository-root `PROCEDURAL_RECOVERY_V2_1_1.md`.

---

# Phase 9 — grounded-recovery student proof

## v1.7.7 selector-gate correction

Stage B remains an intentionally weak, untrained baseline. Its prompt/decode-integrity failures are reported but are no longer allowed to poison all Stage D SFT candidates. Candidate-owned integrity remains strict, and initialization behavior-coverage failures can still invalidate the reference. Under `-ReuseExisting`, Stage D and final DEV selection files are recomputed from existing summaries so old blocked roots can continue without retraining.


## v1.7.2 governing order

The primary question is no longer “which compressed student wins?” It is first:

> Can the reviewed training and evaluation path teach one inherited warm student to use supplied evidence correctly?

The official order is:

```text
A. accepted teacher + exact Oracle evidence
B. untrained warm_344m + exact Oracle evidence
C. 12–24-row deliberate overfit and saved/reloaded verification
D. three-run warm_344m SFT sweep under Oracle evidence
E. Real retrieval and adversarial evaluation
F. progressive KD from the best proven SFT checkpoint
G. only then narrow architecture comparison
```

Use `run_phase_9_grounded_recovery.ps1` for this sequence. The legacy/general `run_phase_9.ps1` remains available, but Hybrid mode now requires a grounded-recovery gate with `architecture_sweep_unlocked=true`, unless the user explicitly marks the run as research-only with `-AllowUnprovenArchitectureSweep`.

The tiny overfit criterion is a technical wiring check, not a production quality floor. It verifies labels, completion masking, optimizer steps, adapter publication, checkpoint reload, prompt consistency, and measurable learning before expensive training.

## Behavior Curriculum v2 foundation retained


Phase 9 consumes the explicitly accepted Phase 8 teacher and establishes whether a smaller student can use supplied evidence, abstain safely, remain concise, and preserve ordinary social conversation. It is a foundation phase, not the final 80M–120M deployment search; Phase 10 performs structural reduction after Phase 9 selects a valid intermediate student.

## Behavior Curriculum v2

The original smoke run exposed three different problems: too little training data, direct KD before the student learned the task, and invalid evaluation denominators. Behavior Curriculum v2 changes the official sequence to:

```text
accepted teacher + accepted real rows
                ↓
leakage-safe real train/dev/test split
                ↓
human-approved real curriculum
                +
versioned fictional behavior foundation (training only)
                ↓
weighted deterministic mixture
                ↓
curriculum SFT
                ↓
confidence-filtered progressive KD from the SFT artifact
                ↓
optional on-policy recovery
                ↓
closed / oracle / real / filtered-adversarial / passthrough-adversarial evaluation
```

The bundled synthetic foundation contains 336 authored prompts covering supported extraction, exact abstention, distractors, retrieval injection, conflicting evidence, semantic-version procedures, concise social responses, and completion boundaries. Its facts are fictional and cannot leak real project answers. It supplements human-reviewed real rows; it does not replace broad language pretraining or held-out real-domain evaluation.

## Student tracks

1. **Warm depth-copy student** — exact complete teacher-layer inheritance; official production-foundation track.
2. **Narrow same-tokenizer random students** — 100M/130M/170M research controls. The behavior foundation can teach task format, but it cannot create broad language competence from a small corpus.
3. **Pretrained compact challenger** — sequence-level SFT first; direct Qwen-vocabulary KL is rejected when tokenizers differ.

The original teacher remains authoritative at every compression rung.

## What changed in evaluation

- Social rows are no longer mislabeled as unsupported factual questions.
- Every functional rate reports its actual numerator and denominator.
- Zero-applicability metrics are `null`, never fake `0.0` or `1.0` values.
- Supported correctness comes from explicit response contracts plus grounding, not a broad style score.
- Training comparisons use bounded absolute deltas, so a zero reference cannot create changes on the order of `1e11`.
- Smoke packets are never production-recommendable.
- A production candidate must demonstrate at least one correct supported answer and one correct abstention whenever those case types exist.
- `adversarial_rag` measures system filtering; `adversarial_passthrough_rag` retains malicious evidence and measures model-level resistance.

## Bootstrap

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Bootstrap `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model
```

Bootstrap creates the initialization, FTS5 index, teacher candidates, and the human-review packet. Authored answers in prompt records are included as clearly labeled candidates, but review remains mandatory for real project rows.

## Research-ready Pilot

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -UseLora `
  -Teacher4Bit `
  -MaxSteps 600
```

The default Pilot requires at least 30 accepted rows, eight independent connected topic/document groups, and development/test coverage for supported and unsupported cases. Add more independent prompts and documents when this gate fails; duplicated paraphrases do not create held-out evidence.

The bundled foundation is included automatically. Use `-SkipFoundation` only for an ablation. Use `-SkipCurriculum` only for an ablation. Direct KD from initialization is disabled unless `-RunDirectKD` is supplied.

## Explicit smoke test

A six-row packet can prove that the code runs, but not that a student learned the task:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -SmokeTest `
  -UseLora `
  -Teacher4Bit `
  -MaxSteps 100
```

Smoke summaries carry `research_ready: false` and cannot be automatically recommended.

## Hybrid comparison

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -StudentSuite Hybrid `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -AcknowledgeRandomControl `
  -UseLora `
  -Teacher4Bit `
  -MaxSteps 600
```

Hybrid mode is expensive because every profile receives curriculum SFT and progressive KD. Run the warm `Single` track first. Narrow profiles remain research controls regardless of their apparent memory margin.

## Interpreting failures

- **Poor oracle RAG:** training, prompt use, checkpoint loading, or the evaluator is failing.
- **Good oracle but poor real RAG:** retrieval or threshold calibration is failing.
- **Good filtered adversarial but poor passthrough adversarial:** the sanitizer works, but the student still follows evidence-borne instructions.
- **Correct answers but poor abstention:** increase real unsupported examples and inspect the external gate.
- **SFT improves but KD regresses:** retain SFT; inspect teacher target confidence and KD loss components rather than forcing KD.
- **Random narrow model remains incoherent:** expected; behavior data is not language pretraining.

No candidate advances only because it is small. Phase 9 still uses the shared proportional opportunity engine, plus functional-readiness blockers and an explicit human decision.

## v2.1.2 broad general-reasoning transfer

After v2.1.1 showed that narrow procedural recovery can preserve the original query-aware path yet produce only a tiny residual challenge gain, v2.1.2 adds a separate **general reasoning** curriculum. It trains transferable operations over prompt-contained facts rather than memorizing general-world knowledge.

Run:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_general_reasoning_v212.ps1 `
  -StudentModel "C:\path\to\student" `
  -StudentAdapter "C:\path\to\adapter" `
  -RequireCuda
```

The runner compares the unchanged input student, Stage A broad reasoning, Stage B1 mixed generalization/preservation, and Stage B2 lower-rate consolidation on three DEV axes: original Phase 9 RAG, v2.1.1 procedural challenge, and v2.1.2 general reasoning. Sealed TEST is optional and only runs after DEV selection.

See `GENERAL_REASONING_RECOVERY_V2_1_2.md`.

## v2.2 capability composition

The v2.2 branch is documented in `../../CAPABILITY_COMPOSITION_V2_2.md`.

Primary runner:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_capability_composition_v22.ps1 `
  -StudentModel <student-or-base> `
  -StudentAdapter <optional-input-adapter> `
  -RequireCuda
```

It implements four tracks from one immutable anchor: broad reasoning, KL-anchored broad reasoning, parallel specialists, and weighted/greedy/layer-separated composition. All candidates use the same DEV evaluation axes and the exact same frozen query-aware retrieval calibrations.

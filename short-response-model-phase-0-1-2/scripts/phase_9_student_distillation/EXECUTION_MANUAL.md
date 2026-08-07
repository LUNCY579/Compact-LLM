# Phase 9 execution manual — grounded recovery + RAG reasoning v1.7.3


## v1.7.3 canonical corpus and sealed-test rule

The official runner now defaults to `data/phase_9_student_distillation/rag_reasoning_v2`. The corpus validator must pass before the ladder starts. The authored split is preserved exactly: 392 train / 67 dev / 71 sealed test rows. Teacher preflight, warm initialization, the SFT sweep, Real-RAG diagnosis, and optional KD refinement all use **dev**. Only after a final SFT/KD artifact is selected on dev does the runner evaluate teacher, initialization, and the single frozen candidate on **test**.

The integrated corpus includes supported extraction, multi-document support, tentative and ambiguous evidence, missing information, equal-authority conflicts, prompt injection, stale/current precedence, procedures, and social replies. Foundation rehearsal examples are deduplicated against both dev and test questions before mixing.

## Governing rule

Do not begin another broad warm/narrow SFT/KD comparison until one inherited warm student has proved that the complete teaching path works. The official order is:

```text
Teacher Oracle evidence
→ untrained warm initialization Oracle baseline
→ 12–24-row deliberate overfit wiring proof
→ three-run warm SFT sweep
→ Oracle selection against initialization
→ Real/adversarial RAG
→ optional progressive KD from the selected SFT artifact
→ only then narrow architecture controls
```

The v1.7.1 prompt-integrity, decode-provenance, bounded-comparison, and teacher-preflight protections remain active. v1.7.2 changes the experimental order and adds direct evidence that training, publication, reload, and evaluation are wired correctly.

## Required inputs

1. An exact local teacher checkpoint accepted by Phase 8, unless `-AllowUnreleasedTeacher` is deliberately used for research.
2. Human-reviewed teacher JSONL with accepted targets.
3. Research-ready reviewed rows: at least 30 rows and eight leakage-safe connected components under the existing split policy.
4. CUDA PyTorch, Transformers, PEFT, and repository requirements.
5. A **new output root** for the first command. Use `-ReuseExisting` only when deliberately continuing later ladder stages in that same root.

## Recommended first run: SFT-only grounded recovery

Run the complete capability ladder without KD:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\data\phase_9_student_distillation\rag_reasoning_v2\teacher_candidates.reviewed.jsonl `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_3 `
  -UseLora `
  -Teacher4Bit
```

Do **not** add `-RunKD` to the first clean experiment. The runner will stop before later stages when an earlier proof fails.

## Continue with progressive KD only after SFT succeeds

After the overfit test and full Oracle SFT stage pass in the existing root:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\data\phase_9_student_distillation\rag_reasoning_v2\teacher_candidates.reviewed.jsonl `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_3 `
  -UseLora `
  -Teacher4Bit `
  -ReuseExisting `
  -RunKD
```

KD starts from the exact selected SFT adapter. Direct KD from initialization is not the official recovery path.

## Stage-by-stage commands

Stage-by-stage execution is preferable when debugging because each command leaves a complete report before the next expensive stage.

### A–B — Teacher and warm-initialization Oracle preflight

Use a fresh output root and do not pass `-ReuseExisting`:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Preflight `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\data\phase_9_student_distillation\rag_reasoning_v2\teacher_candidates.reviewed.jsonl `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_3 `
  -UseLora `
  -Teacher4Bit
```

Inspect:

```text
artifacts\phase_9_grounded_recovery_v1_7_3\GROUNDED_RECOVERY_LADDER.md
artifacts\phase_9_grounded_recovery_v1_7_3\evaluation_teacher_oracle\summary.json
artifacts\phase_9_grounded_recovery_v1_7_3\evaluation_warm_initialization_oracle\summary.json
```

The teacher must pass Stage A. A failure here means the prompt, evidence, answer contract, decoding path, teacher, or reviewed data must be corrected before student training.

### C — Deliberate tiny-overfit wiring proof

Continue the same root deliberately:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Overfit `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\data\phase_9_student_distillation\rag_reasoning_v2\teacher_candidates.reviewed.jsonl `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_3 `
  -UseLora `
  -Teacher4Bit `
  -ReuseExisting
```

Defaults:

```text
18 unique rows
12 training repeats
300 optimizer steps
minimum reloaded capability score: 0.80
minimum improvement over initialization: 0.25
```

These are **technical wiring criteria**, not universal production thresholds. The stage also verifies nonzero optimizer steps, a changed artifact hash, saved/reloaded parity, valid outputs, and applicable case denominators.

### D — Compact warm SFT sweep

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Sweep `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\data\phase_9_student_distillation\rag_reasoning_v2\teacher_candidates.reviewed.jsonl `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_3 `
  -UseLora `
  -Teacher4Bit `
  -ReuseExisting
```

The official diagnostic sweep is intentionally small:

| Candidate | Learning rate | Steps | LoRA rank |
|---|---:|---:|---:|
| `lr5e5_s600` | `5e-5` | 600 | 16 |
| `lr5e5_s1200` | `5e-5` | 1200 | 16 |
| `lr1e4_s1200` | `1e-4` | 1200 | 16 |

Inspect:

```text
artifacts\phase_9_grounded_recovery_v1_7_3\warm_sft_sweep\SELECTION_REPORT.md
artifacts\phase_9_grounded_recovery_v1_7_3\warm_sft_sweep\selection.json
```

Selection is based on Oracle capability relative to the **untrained warm initialization**, not on compression relative to a weak or zero-valued reference. An all-zero or denominator-invalid frontier produces `NO_VALID_QUALITY_FRONTIER`.

### E — Real and adversarial retrieval

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\data\phase_9_student_distillation\rag_reasoning_v2\teacher_candidates.reviewed.jsonl `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_3 `
  -UseLora `
  -Teacher4Bit `
  -ReuseExisting
```

This evaluates:

```text
closed_book
oracle_rag
real_rag
adversarial_rag
adversarial_passthrough_rag
```

Only after Oracle evidence use works does this stage diagnose FTS5 retrieval, evidence selection, calibration, distractors, conflicts, and injection filtering.

## Training mixture and loss emphasis

The runner writes a separate deterministic training-only view with this target composition:

```text
45% supported evidence extraction
25% missing-evidence abstention
10% conflicting evidence
10% injection-resistant evidence use
 5% procedures
 5% social conversation
```

The reviewed source and expanded curriculum remain unchanged. The training objective retains ordinary completion CE/KL while emphasizing:

```text
first answer token:       2.0
evidence-anchor tokens:   1.75
abstention sequence:      2.0
EOS/end-of-turn:          1.5
```

## Returning to Hybrid architecture controls

The ordinary Phase 9 runner now locks `-StudentSuite Hybrid` until the grounded ladder proves both the tiny overfit stage and full Oracle improvement. After that proof:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -StudentSuite Hybrid `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\data\phase_9_student_distillation\rag_reasoning_v2\teacher_candidates.reviewed.jsonl `
  -OutputRoot .\artifacts\phase_9_hybrid_after_grounded_proof `
  -GroundedRecoveryGate .\artifacts\phase_9_grounded_recovery_v1_7_3\grounded_recovery_ladder.json `
  -AcknowledgeRandomControl `
  -UseLora `
  -Teacher4Bit
```

`-AllowUnprovenArchitectureSweep` exists only as an explicit research override. It must not be used to claim an official Phase 9 PASS.

## Existing v1.7.1 failed-run diagnosis

The v1.7.1 diagnostics remain useful for preserving and classifying old artifacts:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_diagnostics.ps1 `
  -ArtifactRoot .\artifacts\phase_9_student_distillation `
  -Benchmark .\artifacts\phase_9_student_distillation\pilot_dataset_v2\test.jsonl
```

Legacy evaluations lacking prompt/decode provenance should be regenerated. Existing student checkpoints may remain as research controls, but they are not used as evidence that the grounded ladder passed.

## Human gate and Phase 9B

A completed command does not self-approve a model. Review `reports\PHASE_9_STOP_GATE.md` and the exact assessment artifact. Human PASS must name the exact candidate and role. Phase 9B remains blocked until that explicit Phase 9 approval exists.

# Phase 9 grounded recovery — partner execution guide

This is the shared runbook for the official grounded-recovery ladder (pipeline v1.7.3 + RAG Reasoning Corpus v2.1).

Personal machine notes and restart history live in `student_distillation.md` and are not required here.

## Goal

Prove one warm 344M depth-copy student on the evidence-use path before any narrow-architecture comparison:

```text
Preflight (Stages A–B)
  → Overfit wiring proof (Stage C)
  → Warm SFT sweep + Oracle selection (Stage D)
  → Full Real/adversarial evaluation (Stage E)
  → optional progressive KD
  → only then narrow 170M/130M/100M controls
```

## Prerequisites

1. Work from the package root:

```powershell
cd <repo>\short-response-model-phase-0-1-2
```

2. Activate the project virtualenv (must provide CUDA PyTorch):

```powershell
.\.venv\Scripts\Activate.ps1
# or, if the venv lives one level up:
..\ .venv\Scripts\Activate.ps1
```

3. Confirm imports:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

4. Integrated corpus must be present (defaults in the runner):

```text
data\phase_9_student_distillation\rag_reasoning_v2\
  teacher_candidates.reviewed.jsonl
  knowledge.jsonl
  procedures.jsonl
```

The runner validates this corpus automatically when the manifest exists.

5. Teacher checkpoint:
   - **Preferred:** local Phase 8–accepted teacher path.
   - **Research only:** an external checkpoint such as local `Qwen/Qwen3-0.6B`, with `-AllowUnreleasedTeacher`.

6. First command in a new experiment must use a **fresh** `-OutputRoot`. Add `-ReuseExisting` only when deliberately continuing later stages in that same root.

## Variables

```powershell
# Preferred production-style teacher (example path — replace with your accepted checkpoint)
$Teacher = '.\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model'

# Fresh experiment root for this partner run
$RunRoot = '.\artifacts\phase_9_grounded_recovery_partner_run'

# Research alternative (only if not using a Phase 8 teacher):
# $Teacher = '<absolute-or-relative-local-Qwen3-0.6B-snapshot>'
# and add -AllowUnreleasedTeacher to every command below
```

Verify:

```powershell
Test-Path $Teacher
```

## 1. Preflight (Stages A–B)

Do **not** pass `-ReuseExisting`.

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Preflight `
  -TeacherModel $Teacher `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora
```

If using an unreleased research teacher, append:

```powershell
  -AllowUnreleasedTeacher
```

Inspect:

```powershell
Get-Content "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
```

Also useful:

```text
$RunRoot\evaluation_teacher_oracle\summary.json
$RunRoot\evaluation_teacher_oracle\results.jsonl
$RunRoot\evaluation_warm_initialization_oracle\summary.json
```

**Gate:** Stage A must be `PASS` before continuing.  
Stage B may `FAIL` for an untrained warm student; that alone does not unlock later stages, but Preflight still completes when Stage A passes.

## 2. Overfit wiring proof (Stage C)

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Overfit `
  -TeacherModel $Teacher `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -ReuseExisting
```

(Add `-AllowUnreleasedTeacher` if used in Preflight.)

Inspect:

```text
$RunRoot\overfit_probe\
$RunRoot\GROUNDED_RECOVERY_LADDER.md
```

**Gate:** Stage C must be `PASS`.

## 3. Warm SFT sweep (Stage D)

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Sweep `
  -TeacherModel $Teacher `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -ReuseExisting
```

Inspect:

```powershell
Get-Content "$RunRoot\warm_sft_sweep\SELECTION_REPORT.md"
```

**Gate:** selection status must be `SFT_CANDIDATE_SELECTED`, not `NO_VALID_QUALITY_FRONTIER`.

## 4. Full Real/adversarial evaluation (Stage E), no KD yet

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel $Teacher `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -ReuseExisting
```

Inspect:

```text
$RunRoot\evaluation_teacher_full\summary.json
$RunRoot\evaluation_selected_sft_full\summary.json
reports\PHASE_9_STOP_GATE.md
```

## 5. Optional progressive KD

Run only after Stage D/E show a selected SFT candidate that improves over initialization.

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel $Teacher `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -Teacher4Bit `
  -RunKD `
  -ReuseExisting
```

If `bitsandbytes` fails on Windows, omit `-Teacher4Bit` and keep the rest unchanged.

## Rules partners should not skip

1. Do not start Overfit/Sweep/Full until Stage A is `PASS`.
2. Do not add `-RunKD` to the first Full run.
3. Do not compare narrow random students until `architecture_sweep_unlocked=true` in `$RunRoot\grounded_recovery_ladder.json`.
4. Do not train on sealed test / held-out evaluator files from the corpus.
5. Use one OutputRoot per experiment; use `-ReuseExisting` only to continue that root.

## One-shot alternative

For an uninterrupted clean run after prerequisites are ready (still no KD):

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel $Teacher `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora
```

The runner stops automatically if an earlier stage fails. Staged Preflight → Overfit → Sweep → Full is preferred when debugging.

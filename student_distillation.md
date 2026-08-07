# Phase 9 grounded recovery — current restart path (v1.7.3)

Personal reference only. Partner-facing commands: [`PHASE_9_GROUNDED_RECOVERY_PARTNER_GUIDE.md`](PHASE_9_GROUNDED_RECOVERY_PARTNER_GUIDE.md).

Use the integrated **RAG Reasoning Corpus v2.1** defaults. Do not continue the failed `..._v3` Sweep; that run hit `NO_VALID_QUALITY_FRONTIER` because warm-SFT candidates scored 0 on supported answers (foundation-template bleed), not because of a path bug.

## Why the previous Sweep stopped

Status was `NO_VALID_QUALITY_FRONTIER` (exit code 2). All three SFT settings were invalid:

- zero correct supported answers (e.g. `9.10.0 Lush runtime.` instead of `900 lumens.`)
- zero successful procedure cases
- Stage B decode-integrity issues also marked every candidate `initialization_reference_invalid`

Overfit (Stage C) had already PASSed, so training wiring is OK; the full balanced curriculum did not transfer to the sealed test facts.

## 0. Shell setup

```powershell
cd C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2

$QwenBase = 'C:\Users\clclc\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca'
$RunRoot = '.\artifacts\phase_9_grounded_recovery_v1_7_3_qwen'

Test-Path $QwenBase
```

Reviewed data / knowledge / procedures default to:

```text
.\data\phase_9_student_distillation\rag_reasoning_v2\
```

(530 accepted reviewed rows; corpus validation PASS.)

## 1. Preflight (start here)

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Preflight `
  -TeacherModel $QwenBase `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -AllowUnreleasedTeacher
```

Inspect:

```powershell
Get-Content "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
```

Continue only when Stage A is `PASS`.

## 2. Overfit

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Overfit `
  -TeacherModel $QwenBase `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## 3. Sweep

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Sweep `
  -TeacherModel $QwenBase `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## 4. Full (no KD first)

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel $QwenBase `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## 5. Optional KD after Full SFT succeeds

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel $QwenBase `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -Teacher4Bit `
  -RunKD `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## Cleanup

Safe to delete when unused: `phase_9_qwen3_06b_grounded_20260806_v3` (failed frontier). Keep `data\phase_9_student_distillation\rag_reasoning_v2` and the new `$RunRoot`.

Yes. Use the local Qwen3-0.6B snapshot as the temporary diagnostic teacher and run the grounded-recovery ladder from the beginning.

The important change is to use a **fresh artifact root** and regenerate the teacher review packet with Qwen rather than reusing answers produced by the weak teacher.

## 0. Open PowerShell in the repository root

```powershell
cd C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2
```

Set reusable variables:

```powershell
$QwenBase = 'C:\Users\clclc\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca'

$RunRoot = '.\artifacts\phase_9_qwen3_06b_grounded_20260806'

$ReviewedData = Join-Path $RunRoot 'teacher_candidates.reviewed.jsonl'
```

Verify the snapshot:

```powershell
Test-Path $QwenBase
```

It should print:

```text
True
```

The scripts load this path locally through Hugging Face `from_pretrained()`. They also explicitly disable Qwen3 thinking mode for Phase 9 prompts, so the 32-token answer budget will not be consumed by `<think>` output.

---

# 1. Generate a new Qwen teacher review packet

Use the expanded research corpus rather than the six-row sample corpus:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Bootstrap `
  -StudentSuite Single `
  -TeacherModel $QwenBase `
  -StudentSource $QwenBase `
  -Knowledge .\data\phase_9_student_distillation\expanded\knowledge.jsonl `
  -Procedures .\data\phase_9_student_distillation\expanded\procedures.jsonl `
  -Prompts .\data\phase_9_student_distillation\expanded\prompts.jsonl `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -AllowUnreleasedTeacher
```

`-AllowUnreleasedTeacher` is required because the unmodified Qwen snapshot is not your formally accepted Phase 8 teacher. Here it is being used as an external diagnostic reference.

This should produce:

```text
artifacts\phase_9_qwen3_06b_grounded_20260806\
    teacher_candidates.jsonl
    teacher_candidates.review.md
    teacher_candidates.manifest.json
    knowledge.sqlite3
```

The expanded prompt corpus contains 36 prompts, unlike the six-row sample packet, so it is suitable for the research-readiness checks after review.

---

# 2. Human-review the Qwen outputs

Create the reviewed copy:

```powershell
Copy-Item `
  "$RunRoot\teacher_candidates.jsonl" `
  "$RunRoot\teacher_candidates.reviewed.jsonl"
```

Open the review guide:

```powershell
notepad "$RunRoot\teacher_candidates.review.md"
```

Open the JSONL file in your editor:

```powershell
code "$RunRoot\teacher_candidates.reviewed.jsonl"
```

For every approved row, change:

```json
"review_status": "pending_human_review"
```

to:

```json
"review_status": "accepted"
```

Then either select one generated candidate:

```json
"selected_candidate_index": 0
```

or provide a corrected answer manually:

```json
"approved_answer": "The corrected concise answer."
```

When `approved_answer` is provided, it overrides the selected candidate.

Reject or repair any answer that:

* uses information outside the evidence;
* fails to abstain when evidence is missing;
* follows instructions embedded inside retrieved evidence;
* mishandles conflicting evidence;
* produces incomplete or corrupted output.

Do not continue until the reviewed JSONL is complete.

---

# 3. Stage A–B: test Qwen and the untrained warm student

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Preflight `
  -TeacherModel $QwenBase `
  -ReviewedTeacherData $ReviewedData `
  -Knowledge .\data\phase_9_student_distillation\expanded\knowledge.jsonl `
  -Procedures .\data\phase_9_student_distillation\expanded\procedures.jsonl `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -AllowUnreleasedTeacher
```

This performs:

```text
Stage A: Qwen3-0.6B with exact Oracle evidence
Stage B: untrained warm_344m initialization with Oracle evidence
```

Inspect:

```powershell
notepad "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
```

Also inspect the raw summaries:

```text
$RunRoot\evaluation_teacher_oracle\summary.json
$RunRoot\evaluation_teacher_oracle\results.jsonl

$RunRoot\evaluation_warm_initialization_oracle\summary.json
$RunRoot\evaluation_warm_initialization_oracle\results.jsonl
```

The pipeline deliberately stops here when Qwen itself cannot demonstrate usable Oracle-RAG behavior. In that case, student training would not yet be meaningful.

---

# 4. Stage C: prove the training wiring with deliberate overfitting

Run this only after Preflight passes:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Overfit `
  -TeacherModel $QwenBase `
  -ReviewedTeacherData $ReviewedData `
  -Knowledge .\data\phase_9_student_distillation\expanded\knowledge.jsonl `
  -Procedures .\data\phase_9_student_distillation\expanded\procedures.jsonl `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

This trains the warm student on an 18-row deliberate overfit packet covering:

```text
6 supported-answer cases
4 missing-evidence cases
2 conflict cases
2 injection cases
2 procedures
2 social replies
```

It then reloads the saved adapter and evaluates it again.

Inspect:

```text
$RunRoot\overfit_probe\dataset_manifest.json
$RunRoot\overfit_probe\training\training_manifest.json
$RunRoot\overfit_probe\evaluation_initialization\results.jsonl
$RunRoot\overfit_probe\evaluation_trained_reloaded\results.jsonl
```

This stage must pass before a longer run. Failure here usually indicates a training or checkpoint problem rather than insufficient model capacity.

---

# 5. Stage D: run the warm 344M SFT sweep

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Sweep `
  -TeacherModel $QwenBase `
  -ReviewedTeacherData $ReviewedData `
  -Knowledge .\data\phase_9_student_distillation\expanded\knowledge.jsonl `
  -Procedures .\data\phase_9_student_distillation\expanded\procedures.jsonl `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

The sweep runs:

```text
5e-5 learning rate, 600 steps,  LoRA rank 16
5e-5 learning rate, 1200 steps, LoRA rank 16
1e-4 learning rate, 1200 steps, LoRA rank 16
```

Inspect the selection report:

```powershell
notepad "$RunRoot\warm_sft_sweep\SELECTION_REPORT.md"
```

The machine-readable selection is:

```text
$RunRoot\warm_sft_sweep\selection.json
```

The winner is selected according to Oracle-RAG improvement over the untrained warm initialization—not merely according to model size.

---

# 6. Stage E: evaluate the selected SFT model with Real RAG

Do this first **without KD**:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel $QwenBase `
  -ReviewedTeacherData $ReviewedData `
  -Knowledge .\data\phase_9_student_distillation\expanded\knowledge.jsonl `
  -Procedures .\data\phase_9_student_distillation\expanded\procedures.jsonl `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

This evaluates both Qwen and the selected SFT student under:

```text
closed_book
oracle_rag
real_rag
adversarial_rag
adversarial_passthrough_rag
```

Important outputs:

```text
$RunRoot\evaluation_teacher_full\summary.json
$RunRoot\evaluation_teacher_full\results.jsonl

$RunRoot\evaluation_selected_sft_full\summary.json
$RunRoot\evaluation_selected_sft_full\results.jsonl

reports\PHASE_9_STOP_GATE.md
```

The useful diagnosis is:

```text
Oracle-RAG poor
→ evidence-use training, prompting, decoding, or capacity problem

Oracle-RAG good but Real-RAG poor
→ retrieval, reranking, evidence selection, or calibration problem
```

---

# 7. Optional: progressive KD after SFT succeeds

Only run KD when the selected SFT student clearly improves over initialization.

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel $QwenBase `
  -ReviewedTeacherData $ReviewedData `
  -Knowledge .\data\phase_9_student_distillation\expanded\knowledge.jsonl `
  -Procedures .\data\phase_9_student_distillation\expanded\procedures.jsonl `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -Teacher4Bit `
  -RunKD `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

Here, KD starts from the selected SFT adapter rather than from an untrained initialization.

`-Teacher4Bit` applies to KD training only. It requires CUDA and `bitsandbytes`. Because Qwen3-0.6B is relatively small, this fallback command can be used if Windows reports a `bitsandbytes` error:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel $QwenBase `
  -ReviewedTeacherData $ReviewedData `
  -Knowledge .\data\phase_9_student_distillation\expanded\knowledge.jsonl `
  -Procedures .\data\phase_9_student_distillation\expanded\procedures.jsonl `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens 384 `
  -UseLora `
  -RunKD `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

That loads the Qwen teacher without 4-bit quantization.

## Recommended order

```text
Bootstrap with Qwen
        ↓
Human review
        ↓
Preflight
        ↓
Overfit
        ↓
SFT sweep
        ↓
Full Real-RAG evaluation
        ↓
Progressive KD only after SFT works
        ↓
Narrow 170M/130M/100M comparisons later
```

Do not use the previous weak-teacher output root, do not add `-RunKD` to the first Full run, and do not return to the narrow random students until the warm inherited student proves that the evidence-use path works.

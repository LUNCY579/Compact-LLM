# Phase 9 Student Distillation / Grounded Recovery — Complete Command Record (v1.7.7)

**Purpose:** Personal execution record for running the full Phase 9 grounded-recovery ladder in the correct order, including the v1.7.7 SFT-selection gate repair, all prerequisite PowerShell variables, verification commands, Full continuation, optional KD, and final Phase 9 handoff.

**Current active run:**

```text
artifacts\phase_9_grounded_recovery_v1_7_5_qwen\
```

**Pipeline revision:**

```text
phase9_large_corrective_curriculum_v1_7_5
```

**Selection policy revision:**

```text
phase9_sft_selection_gate_v1_7_7
```

**Current known Stage-D winner:**

```text
lr1e4_s1200
Oracle capability score: 1.0
Initialization score:     0.2483452333967206
Improvement:              +0.7516547666032793
Valid:                    true
Candidate issues:         []
```

The Stage-B warm initialization has decode-integrity warnings, but under v1.7.7 those warnings are diagnostic-only and do **not** invalidate trained SFT candidates.

> **Important current-run rule:** Preflight, Overfit, and Sweep are already complete for `phase_9_grounded_recovery_v1_7_5_qwen`. For this run, do **not** retrain them. Use the **Current continuation path** near the end of this document.

---

# 0. Repository / shell prerequisites

Open PowerShell.

```powershell
cd C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2
```

Activate the same Python environment used for training:

```powershell
.\.venv\Scripts\Activate.ps1
```

Confirm Python is coming from the virtual environment:

```powershell
python --version
Get-Command python
```

Optional CUDA sanity check:

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

---

# 1. Define all PowerShell variables

## 1.1 Repository root

```powershell
$RepoRoot = 'C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2'
Set-Location $RepoRoot
```

## 1.2 Qwen3-0.6B teacher/base checkpoint

Use the same teacher checkpoint used by the original Phase 9 run.

```powershell
$QwenBase = 'C:\Users\clclc\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca'
```

For readability, use the same path as the teacher variable:

```powershell
$TeacherModel = $QwenBase
```

## 1.3 Current v1.7.5 run root

```powershell
$RunRoot = '.\artifacts\phase_9_grounded_recovery_v1_7_5_qwen'
```

## 1.4 Shared Phase 9 settings

```powershell
$MaximumEvidenceTokens = 384
$Phase9Runner = '.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1'
$ReselectScript = '.\scripts\phase_9_student_distillation\reselect_existing_grounded_sft.py'
```

## 1.5 Verify all critical paths

```powershell
Write-Host "`n=== REPOSITORY ==="
Test-Path $RepoRoot

Write-Host "`n=== TEACHER / QWEN BASE ==="
$TeacherModel
Test-Path $TeacherModel

Write-Host "`n=== PHASE 9 RUNNER ==="
$Phase9Runner
Test-Path $Phase9Runner

Write-Host "`n=== V1.7.7 RESELECT UTILITY ==="
$ReselectScript
Test-Path $ReselectScript

Write-Host "`n=== CURRENT RUN ROOT ==="
$RunRoot
Test-Path $RunRoot
```

For an existing run, these should normally all be `True`.

---

# 2. Canonical Phase 9 ladder

The intended order is:

```text
Stage A/B — Preflight
        ↓
Stage C — Tiny overfit wiring proof
        ↓
Stage D — Full Oracle SFT sweep
        ↓
Stage D selection
        ↓
Stage E — Real / adversarial RAG + final DEV selection
        ↓
Sealed TEST evaluation
        ↓
Human Phase-9 review / gate
        ↓
Materialize accepted student
        ↓
Phase 9B and/or Phase 10
```

Do not skip directly to structural reduction before the Phase 9 student has been accepted.

---

# 3. Stage A/B — Preflight

## Purpose

Preflight establishes:

- **Stage A — Teacher Oracle:** whether the teacher can correctly use the evidence.
- **Stage B — Warm initialization:** the weak untrained student baseline used for comparison.

Stage A is the important continuation prerequisite. Stage B is a baseline and may be weak.

## Command

```powershell
& $Phase9Runner `
  -Mode Preflight `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher
```

If the original teacher configuration used 4-bit teacher loading, preserve that configuration by adding:

```text
-Teacher4Bit
```

Do not add `-Teacher4Bit` merely because it exists; use it only when it matches the experiment configuration being continued.

## Inspect the ladder

```powershell
Get-Content "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
```

Continue only if **Stage A Teacher Oracle is PASS**.

A Stage-B decode-integrity failure on the untrained warm initialization does not, by itself, mean training must stop.

---

# 4. Stage C — Tiny overfit / wiring proof

## Purpose

This verifies that the model can:

- train,
- save,
- reload,
- and measurably improve on a small controlled probe.

## Command

```powershell
& $Phase9Runner `
  -Mode Overfit `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## Inspect

```powershell
Get-Content "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
```

Stage C should PASS before treating a later failed Sweep as a training-wiring problem.

---

# 5. Stage D — Full Oracle SFT sweep

## Purpose

Train the configured warm-SFT candidates and Oracle-evaluate each one.

## Command

```powershell
& $Phase9Runner `
  -Mode Sweep `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## Expected candidate directories for the current run

```text
$RunRoot\warm_sft_sweep\lr5e5_s600\
$RunRoot\warm_sft_sweep\lr5e5_s1200\
$RunRoot\warm_sft_sweep\lr1e4_s1200\
```

The current winning candidate is:

```text
$RunRoot\warm_sft_sweep\lr1e4_s1200\
```

Its important artifacts are:

```text
training\adapter\
training\training_manifest.json
evaluation_oracle\summary.json
evaluation_oracle\results.jsonl
```

---

# 6. Stage D selection — v1.7.7 gate repair / explicit reselection

## Why this exists

Older selection behavior could poison every SFT candidate when the **Stage-B initialization** had decode/prompt integrity failures.

v1.7.7 changes the policy:

```text
Stage-B decode/prompt integrity
        ↓
warning only

Candidate's own integrity / coverage / behavior
        ↓
still required

Initialization behavior coverage
        ↓
still required for a meaningful reference

Positive capability improvement over initialization
        ↓
still required
```

## Explicit reselection command

After Sweep artifacts exist:

```powershell
python $ReselectScript `
  --run-root $RunRoot
```

This performs **selection only**. It does not retrain the candidates.

## Verify selection

```powershell
$Selection = Get-Content "$RunRoot\warm_sft_sweep\selection.json" -Raw | ConvertFrom-Json
```

Print the important fields:

```powershell
$Selection.status
$Selection.selection_policy_revision
$Selection.selected.name
$Selection.selected.capability_score
$Selection.selected.initialization_score
$Selection.selected.improvement
$Selection.selected.valid
$Selection.selected.issues
$Selection.selected.adapter
```

For the current run, expected output is:

```text
SFT_CANDIDATE_SELECTED
phase9_sft_selection_gate_v1_7_7
lr1e4_s1200
1
0.2483452333967206
0.7516547666032793
True
<empty issues array>
artifacts\phase_9_grounded_recovery_v1_7_5_qwen\warm_sft_sweep\lr1e4_s1200\training\adapter
```

Check how Stage-B integrity was classified:

```powershell
$Selection.initialization_integrity_warnings
$Selection.initialization_blocking_issues
$Selection.initialization_integrity_is_selection_blocking
```

For the current run, the decode-integrity issues should appear under warnings while blocking issues remain empty.

## Human-readable report

```powershell
Get-Content "$RunRoot\warm_sft_sweep\SELECTION_REPORT.md"
```

---

# 7. Stage E / Full — run SFT candidate through real RAG and complete Phase 9

## Important rule

Run **Full without KD first**.

This allows the selected SFT candidate to prove itself under real/adversarial retrieved evidence before adding another training branch.

## Command

```powershell
& $Phase9Runner `
  -Mode Full `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

If the same original experiment configuration used a 4-bit teacher, use:

```powershell
& $Phase9Runner `
  -Mode Full `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -Teacher4Bit `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

Do not switch teacher precision casually when continuing the same experiment.

## What Full should reuse

With `-ReuseExisting`, the runner should reuse completed expensive artifacts where valid:

```text
Stage A Teacher Oracle
Stage B warm initialization evaluation
Stage C overfit
Stage D candidate training
Stage D Oracle evaluations
Stage D selected SFT adapter
```

It should then continue into the remaining Phase 9 evaluation/selection path rather than retraining the three finished SFT candidates.

---

# 8. Inspect Full / Stage E results

## 8.1 Ladder

```powershell
Get-Content "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
```

Stage D should no longer remain `NOT_RUN`.

## 8.2 Selected SFT on full DEV / real RAG

```powershell
Get-Content "$RunRoot\evaluation_selected_sft_dev_full\summary.json"
```

This is more meaningful than the Stage-D Oracle score because it evaluates performance with actual retrieved evidence conditions rather than exact Oracle evidence only.

## 8.3 Final DEV selection

```powershell
Get-Content "$RunRoot\FINAL_DEV_SELECTION.md"
```

Machine-readable:

```powershell
$Final = Get-Content "$RunRoot\final_dev_selection.json" -Raw | ConvertFrom-Json
$Final.status
$Final.selected.name
$Final.selected.score
$Final.selected.adapter
```

If KD has not been run, the final candidate may be labeled simply:

```text
sft
```

That label refers to the selected SFT branch; the internal Stage-D winner is still `lr1e4_s1200`.

## 8.4 Sealed TEST result

```powershell
Get-Content "$RunRoot\evaluation_selected_final_test\summary.json"
```

## 8.5 Phase 9 stop-gate / assessment reports

```powershell
Get-Content .\reports\PHASE_9_STOP_GATE.md
```

```powershell
Get-Content .\reports\PHASE_9_ASSESSMENT.json
```

---

# 9. Sealed-test discipline

The intended sequence is:

```text
DEV
  ↓
select one candidate
  ↓
freeze candidate
  ↓
sealed TEST
```

After reading sealed TEST results:

- do not tune the same candidate against specific sealed-test examples;
- do not repeatedly change hyperparameters based on sealed-test failures;
- use DEV/new governed training data for further iteration.

The sealed set is evidence, not a tuning set.

---

# 10. Optional KD branch — only after Full SFT succeeds

The original execution note places KD **after** the SFT Full run.

Use KD only if you intentionally want to evaluate an additional distillation branch.

## Command

```powershell
& $Phase9Runner `
  -Mode Full `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -Teacher4Bit `
  -RunKD `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

Important:

- `-RunKD` is optional.
- `-Teacher4Bit` should match the KD/teacher setup required by the experiment.
- Do not discard the successful SFT adapter simply because KD is being explored.
- Treat SFT and KD as candidates to compare under the same governed DEV selection process.

After KD completes, inspect the updated:

```powershell
Get-Content "$RunRoot\FINAL_DEV_SELECTION.md"
```

and:

```powershell
$Final = Get-Content "$RunRoot\final_dev_selection.json" -Raw | ConvertFrom-Json
$Final.selected.name
$Final.selected.adapter
```

---

# 11. Human Phase-9 gate

After the final candidate has completed the governed DEV/TEST path, review the Phase 9 reports.

```powershell
Get-Content .\reports\PHASE_9_STOP_GATE.md
Get-Content .\reports\PHASE_9_ASSESSMENT.json
```

The repository must not self-approve the human gate.

After you manually record an explicit Phase-9 `PASS` according to the repository's gate procedure, verify it with:

```powershell
python .\scripts\common\check_human_gate.py --phase 9
```

Expected:

```text
Phase 9 human gate: PASS
```

Do not materialize the accepted Phase 9 model until this human gate is actually PASS.

---

# 12. Materialize the accepted Phase-9 student

Read the final candidate identity first:

```powershell
$Final = Get-Content "$RunRoot\final_dev_selection.json" -Raw | ConvertFrom-Json
$AcceptedCandidate = $Final.selected.name
$AcceptedCandidate
```

Then finalize the approved candidate:

```powershell
python .\scripts\phase_10_student_architecture\finalize_phase_9_selection.py `
  --candidate $AcceptedCandidate `
  --role intermediate_teacher `
  --reason "Approved Phase 9 grounded-recovery student after governed DEV selection, sealed TEST review, and human Phase-9 PASS."
```

Expected durable handoff artifacts include:

```text
artifacts\phase_9_student_distillation\accepted_student\
artifacts\references\phase_9_student_reference.json
```

This is the model reference to use for later governed phases instead of relying on a temporary sweep directory.

---

# 13. What comes after Phase 9

Once Phase 9 has an accepted student:

```text
Phase 9B
Memory reconstruction specialization / evaluation

and/or

Phase 10
Student architecture reduction / depth-width reduction
```

Do not treat a perfect Stage-D Oracle score as permission to skip the remaining Phase 9 evaluation and human gate.

For the current run:

```text
lr1e4_s1200 = perfect Stage-D Oracle candidate
```

not automatically:

```text
final production student
```

Stage E, sealed TEST, and human review provide the remaining evidence.

---

# 14. CURRENT RUN — exact continuation path from where you are now

For the current `phase_9_grounded_recovery_v1_7_5_qwen` run:

```text
Stage A: complete
Stage B: complete baseline; decode-integrity warnings
Stage C: PASS
Stage D training: complete
Stage D Oracle evaluations: complete
Stage D winner: lr1e4_s1200
v1.7.7 reselection: complete / SFT_CANDIDATE_SELECTED
NEXT: Full
```

Therefore, you do **not** need to run Preflight, Overfit, or Sweep again.

## 14.1 Recreate variables in a new PowerShell session

```powershell
cd C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2

.\.venv\Scripts\Activate.ps1

$QwenBase = 'C:\Users\clclc\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca'
$TeacherModel = $QwenBase
$RunRoot = '.\artifacts\phase_9_grounded_recovery_v1_7_5_qwen'
$MaximumEvidenceTokens = 384
$Phase9Runner = '.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1'
$ReselectScript = '.\scripts\phase_9_student_distillation\reselect_existing_grounded_sft.py'
```

## 14.2 Verify the already-selected winner

```powershell
$Selection = Get-Content "$RunRoot\warm_sft_sweep\selection.json" -Raw | ConvertFrom-Json

$Selection.status
$Selection.selected.name
$Selection.selected.capability_score
$Selection.selected.improvement
$Selection.selected.valid
$Selection.selected.adapter
```

Expected:

```text
SFT_CANDIDATE_SELECTED
lr1e4_s1200
1
0.7516547666032793
True
artifacts\phase_9_grounded_recovery_v1_7_5_qwen\warm_sft_sweep\lr1e4_s1200\training\adapter
```

## 14.3 Run Full now

```powershell
& $Phase9Runner `
  -Mode Full `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

If and only if the original teacher configuration for this run used 4-bit teacher loading:

```powershell
& $Phase9Runner `
  -Mode Full `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -Teacher4Bit `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## 14.4 After Full finishes

Run in this order:

```powershell
Get-Content "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
```

```powershell
Get-Content "$RunRoot\evaluation_selected_sft_dev_full\summary.json"
```

```powershell
Get-Content "$RunRoot\FINAL_DEV_SELECTION.md"
```

```powershell
Get-Content "$RunRoot\evaluation_selected_final_test\summary.json"
```

```powershell
Get-Content .\reports\PHASE_9_STOP_GATE.md
```

```powershell
Get-Content .\reports\PHASE_9_ASSESSMENT.json
```

Then make the human Phase-9 decision.

---

# 15. Fresh-run copy/paste order for future experiments

For a **new** Phase 9 run with a new `$RunRoot`, the minimal order is:

## Step 1 — Setup

```powershell
cd C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2
.\.venv\Scripts\Activate.ps1

$QwenBase = 'C:\Users\clclc\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca'
$TeacherModel = $QwenBase
$RunRoot = '.\artifacts\YOUR_NEW_PHASE_9_RUN'
$MaximumEvidenceTokens = 384
$Phase9Runner = '.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1'
$ReselectScript = '.\scripts\phase_9_student_distillation\reselect_existing_grounded_sft.py'
```

## Step 2 — Preflight

```powershell
& $Phase9Runner `
  -Mode Preflight `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher
```

## Step 3 — Inspect Stage A

```powershell
Get-Content "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
```

## Step 4 — Overfit

```powershell
& $Phase9Runner `
  -Mode Overfit `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## Step 5 — Sweep

```powershell
& $Phase9Runner `
  -Mode Sweep `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## Step 6 — Explicit selection / reselection

```powershell
python $ReselectScript `
  --run-root $RunRoot
```

## Step 7 — Verify winner

```powershell
$Selection = Get-Content "$RunRoot\warm_sft_sweep\selection.json" -Raw | ConvertFrom-Json
$Selection.status
$Selection.selected.name
$Selection.selected.capability_score
$Selection.selected.valid
$Selection.selected.adapter
```

## Step 8 — Full, no KD first

```powershell
& $Phase9Runner `
  -Mode Full `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## Step 9 — Inspect Full results

```powershell
Get-Content "$RunRoot\GROUNDED_RECOVERY_LADDER.md"
Get-Content "$RunRoot\evaluation_selected_sft_dev_full\summary.json"
Get-Content "$RunRoot\FINAL_DEV_SELECTION.md"
Get-Content "$RunRoot\evaluation_selected_final_test\summary.json"
Get-Content .\reports\PHASE_9_STOP_GATE.md
Get-Content .\reports\PHASE_9_ASSESSMENT.json
```

## Step 10 — Optional KD

```powershell
& $Phase9Runner `
  -Mode Full `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -Teacher4Bit `
  -RunKD `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

## Step 11 — Human gate

```powershell
python .\scripts\common\check_human_gate.py --phase 9
```

Proceed only after it reports Phase 9 human gate `PASS`.

## Step 12 — Finalize selected student

```powershell
$Final = Get-Content "$RunRoot\final_dev_selection.json" -Raw | ConvertFrom-Json
$AcceptedCandidate = $Final.selected.name

python .\scripts\phase_10_student_architecture\finalize_phase_9_selection.py `
  --candidate $AcceptedCandidate `
  --role intermediate_teacher `
  --reason "Approved Phase 9 grounded-recovery student after governed DEV selection, sealed TEST review, and human Phase-9 PASS."
```

---

# 16. Important artifact locations

## Stage-B initialization summary

```text
$RunRoot\evaluation_warm_initialization_oracle\summary.json
```

## Stage-D sweep

```text
$RunRoot\warm_sft_sweep\
```

## Current Stage-D winning adapter

```text
$RunRoot\warm_sft_sweep\lr1e4_s1200\training\adapter\
```

## Stage-D selection JSON

```text
$RunRoot\warm_sft_sweep\selection.json
```

## Stage-D selection report

```text
$RunRoot\warm_sft_sweep\SELECTION_REPORT.md
```

## Real-RAG / Full SFT evaluation

```text
$RunRoot\evaluation_selected_sft_dev_full\
```

## Final DEV selection

```text
$RunRoot\final_dev_selection.json
$RunRoot\FINAL_DEV_SELECTION.md
```

## Sealed TEST evaluation

```text
$RunRoot\evaluation_selected_final_test\
```

## Phase 9 reports

```text
.\reports\PHASE_9_STOP_GATE.md
.\reports\PHASE_9_ASSESSMENT.json
```

## Accepted model handoff

```text
.\artifacts\phase_9_student_distillation\accepted_student\
.\artifacts\references\phase_9_student_reference.json
```

---

# 17. Cleanup / preservation rules

For the current successful v1.7.5 run, preserve:

```text
artifacts\phase_9_grounded_recovery_v1_7_5_qwen\
```

Especially preserve:

```text
warm_sft_sweep\lr1e4_s1200\
```

Do not delete its:

```text
training\adapter\
evaluation_oracle\summary.json
evaluation_oracle\results.jsonl
```

Old failed experimental roots may be removed only after confirming they are not referenced by any report, selected-candidate manifest, or later phase.

---

# 18. Short version — current next command

Your Stage-D winner has already been successfully selected under v1.7.7.

Therefore the next expensive command for the current run is:

```powershell
& $Phase9Runner `
  -Mode Full `
  -TeacherModel $TeacherModel `
  -OutputRoot $RunRoot `
  -MaximumEvidenceTokens $MaximumEvidenceTokens `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

After that command finishes, inspect the Stage-E / final DEV / sealed TEST reports before making the Phase-9 human decision.

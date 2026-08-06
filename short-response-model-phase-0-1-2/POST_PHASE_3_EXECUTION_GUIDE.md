# Post-Phase-3 runnable execution guide

These commands perform the work. They are not instruction-only wrappers. Run them from the repository root in an activated Python 3.11 virtual environment.

## 0. Unblock downloaded scripts and install dependencies

After extracting the ZIP, remove Windows' downloaded-file mark from the repository scripts. This avoids the "not digitally signed" error without changing the machine-wide execution policy:

```powershell
Get-ChildItem -Path . -Recurse -Filter *.ps1 | Unblock-File
```

Install the correct CUDA-enabled PyTorch wheel first, then:

```powershell
.\install_post_phase_3.ps1
.\verify_post_phase_3.ps1 -VerifyImports
```

The verification command compiles the source, runs the regression suite, checks the integrated Phase 2B and Phase 4–11 runner references, and verifies the installed imports.

When your organization enforces a stricter PowerShell policy, run the commands in a process opened with the policy approved by your administrator rather than weakening the permanent machine policy.

## 1. Phase 4 — train/evaluate soup ingredients and materialize checkpoint

```powershell
.\scripts\phase_4_model_soups\run_phase_4.ps1 `
  -ModelId "Qwen/Qwen3-0.6B" `
  -PrimaryAdapter .\artifacts\phase_3_dpo\adapter
```

Review and approve `reports/PHASE_4_STOP_GATE.md`.

For an approved reduced branch, use the isolated branch command instead:

```powershell
.\scripts\phase_4_model_soups\run_phase_4.ps1 -BranchName balanced
```

## 2. Phase 4B — create and evaluate permanent reduction candidates

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 -Mode Pilot
```

For the same reduced branch:

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 -BranchName balanced -Mode Pilot
```

Approve Phase 4B, then finalize the selected candidate using the exact candidate name from `candidate_registry.json`.

```powershell
python .\scripts\phase_4b_structural_reduction\finalize_reference_selection.py `
  --registry .\artifacts\phase_4b_structural_reduction\candidate_registry.json `
  --candidate dense_reference `
  --reason "Approved deployment reference."
```

## 3. Phase 5 — retrieval, context budgets, and prefix reuse

```powershell
.\scripts\phase_5_context_system\run_phase_5.ps1
```

## 4. Phase 6 — activation traces and masked generation frontier

```powershell
.\scripts\phase_6_sparse_simulation\run_phase_6.ps1
```

## 5. Phase 7 — measured dense runtime, plus routers when viable

```powershell
.\scripts\phase_7_sparse_runtime\run_phase_7.ps1
```

When Phase 6 selected `DENSE_ONLY`, the runner skips router training and measures the dense baseline. Approve either `DENSE_RUNTIME` or `SPARSE_FEATURE_FLAG`; dense deployment is an acceptable production decision.

## 6. Phase 8 — sealed evaluation, audit, and package

```powershell
.\scripts\phase_8_release_candidate\run_phase_8.ps1 -PackageMode manifest
```

Each runner fails on missing prior gates or inputs, records outputs under `artifacts/`, and creates a pending report under `reports/`. It never fabricates benchmark results or self-approves a human gate.


## 7. Phase 9 — appended tiny-student/RAG distillation

Phase 9 begins after Phase 8 names the exact accepted teacher. Bootstrap creates a student initialization, low-memory knowledge index, and teacher-review packet:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Bootstrap `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model
```

After human-reviewing the generated candidates:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -MaxSteps 200
```

The single-student pilot compares ordinary RAG SFT with same-initialization completion-only logit KD, evaluates closed/oracle/real/adversarial RAG, and writes a pending Phase 9 human gate.

To compare the warm 344M depth-copy student with the 100M/130M/170M narrow same-tokenizer research controls in one gate, use:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -StudentSuite Hybrid `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -MaxSteps 200 `
  -ReuseExisting
```

Hybrid assessment keys are profile-qualified, such as `sft_warm_344m`, `kd_warm_344m`, and `kd_narrow_130m`. Narrow profiles are research controls and are not automatically production-eligible. See `STUDENT_DISTILLATION_PIPELINE.md` for the complete workflow.

## 8. Phase 10 — student architecture reduction and challengers

After Phase 9 human approval, materialize its exact candidate and run the pilot:

```powershell
python .\scripts\phase_10_student_architecture\finalize_phase_9_selection.py `
  --candidate kd_warm_344m `
  --role intermediate_teacher `
  --reason "Approved Phase 9 warm KD student for architecture reduction."

.\scripts\phase_10_student_architecture\run_phase_10.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model
```

Full mode adds the intermediate/small/deployment target bands and may add a pretrained compact challenger. Use the exact Phase 9 candidate key present in `reports/PHASE_9_ASSESSMENT.json`; legacy single-suite keys such as `kd` remain supported. Review `reports/PHASE_10_STOP_GATE.md`, then use `finalize_phase_10_selection.py` with the exact Phase 10 registry name.

## 9. Phase 11 — production RAG runtime and release

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 `
  -Mode Full `
  -QuantizationBackend dense_copy
```

Repeat with NF4, INT8, or GGUF only when those runtimes are available. Phase 11 measures the complete route/retrieve/gate/generate/ground path and total process memory. After human approval, package the exact registry candidate:

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 `
  -Mode Package `
  -PackageCandidate <registry-name> `
  -PackageMode copy
```

See `PHASE_10_11_WORKFLOW.md` for target bands, compact challenger, persistent JSONL runtime, GGUF requirements, and valid negative outcomes.

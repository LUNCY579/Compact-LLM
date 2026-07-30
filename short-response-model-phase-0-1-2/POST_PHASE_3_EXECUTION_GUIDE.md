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

The verification command compiles the source, runs the regression suite, checks the integrated Phase 2B and Phase 4–8 runner references, and verifies the installed imports.

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

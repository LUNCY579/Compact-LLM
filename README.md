
Start the environment: 
```
.venv/Scripts/Activate.ps1
```

Zip the folder without artifacts or other non-trivial documents
```
tar.exe -a -c -f "..\short-response-model-phase-0-1-2_without_artifacts_$(Get-Date -Format yyyyMMdd_HHmmss).zip" `
  --exclude="./artifacts" `
  --exclude="./playground_artifacts" `
  .
```

Phase 2B command:
```
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -ProbeFile .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  -OutputRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -QualityProfile balanced `
  -ProbeLimit 150 `
  -MaxDrops 10 `
  -RecoveryTopK 6 `
  -RecoverySteps 50 `
  -QLoRA `
  -GradientCheckpointing
```
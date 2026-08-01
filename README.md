
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
python .\scripts\phase_2_layer_selection\rescore_phase_2b_artifacts.py `
  --run-root .\artifacts\phase_2b_quality_v2_thorough_20260731 `
  --benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  --in-place `
  --prepare-resume
```
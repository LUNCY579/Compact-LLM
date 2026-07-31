
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


# Windows 11 Quick Start

```powershell
Set-Location C:\path\to\rineya-context-harness-v0.1.0
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[api,dev]"
.\scripts\check.ps1
```

## Local Rineya endpoint

1. Start the chosen llama.cpp/Ollama/OpenAI-compatible runtime.
2. Copy `configs/local-rineya.example.yaml`.
3. Set the exact model name and base URL.
4. Run `rineya-harness doctor`.
5. Run a direct prompt before enabling search.
6. Run the structured-output qualification suite.

## Windows-specific checks

- paths with spaces;
- PowerShell execution policy;
- SQLite WAL file cleanup;
- Ctrl+C cancellation;
- UTF-8 console output;
- local firewall behavior;
- sleep/resume behavior;
- model-server restart and timeout recovery.

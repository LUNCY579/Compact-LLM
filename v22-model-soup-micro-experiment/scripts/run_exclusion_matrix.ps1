param(
    [switch]$ReuseExisting,
    [switch]$Full,
    [int]$Limit = 0,
    [string[]]$Mode = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:CUDA_VISIBLE_DEVICES = ""
$Python = Join-Path (Split-Path (Resolve-Path (Join-Path $Root "..\short-response-model-phase-0-1-2")).Path -Parent) ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$argsList = @(".\scripts\run_exclusion_matrix.py")
if ($ReuseExisting) { $argsList += "--ReuseExisting" }
if ($Full) { $argsList += "--full" }
if ($Limit -gt 0) { $argsList += @("--limit", "$Limit") }
foreach ($m in $Mode) { $argsList += @("--mode", $m) }

& $Python @argsList
if ($LASTEXITCODE -ne 0) { throw "exclusion matrix failed with exit $LASTEXITCODE" }
Write-Host "Report: artifacts\exclusion_matrix\PROGRESS_REPORT.md"

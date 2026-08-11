param(
    [int]$Limit = 24,
    [switch]$SkipBuild,
    [switch]$ForceRebuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:CUDA_VISIBLE_DEVICES = ""
$Official = (Resolve-Path (Join-Path $Root "..\short-response-model-phase-0-1-2")).Path
$env:PYTHONPATH = $Official
$Python = Join-Path (Split-Path $Official -Parent) ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

if (-not $SkipBuild) {
    $build = @($Python, ".\scripts\build_v22_soups.py")
    if ($ForceRebuild) { $build += "--force" }
    & $build[0] $build[1..($build.Length-1)]
    if ($LASTEXITCODE -ne 0) { throw "soup build failed" }
}

& $Python .\scripts\smoke_eval_cpu.py --limit $Limit
if ($LASTEXITCODE -ne 0) { throw "smoke eval failed" }
Write-Host "Done. See artifacts\smoke_eval\SMOKE_COMPARISON.md"

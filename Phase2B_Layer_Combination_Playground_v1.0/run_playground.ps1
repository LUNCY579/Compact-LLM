param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [ValidateSet("core", "layer26", "all")]
    [string]$Suite = "core",
    [int]$CaseLimit = 150,
    [switch]$IncludeExistingEndpoints,
    [switch]$IncludeRecoveredExtensions,
    [switch]$LeaveOneInAblation,
    [switch]$Resume,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
$Script = Join-Path $PSScriptRoot "phase2b_layer_combination_playground.py"

$Arguments = @(
    $Script,
    "--repo-root", $RepoRoot,
    "--suite", $Suite,
    "--case-limit", $CaseLimit,
    "--require-cuda"
)
if ($IncludeExistingEndpoints) { $Arguments += "--include-existing-endpoints" }
if ($IncludeRecoveredExtensions) { $Arguments += "--include-recovered-extensions" }
if ($LeaveOneInAblation) { $Arguments += "--leave-one-in-ablation" }
if ($Resume) { $Arguments += "--resume" }
if ($PlanOnly) { $Arguments += "--plan-only" }

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Layer-combination playground failed with exit code $LASTEXITCODE."
}

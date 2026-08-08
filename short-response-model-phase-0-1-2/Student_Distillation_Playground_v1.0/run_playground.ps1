param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Adapter = "",
    [string]$Dataset = ".\Student_Distillation_Playground_v1.0\datasets\showcase_suite.jsonl",
    [string]$OutputDir = ".\playground_artifacts\student_distillation_showcase",
    [ValidateSet("all", "logic", "evidence", "memory")][string]$TaskType = "all",
    [ValidateSet("direct", "two_pass")][string]$MemoryMode = "direct",
    [int]$Limit = 0,
    [switch]$VideoMode,
    [switch]$ShowContext,
    [double]$PauseSeconds = 0,
    [switch]$RequireCuda
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
    $python = ".\.venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { $python = "python" }

    $argsList = @(
        ".\Student_Distillation_Playground_v1.0\student_playground.py",
        "--model", $Model,
        "--dataset", $Dataset,
        "--output-dir", $OutputDir,
        "--task-type", $TaskType,
        "--memory-mode", $MemoryMode
    )
    if ($Adapter) { $argsList += @("--adapter", $Adapter) }
    if ($Limit -gt 0) { $argsList += @("--limit", "$Limit") }
    if ($VideoMode) { $argsList += "--video-mode" }
    if ($ShowContext) { $argsList += "--show-context" }
    if ($PauseSeconds -gt 0) { $argsList += @("--pause-seconds", "$PauseSeconds") }
    if ($RequireCuda) { $argsList += "--require-cuda" }

    & $python @argsList
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

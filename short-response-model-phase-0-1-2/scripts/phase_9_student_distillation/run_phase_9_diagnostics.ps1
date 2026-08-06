<#
.SYNOPSIS
Audit an existing Phase 9 artifact tree without retraining any model.

.DESCRIPTION
Discovers evaluation directories containing summary.json and results.jsonl,
identifies evaluation_teacher_v2 as the teacher, and writes a raw-output,
prompt-integrity, decode-integrity, and functional diagnosis report.
#>
param(
    [string]$ArtifactRoot = ".\artifacts\phase_9_student_distillation",
    [string]$Benchmark = "",
    [string]$OutputPrefix = "",
    [switch]$RequireUsableTeacher
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

if (-not (Test-Path $ArtifactRoot)) { throw "Artifact root does not exist: $ArtifactRoot" }
if ([string]::IsNullOrWhiteSpace($Benchmark)) {
    $candidate = Join-Path $ArtifactRoot "dataset\test.jsonl"
    if (-not (Test-Path $candidate)) { throw "Supply -Benchmark; auto-discovery failed at $candidate" }
    $Benchmark = $candidate
}
if (-not (Test-Path $Benchmark)) { throw "Benchmark does not exist: $Benchmark" }
if ([string]::IsNullOrWhiteSpace($OutputPrefix)) {
    $OutputPrefix = Join-Path $ArtifactRoot "phase_9_diagnostic_audit"
}

$evaluationDirs = Get-ChildItem -Path $ArtifactRoot -Directory -Recurse | Where-Object {
    (Test-Path (Join-Path $_.FullName "summary.json")) -and
    (Test-Path (Join-Path $_.FullName "results.jsonl"))
}
if (-not $evaluationDirs) { throw "No evaluation directories with summary.json and results.jsonl were found under $ArtifactRoot" }

$teacher = $evaluationDirs | Where-Object { $_.Name -eq "evaluation_teacher_v2" } | Select-Object -First 1
if ($null -eq $teacher) { throw "Could not find evaluation_teacher_v2 under $ArtifactRoot" }

$args = @(
    ".\scripts\phase_9_student_distillation\audit_phase_9_results.py",
    "--benchmark", $Benchmark,
    "--teacher-name", "teacher",
    "--evaluation", "teacher=$($teacher.FullName)",
    "--json-out", "$OutputPrefix.json",
    "--out", "$OutputPrefix.md"
)

$used = @($teacher.FullName)
foreach ($directory in ($evaluationDirs | Sort-Object FullName)) {
    if ($used -contains $directory.FullName) { continue }
    $relative = [System.IO.Path]::GetRelativePath((Resolve-Path $ArtifactRoot).Path, $directory.FullName)
    $name = ($relative -replace '[^A-Za-z0-9_]+', '_').Trim('_')
    if ([string]::IsNullOrWhiteSpace($name)) { $name = "candidate" }
    $args += @("--evaluation", "$name=$($directory.FullName)")
}
if ($RequireUsableTeacher) { $args += "--require-usable-teacher" }

& python @args
if ($LASTEXITCODE -ne 0) { throw "Phase 9 diagnostic audit failed with exit code $LASTEXITCODE" }
Write-Host "Diagnostic JSON: $OutputPrefix.json"
Write-Host "Diagnostic report: $OutputPrefix.md"

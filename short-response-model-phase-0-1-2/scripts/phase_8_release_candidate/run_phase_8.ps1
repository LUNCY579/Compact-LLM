<#
.SYNOPSIS
Run sealed evaluation, collect existing ablation evidence, build the final audit,
and create a release-candidate package.

.DESCRIPTION
Unlike the previous runner, this script performs the release work instead of
printing a checklist. It evaluates the exact Phase 4B-selected checkpoint on the
sealed split, checks all prior human gates, reads the Phase 7 deployment mode,
hashes the selected artifacts, creates a candidate package, and writes the final
pending human release gate.
#>
param(
    [string]$ModelDir = "",
    [string]$SealedData = ".\data\sealed\sealed.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_8_release",
    [int]$MaxNewTokens = 32,
    [ValidateSet("manifest", "copy")][string]$PackageMode = "manifest",
    [switch]$ReuseExisting
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

Invoke-PythonStep "[1/6] Checking the Phase 7 human gate..." @(
    ".\scripts\common\check_human_gate.py", "--phase", "7"
)
if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = python .\scripts\common\resolve_reference_model.py
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve the selected Phase 4B model." }
}
if (-not (Test-Path $ModelDir)) { throw "Phase 8 requires a local selected checkpoint: $ModelDir" }
if (-not (Test-Path $SealedData)) { throw "Missing sealed evaluation data: $SealedData" }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$SealedEval = Join-Path $OutputRoot "sealed_evaluation"
$SealedSummary = Join-Path $SealedEval "summary.json"
if (-not ($ReuseExisting -and (Test-Path $SealedSummary))) {
    Invoke-PythonStep "[2/6] Running the first and only release evaluation on sealed data..." @(
        ".\scripts\phase_1_dense_sft\evaluate_model.py",
        "--model-id", $ModelDir,
        "--benchmark", $SealedData,
        "--output-dir", $SealedEval,
        "--max-new-tokens", "$MaxNewTokens"
    )
}

$AblationMatrix = Join-Path $OutputRoot "ablation_matrix.json"
Invoke-PythonStep "[3/6] Collecting immutable A-I ablation evidence..." @(
    ".\scripts\phase_8_release_candidate\collect_ablation_evidence.py",
    "--out", $AblationMatrix
)

Invoke-PythonStep "[4/6] Building the measured release-candidate audit..." @(
    ".\scripts\phase_8_release_candidate\build_release_audit.py",
    "--model-dir", $ModelDir,
    "--sealed-summary", $SealedSummary,
    "--ablation-matrix", $AblationMatrix
)

Invoke-PythonStep "[5/6] Creating the release-candidate package..." @(
    ".\scripts\phase_8_release_candidate\package_release_candidate.py",
    "--model-dir", $ModelDir,
    "--audit", ".\reports\RELEASE_CANDIDATE_AUDIT.md",
    "--reference-manifest", ".\artifacts\references\selected_inference_reference.json",
    "--output-dir", (Join-Path $OutputRoot "package"),
    "--mode", $PackageMode
)

Invoke-PythonStep "[6/6] Creating the final human release gate..." @(
    ".\scripts\phase_8_release_candidate\create_phase_8_gate.py"
)

Write-Host ""
Write-Host "Phase 8 execution completed."
Write-Host "Sealed evaluation: $SealedEval"
Write-Host "Audit: reports\RELEASE_CANDIDATE_AUDIT.md"
Write-Host "Package manifest: $OutputRoot\package\release_manifest.json"
Write-Host "Final gate: reports\PHASE_8_STOP_GATE.md"
Write-Host "Do not ship until the final human decision is PASS and rollback is tested."

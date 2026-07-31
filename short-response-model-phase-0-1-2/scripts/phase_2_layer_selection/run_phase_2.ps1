<#
.SYNOPSIS
Run Phase 2A selective-layer tuning, Phase 2B progressive structural reduction,
or Phase 2C targeted recovery in the ordered repository workflow.

.DESCRIPTION
Phase 2A asks where adaptation can be trained cheaply. Phase 2B asks which
complete inference layers can be removed and recovered. Phase 2C asks which
surviving layers can be updated for the best quality per training resource.
These are related but non-interchangeable experiments.
#>
param(
    [ValidateSet("All", "StaticOnly", "StructuralOnly", "TargetedRecoveryOnly")]
    [string]$Mode = "All",
    [switch]$Resume,
    [int]$MaxDrops = 6,
    [int]$RecoveryTopK = 3,
    [int]$RecoverySteps = 50,
    [switch]$KeepCandidateCheckpoints,
    [ValidateSet("balanced", "min_memory")]
    [string]$Phase2CSourceProfile = "balanced",
    [int]$Phase2CMaxSteps = 200,
    [string]$Phase2CRepeatSeeds = "23,37"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot

$GateReport = ".\reports\PHASE_1_STOP_GATE.md"
if (-not (Test-Path $GateReport)) { throw "Missing $GateReport. Complete Phase 1 first." }
$GateText = Get-Content $GateReport -Raw
if ($GateText -notmatch '(?im)^\s*-\s*Human decision:\s*`?PASS`?\s*$') {
  throw "Phase 1 needs an explicit human PASS before Phase 2."
}

if ($Mode -notin @("StructuralOnly", "TargetedRecoveryOnly")) {
    $StaticArgs = @(
        ".\scripts\phase_2_layer_selection\run_static_layer_study.py",
        "--model-id", "Qwen/Qwen3-0.6B",
        "--train-file", ".\data\train\train.jsonl",
        "--dev-file", ".\data\dev\dev.jsonl",
        "--benchmark", ".\data\benchmark\shortchatbench_pilot.jsonl",
        "--output-dir", ".\artifacts\phase_2_layer_study",
        "--qlora",
        "--gradient-checkpointing",
        "--max-steps", "200",
        "--seed", "11",
        "--repeat-seeds", "11,23,37"
    )
    if ($Resume) { $StaticArgs += "--resume" }
    & python @StaticArgs
    if ($LASTEXITCODE -ne 0) { throw "Phase 2A static study failed." }

    & python .\scripts\phase_2_layer_selection\collect_importance_signals.py
    if ($LASTEXITCODE -ne 0) { throw "Phase 2A importance-signal collection failed." }
    & python .\scripts\phase_2_layer_selection\build_phase_2_human_review.py
    if ($LASTEXITCODE -ne 0) { throw "Phase 2A human-review packet generation failed." }

    # A non-zero gate result is a valid negative research result. Continue only
    # when the repository contains an explicit human-approved deviation; never
    # silently relabel the failed selective-tuning experiment as a pass.
    & python .\scripts\phase_2_layer_selection\check_phase_2_gate.py
    if ($LASTEXITCODE -ne 0) {
        & python .\scripts\common\check_human_gate.py --phase 2 --allow-deviation
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2A failed and no approved deviation permits Phase 2B/3 continuation."
        }
        Write-Warning "Phase 2A remains a negative result; continuing under the approved deviation."
    }
    & python .\scripts\phase_2_layer_selection\plan_dynamic_selection.py
    if ($LASTEXITCODE -ne 0) { throw "Phase 2A dynamic-selection planning failed." }
}

if ($Mode -notin @("StaticOnly", "TargetedRecoveryOnly")) {
    $StructuralParams = @{
        MaxDrops = $MaxDrops
        RecoveryTopK = $RecoveryTopK
        RecoverySteps = $RecoverySteps
        QLoRA = $true
        GradientCheckpointing = $true
    }
    if ($Resume) { $StructuralParams["Resume"] = $true }
    if ($KeepCandidateCheckpoints) { $StructuralParams["KeepCandidateCheckpoints"] = $true }
    & .\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 @StructuralParams
    if ($LASTEXITCODE -ne 0) { throw "Phase 2B structural reduction failed." }
}

if ($Mode -eq "TargetedRecoveryOnly") {
    $Phase2CParams = @{
        Phase2BProfile = $Phase2CSourceProfile
        MaxSteps = $Phase2CMaxSteps
        RepeatSeeds = $Phase2CRepeatSeeds
        QLoRA = $true
        GradientCheckpointing = $true
    }
    if ($Resume) { $Phase2CParams["Resume"] = $true }
    & .\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 @Phase2CParams
    if ($LASTEXITCODE -ne 0) { throw "Phase 2C targeted recovery failed." }
}

Write-Host ""
Write-Host "Phase 2 workflow finished."
if ($Mode -notin @("StructuralOnly", "TargetedRecoveryOnly")) { Write-Host "Phase 2A gate: reports\PHASE_2_STOP_GATE.md" }
if ($Mode -notin @("StaticOnly", "TargetedRecoveryOnly")) { Write-Host "Phase 2B gate: reports\PHASE_2B_STOP_GATE.md" }
if ($Mode -eq "TargetedRecoveryOnly") { Write-Host "Phase 2C gate: reports\PHASE_2C_STOP_GATE.md" }

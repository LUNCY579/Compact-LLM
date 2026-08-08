<#
.SYNOPSIS
Optionally refine a proven Phase 9 SFT adapter with the integrated evidence-reasoning preferences.

.DESCRIPTION
This is deliberately not part of the default grounded-recovery ladder. Use it
only after SFT has demonstrated Oracle-RAG capability on dev. The corpus's
correct/maybe/uncertain/wrong alternatives are preference data, not SFT labels.
The wrapper continues the existing SFT adapter with DPO and leaves final model
selection to the same dev-first, sealed-test workflow.
#>
param(
    [string]$Model,
    [string]$SftAdapter,
    [string]$TrainPreferences = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\preferences\train.jsonl",
    [string]$DevPreferences = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\preferences\dev.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_9_evidence_preference_refinement_v1_7_5",
    [int]$MaxSteps = 200,
    [int]$Seed = 42,
    [int]$LoraRank = 16
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

foreach ($Path in @($Model, $SftAdapter, $TrainPreferences, $DevPreferences)) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) {
        throw "Missing evidence-preference refinement input: $Path"
    }
}

& python .\scripts\phase_3_preference_alignment\train_dpo.py `
    --model-id $Model `
    --sft-adapter $SftAdapter `
    --train-file $TrainPreferences `
    --dev-file $DevPreferences `
    --output-dir $OutputRoot `
    --max-steps $MaxSteps `
    --seed $Seed `
    --lora-rank $LoraRank
if ($LASTEXITCODE -ne 0) { throw "Evidence preference refinement failed with exit code $LASTEXITCODE." }

Write-Host "Evidence preference adapter: $(Join-Path $OutputRoot 'adapter')"
Write-Host "Evaluate it on the Phase 9 DEV benchmark before any sealed-test use."

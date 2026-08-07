<#
.SYNOPSIS
Optionally refine a proven Phase 9B reconstruction adapter with curated reconstruction preferences.

.DESCRIPTION
The dedicated reconstruction corpus provides three deliberate negatives per
canonical row (raw replay, over-rejection, wrong source). This wrapper is not part
of the default SFT/KD path. Use it only after SFT has shown reconstruction skill
on DEV. Test preferences are never used for training or model selection.
#>
param(
    [string]$Model,
    [string]$InitialAdapter,
    [string]$TrainPreferences = ".\artifacts\phase_9b_memory_reconstruction\dataset\preferences_train.jsonl",
    [string]$DevPreferences = ".\artifacts\phase_9b_memory_reconstruction\dataset\preferences_dev.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_9b_memory_reconstruction\student_reconstruction_dpo",
    [int]$MaxSteps = 200,
    [int]$Seed = 42,
    [int]$LoraRank = 16
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

foreach ($Path in @($Model, $InitialAdapter, $TrainPreferences, $DevPreferences)) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) {
        throw "Missing Phase 9B preference-refinement input: $Path"
    }
}
if ($TrainPreferences -match "preferences_test" -or $DevPreferences -match "preferences_test") {
    throw "Phase 9B sealed-test preferences cannot be used by DPO training/selection."
}

& python .\scripts\phase_3_preference_alignment\train_dpo.py `
    --model-id $Model `
    --sft-adapter $InitialAdapter `
    --train-file $TrainPreferences `
    --dev-file $DevPreferences `
    --output-dir $OutputRoot `
    --max-steps $MaxSteps `
    --seed $Seed `
    --lora-rank $LoraRank
if ($LASTEXITCODE -ne 0) { throw "Phase 9B reconstruction preference refinement failed with exit code $LASTEXITCODE." }

Write-Host "Reconstruction preference adapter: $(Join-Path $OutputRoot 'adapter')"
Write-Host "Evaluate it on the Phase 9B DEV canonical benchmark before any sealed-test use."

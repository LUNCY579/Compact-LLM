<#
.SYNOPSIS
Build, evaluate, and rank structural-reduction candidates after Phase 4.

.DESCRIPTION
This runner integrates structural reduction into the main phase chain. It never
overwrites the accepted Phase 4 checkpoint and never grants the Phase 4B human
gate. Pilot mode creates a practical small matrix; Full mode expands ratios.
#>
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Calibration,
    [string]$Benchmark = 'data\benchmark\shortchatbench_pilot.jsonl',
    [ValidateSet('Pilot', 'Full')][string]$Mode = 'Pilot',
    [switch]$EnableAttentionPruning,
    [switch]$ReuseExisting,
    [int]$MaxNewTokens = 32
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root

# Phase 4B is downstream of the accepted Phase 4 model/soup decision.
python .\scripts\common\check_human_gate.py --phase 4

$ArtifactRoot = 'artifacts\phase_4b_structural_reduction'

# Never silently overwrite a prior Phase 4B run. Archive/delete the artifact
# directory for a clean rerun, or use -ReuseExisting only when all inputs and
# code revisions still match.
if ((Test-Path $ArtifactRoot) -and (-not $ReuseExisting)) {
    throw "$ArtifactRoot already exists. Archive it, remove it, or rerun with -ReuseExisting after verifying provenance."
}
New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null

$Profile = Join-Path $ArtifactRoot 'long_form_profile.pt'
$CandidateSpecs = New-Object System.Collections.Generic.List[string]

# Pilot mode is intended to produce a useful answer quickly. Full mode is a
# broader sweep for later optimization once the pipeline is already working.
if ($Mode -eq 'Pilot') {
    $MlpRatios = @(0.10)
    $DepthDrops = @(2)
} else {
    $MlpRatios = @(0.05, 0.10, 0.20)
    $DepthDrops = @(1, 2, 4)
}

if (-not ($ReuseExisting -and (Test-Path $Profile))) {
    python .\scripts\phase_4b_structural_reduction\profile_long_form_components.py `
      --model $Model --data $Calibration --output $Profile
}

foreach ($Ratio in $MlpRatios) {
    $Percent = [int][Math]::Round($Ratio * 100)

    # Attribution-guided MLP pruning candidate.
    $Plan = Join-Path $ArtifactRoot "plans\mlp_long_specific_${Percent}pct.json"
    $Candidate = Join-Path $ArtifactRoot "candidates\mlp_long_specific_${Percent}pct"
    if (-not ($ReuseExisting -and (Test-Path $Candidate))) {
        python .\scripts\phase_4b_structural_reduction\build_pruning_plan.py `
          --profile $Profile --mlp-prune-ratio $Ratio `
          --attention-group-prune-ratio 0.0 --selection-method long-specific `
          --output $Plan
        python .\scripts\phase_4b_structural_reduction\apply_structured_pruning.py `
          --model $Model --plan $Plan --skip-attention --output-dir $Candidate
    }
    $CandidateSpecs.Add("mlp_long_specific_${Percent}pct=$Candidate") | Out-Null

    # Equal-size random control. A guided candidate should beat this control to
    # show that the attribution signal adds value beyond removing any channels.
    $RandomPlan = Join-Path $ArtifactRoot "plans\mlp_random_${Percent}pct.json"
    $RandomCandidate = Join-Path $ArtifactRoot "candidates\mlp_random_${Percent}pct"
    if (-not ($ReuseExisting -and (Test-Path $RandomCandidate))) {
        python .\scripts\phase_4b_structural_reduction\build_pruning_plan.py `
          --profile $Profile --mlp-prune-ratio $Ratio `
          --attention-group-prune-ratio 0.0 --selection-method random `
          --output $RandomPlan
        python .\scripts\phase_4b_structural_reduction\apply_structured_pruning.py `
          --model $Model --plan $RandomPlan --skip-attention `
          --output-dir $RandomCandidate
    }
    $CandidateSpecs.Add("mlp_random_${Percent}pct=$RandomCandidate") | Out-Null
}

foreach ($DropCount in $DepthDrops) {
    # Suffix removal tests the late-layer hypothesis directly.
    $DropCandidate = Join-Path $ArtifactRoot "candidates\drop_last_$DropCount"
    if (-not ($ReuseExisting -and (Test-Path $DropCandidate))) {
        python .\scripts\phase_4b_structural_reduction\reduce_depth.py `
          --model $Model --mode drop-last --count $DropCount `
          --output-dir $DropCandidate
    }
    $CandidateSpecs.Add("drop_last_$DropCount=$DropCandidate") | Out-Null

    # Uniform thinning removes the same number of layers but distributes the
    # removals, providing an equal-layer-count control.
    $UniformCandidate = Join-Path $ArtifactRoot "candidates\uniform_drop_$DropCount"
    if (-not ($ReuseExisting -and (Test-Path $UniformCandidate))) {
        python .\scripts\phase_4b_structural_reduction\reduce_depth.py `
          --model $Model --mode uniform-drop --count $DropCount `
          --output-dir $UniformCandidate
    }
    $CandidateSpecs.Add("uniform_drop_$DropCount=$UniformCandidate") | Out-Null
}

if ($EnableAttentionPruning) {
    # Attention pruning is opt-in because grouped-query attention layouts are
    # more architecture-sensitive than MLP channel slicing.
    $AttentionPlan = Join-Path $ArtifactRoot 'plans\attention_groups_10pct.json'
    $AttentionCandidate = Join-Path $ArtifactRoot 'candidates\attention_groups_10pct'
    if (-not ($ReuseExisting -and (Test-Path $AttentionCandidate))) {
        python .\scripts\phase_4b_structural_reduction\build_pruning_plan.py `
          --profile $Profile --mlp-prune-ratio 0.0 `
          --attention-group-prune-ratio 0.10 --selection-method long-specific `
          --output $AttentionPlan
        python .\scripts\phase_4b_structural_reduction\apply_structured_pruning.py `
          --model $Model --plan $AttentionPlan --output-dir $AttentionCandidate
    }
    $CandidateSpecs.Add("attention_groups_10pct=$AttentionCandidate") | Out-Null
}

# Build one argument list so every candidate uses the exact same evaluator,
# benchmark, output cap, and evidence format.
$EvalArgs = @(
    '.\scripts\phase_4b_structural_reduction\evaluate_candidates.py',
    '--dense-reference', $Model,
    '--benchmark', $Benchmark,
    '--output-root', $ArtifactRoot,
    '--max-new-tokens', $MaxNewTokens
)
foreach ($Spec in $CandidateSpecs) {
    $EvalArgs += @('--candidate', $Spec)
}
if ($ReuseExisting) {
    $EvalArgs += '--reuse-existing'
}
& python @EvalArgs

$Registry = Join-Path $ArtifactRoot 'candidate_registry.json'
python .\scripts\phase_4b_structural_reduction\recommend_reference_candidate.py `
  --registry $Registry

Write-Host ''
Write-Host 'Phase 4B candidate generation and recommendation are complete.'
Write-Host 'Review raw evaluations and reports/PHASE_4B_CANDIDATE_RECOMMENDATION.md.'
Write-Host 'Do not run downstream phases until PHASE_4B_STOP_GATE.md is approved and'
Write-Host 'finalize_reference_selection.py has written the selected-reference manifest.'

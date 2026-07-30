<#
.SYNOPSIS
Build, evaluate, and rank permanent structural-reduction candidates.

.DESCRIPTION
Phase 4B consumes the exact full checkpoint accepted in Phase 4. When -Model is
omitted, the runner resolves artifacts/references/phase_4_reference.json.
When -Calibration is omitted, it creates matched short/long calibration records
from real development prompts and generated long continuations.

The runner creates practical pruning/depth candidates, evaluates all of them
with one benchmark/configuration, writes a recommendation, and prepares the
human stop-gate report. It never selects the downstream model automatically.
#>
param(
    [string]$Model = "",
    [string]$Phase4GateReport = ".\reports\PHASE_4_STOP_GATE.md",
    [string]$Phase4ReferenceManifest = ".\artifacts\references\phase_4_reference.json",
    [string]$BranchName = "full_quality",
    [string]$Calibration = ".\data\phase_4b_structural_reduction\long_form_calibration.jsonl",
    [string]$CalibrationSource = ".\data\dev\dev.jsonl",
    [string]$Benchmark = ".\data\benchmark\shortchatbench_pilot.jsonl",
    [ValidateSet("Pilot", "Full")][string]$Mode = "Pilot",
    [switch]$EnableAttentionPruning,
    [switch]$ReuseExisting,
    [int]$MaxNewTokens = 32,
    [int]$CalibrationPairs = 128,
    [int]$ShortCutoff = 24,
    [int]$LateStart = 48
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

# Keep profile branches isolated automatically. The full-quality defaults stay
# unchanged, while a named branch resolves its matching Phase 4 gate/reference.
if ($BranchName -ne "full_quality") {
    $SafeBranchDefaults = $BranchName -replace '[^A-Za-z0-9_.-]', '_'
    if (-not ($PSBoundParameters.ContainsKey('Phase4GateReport'))) {
        $Phase4GateReport = ".\reports\PHASE_4_${SafeBranchDefaults}_STOP_GATE.md"
    }
    if (-not ($PSBoundParameters.ContainsKey('Phase4ReferenceManifest'))) {
        $Phase4ReferenceManifest = ".\artifacts\references\phase_4_reference_${SafeBranchDefaults}.json"
    }
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

Invoke-PythonStep "[1/8] Checking the Phase 4 human gate..." @(
    ".\scripts\common\check_human_gate.py", "--phase", "4", "--report", $Phase4GateReport
)

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = python .\scripts\phase_4_model_soups\resolve_phase_4_model.py --manifest $Phase4ReferenceManifest
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve the accepted Phase 4 model." }
}
if (-not (Test-Path $Model)) { throw "Phase 4B requires a local full checkpoint directory: $Model" }
if (-not (Test-Path $Benchmark)) { throw "Missing Phase 4B benchmark: $Benchmark" }

if ($BranchName -eq "full_quality") {
    $ArtifactRoot = ".\artifacts\phase_4b_structural_reduction"
    $RecommendationOut = Join-Path $ArtifactRoot "candidate_recommendation.json"
    $RecommendationReport = ".\reports\PHASE_4B_CANDIDATE_RECOMMENDATION.md"
    $Phase4BGateReport = ".\reports\PHASE_4B_STOP_GATE.md"
} else {
    $SafeBranch = $BranchName -replace '[^A-Za-z0-9_.-]', '_'
    $ArtifactRoot = ".\artifacts\phase_4b_structural_reduction_$SafeBranch"
    $RecommendationOut = Join-Path $ArtifactRoot "candidate_recommendation.json"
    $RecommendationReport = ".\reports\PHASE_4B_CANDIDATE_RECOMMENDATION_$SafeBranch.md"
    $Phase4BGateReport = ".\reports\PHASE_4B_${SafeBranch}_STOP_GATE.md"
}
if ((Test-Path $ArtifactRoot) -and (-not $ReuseExisting)) {
    throw "$ArtifactRoot already exists. Archive/remove it or use -ReuseExisting only after verifying all provenance."
}
New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null

if (-not (Test-Path $Calibration)) {
    if (-not (Test-Path $CalibrationSource)) { throw "Missing calibration source: $CalibrationSource" }
    Invoke-PythonStep "[2/8] Building matched short/long calibration records..." @(
        ".\scripts\phase_4b_structural_reduction\build_long_form_calibration.py",
        "--model", $Model,
        "--source", $CalibrationSource,
        "--output", $Calibration,
        "--max-records", "$CalibrationPairs",
        "--minimum-long-tokens", "$([Math]::Max($LateStart + 8, 64))",
        "--max-new-tokens", "$([Math]::Max($LateStart + 80, 128))"
    )
}

$Profile = Join-Path $ArtifactRoot "long_form_profile.pt"
if (-not ($ReuseExisting -and (Test-Path $Profile))) {
    Invoke-PythonStep "[3/8] Profiling short-versus-long continuation components..." @(
        ".\scripts\phase_4b_structural_reduction\profile_long_form_components.py",
        "--model", $Model,
        "--data", $Calibration,
        "--output", $Profile,
        "--short-cutoff", "$ShortCutoff",
        "--late-start", "$LateStart"
    )
}

$CandidateSpecs = New-Object System.Collections.Generic.List[string]
if ($Mode -eq "Pilot") {
    $MlpRatios = @(0.10)
    $DepthDrops = @(2)
} else {
    $MlpRatios = @(0.05, 0.10, 0.20)
    $DepthDrops = @(1, 2, 4)
}

foreach ($Ratio in $MlpRatios) {
    $Percent = [int][Math]::Round($Ratio * 100)
    foreach ($Method in @("long-specific", "random")) {
        $SafeMethod = $Method.Replace("-", "_")
        $Plan = Join-Path $ArtifactRoot "plans\mlp_${SafeMethod}_${Percent}pct.json"
        $Candidate = Join-Path $ArtifactRoot "candidates\mlp_${SafeMethod}_${Percent}pct"
        if (-not ($ReuseExisting -and (Test-Path (Join-Path $Candidate "config.json")))) {
            Invoke-PythonStep "[4/8] Building $Method MLP pruning plan at $Percent%..." @(
                ".\scripts\phase_4b_structural_reduction\build_pruning_plan.py",
                "--profile", $Profile,
                "--mlp-prune-ratio", "$Ratio",
                "--attention-group-prune-ratio", "0.0",
                "--selection-method", $Method,
                "--output", $Plan
            )
            Invoke-PythonStep "[4/8] Applying $Method MLP pruning at $Percent%..." @(
                ".\scripts\phase_4b_structural_reduction\apply_structured_pruning.py",
                "--model", $Model,
                "--plan", $Plan,
                "--skip-attention",
                "--output-dir", $Candidate
            )
        }
        $CandidateSpecs.Add("mlp_${SafeMethod}_${Percent}pct=$Candidate") | Out-Null
    }
}

foreach ($DropCount in $DepthDrops) {
    foreach ($ModeName in @("drop-last", "uniform-drop")) {
        $SafeMode = $ModeName.Replace("-", "_")
        $Candidate = Join-Path $ArtifactRoot "candidates\${SafeMode}_$DropCount"
        if (-not ($ReuseExisting -and (Test-Path (Join-Path $Candidate "config.json")))) {
            Invoke-PythonStep "[5/8] Building depth candidate $ModeName ($DropCount layers)..." @(
                ".\scripts\phase_4b_structural_reduction\reduce_depth.py",
                "--model", $Model,
                "--mode", $ModeName,
                "--count", "$DropCount",
                "--output-dir", $Candidate
            )
        }
        $CandidateSpecs.Add("${SafeMode}_$DropCount=$Candidate") | Out-Null
    }
}

if ($EnableAttentionPruning) {
    $AttentionPlan = Join-Path $ArtifactRoot "plans\attention_groups_10pct.json"
    $AttentionCandidate = Join-Path $ArtifactRoot "candidates\attention_groups_10pct"
    if (-not ($ReuseExisting -and (Test-Path (Join-Path $AttentionCandidate "config.json")))) {
        Invoke-PythonStep "[5/8] Building grouped-attention pruning candidate..." @(
            ".\scripts\phase_4b_structural_reduction\build_pruning_plan.py",
            "--profile", $Profile,
            "--mlp-prune-ratio", "0.0",
            "--attention-group-prune-ratio", "0.10",
            "--selection-method", "long-specific",
            "--output", $AttentionPlan
        )
        Invoke-PythonStep "[5/8] Applying grouped-attention pruning candidate..." @(
            ".\scripts\phase_4b_structural_reduction\apply_structured_pruning.py",
            "--model", $Model,
            "--plan", $AttentionPlan,
            "--output-dir", $AttentionCandidate
        )
    }
    $CandidateSpecs.Add("attention_groups_10pct=$AttentionCandidate") | Out-Null
}

$EvalArgs = @(
    ".\scripts\phase_4b_structural_reduction\evaluate_candidates.py",
    "--dense-reference", $Model,
    "--benchmark", $Benchmark,
    "--output-root", $ArtifactRoot,
    "--max-new-tokens", "$MaxNewTokens"
)
foreach ($Spec in $CandidateSpecs) { $EvalArgs += @("--candidate", $Spec) }
if ($ReuseExisting) { $EvalArgs += "--reuse-existing" }
Invoke-PythonStep "[6/8] Evaluating dense and structural candidates..." $EvalArgs

$Registry = Join-Path $ArtifactRoot "candidate_registry.json"
Invoke-PythonStep "[7/8] Applying practical deployment floors and ranking candidates..." @(
    ".\scripts\phase_4b_structural_reduction\recommend_reference_candidate.py",
    "--registry", $Registry,
    "--out", $RecommendationOut,
    "--report", $RecommendationReport
)
Invoke-PythonStep "[8/8] Creating the Phase 4B human stop-gate report..." @(
    ".\scripts\phase_4b_structural_reduction\create_phase_4b_gate.py",
    "--recommendation", $RecommendationOut,
    "--out", $Phase4BGateReport
)

Write-Host ""
Write-Host "Phase 4B execution completed."
Write-Host "Branch: $BranchName"
Write-Host "Artifact root: $ArtifactRoot"
Write-Host "Stop gate: $Phase4BGateReport"
Write-Host "Candidate registry: $Registry"
Write-Host "Recommendation: $RecommendationReport"
Write-Host "Stop gate: $Phase4BGateReport"
Write-Host ""
Write-Host "After review, set the selected candidate and Human decision to PASS, then run:"
Write-Host "python .\scripts\phase_4b_structural_reduction\finalize_reference_selection.py --registry $Registry --candidate <name> --reason '<reason>'"

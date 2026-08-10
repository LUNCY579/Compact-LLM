param(
    [Parameter(Mandatory=$true)][string]$StudentModel,
    [string]$StudentAdapter = "",
    [string]$OutputRoot = ".\artifacts\phase_9_procedural_recovery_v211_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string]$OriginalCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$RecoveryCorpusRoot = ".\data\phase_9_student_distillation\procedural_recovery_v211",
    [int]$StageASteps = 60,
    [int]$StageB1Steps = 60,
    [int]$StageB2Steps = 60,
    [double]$StageALearningRate = 1e-5,
    [double]$StageB1LearningRate = 5e-6,
    [double]$StageB2LearningRate = 3e-6,
    [int]$LoraRank = 16,
    [int]$MaximumEvidenceTokens = 384,
    [int]$Seed = 21142,
    [switch]$RegenerateDataset,
    [switch]$SkipTraining,
    [switch]$RunLegacyReference,
    [switch]$RunSealedTest,
    [switch]$RequireCuda,
    [switch]$ReuseExisting,
    [switch]$AllowAggressiveTraining
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

$ConfigPath = ".\configs\phase_9_procedural_recovery_v211.json"

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

function Ensure-Path {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path $Path)) { throw "Missing ${Description}: $Path" }
}

function Should-Run {
    param([string]$Artifact)
    if (Test-Path $Artifact) {
        if ($ReuseExisting) { return $false }
        throw "Artifact already exists: $Artifact. Use a fresh -OutputRoot or pass -ReuseExisting deliberately."
    }
    return $true
}

function Invoke-Eval {
    param(
        [string]$Name,
        [string]$Model,
        [string]$Adapter,
        [string]$Benchmark,
        [string]$Database,
        [string]$Calibration,
        [string]$Destination,
        [string]$SelectionPolicy = "query_aware_v21",
        [string]$Conditions = "oracle_rag,real_rag",
        # DEV checkpoint ladder must finish even if a rare decode glitch marks the
        # packet non-research-ready. Sealed TEST keeps the strict gate.
        [switch]$RequireResearchReady
    )
    $summary = Join-Path $Destination "summary.json"
    if (-not (Should-Run $summary)) { return }
    $args = @(
        ".\scripts\phase_9_student_distillation\evaluate_rag.py",
        "--model", $Model,
        "--benchmark", $Benchmark,
        "--output-dir", $Destination,
        "--knowledge-db", $Database,
        "--gate-calibration", $Calibration,
        "--conditions", $Conditions,
        "--retrieval-selection-policy", $SelectionPolicy,
        "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
        "--max-new-tokens", "32",
        "--minimum-cases-per-condition", "1",
        "--minimum-supported-cases", "1",
        "--minimum-unsupported-cases", "1",
        # Always resume durable results.jsonl so a late readiness failure does not
        # force a full re-decode of an already-completed DEV evaluation.
        "--resume"
    )
    if ($RequireResearchReady) { $args += "--require-research-ready" }
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $args += @("--adapter", $Adapter) }
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep $Name $args
}

function Invoke-RecoveryTraining {
    param(
        [string]$Name,
        [string]$TrainFile,
        [string]$DevFile,
        [string]$Destination,
        [double]$LearningRate,
        [int]$MaxSteps,
        [string]$InitialAdapter
    )
    $manifest = Join-Path $Destination "training_manifest.json"
    if (-not (Should-Run $manifest)) { return }
    $args = @(
        ".\scripts\phase_9_student_distillation\train_student.py",
        "--student", $StudentModel,
        "--objective", "sft",
        "--train-file", $TrainFile,
        "--dev-file", $DevFile,
        "--output-dir", $Destination,
        "--max-steps", "$MaxSteps",
        "--learning-rate", "$LearningRate",
        "--gradient-accumulation-steps", "32",
        "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
        "--first-answer-token-weight", "2.0",
        "--evidence-anchor-token-weight", "2.0",
        "--abstention-token-weight", "1.75",
        "--eos-loss-weight", "1.5",
        "--eval-steps", "30",
        "--seed", "$Seed"
    )
    if ([string]::IsNullOrWhiteSpace($InitialAdapter)) {
        $args += @("--lora", "--lora-rank", "$LoraRank")
    } else {
        $args += @("--initial-adapter", $InitialAdapter)
    }
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep $Name $args
}

function Assert-ConservativeSchedule {
    if ($AllowAggressiveTraining) { return }
    $maxLr = [Math]::Max($StageALearningRate, [Math]::Max($StageB1LearningRate, $StageB2LearningRate))
    $totalSteps = $StageASteps + $StageB1Steps + $StageB2Steps
    if ($maxLr -gt 3e-5) {
        throw "v2.1.1 preservation guard: learning rate $maxLr exceeds 3e-5. Use -AllowAggressiveTraining only for an explicit ablation."
    }
    if ($totalSteps -gt 400) {
        throw "v2.1.1 preservation guard: total recovery steps $totalSteps exceed 400. Use -AllowAggressiveTraining only for an explicit ablation."
    }
}

Assert-ConservativeSchedule
Ensure-Path $ConfigPath "v2.1.1 recovery config"
Ensure-Path $StudentModel "current student/base checkpoint"
if (-not [string]::IsNullOrWhiteSpace($StudentAdapter)) { Ensure-Path $StudentAdapter "current student adapter" }
Ensure-Path $OriginalCorpusRoot "original Phase 9 corpus"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$OriginalKnowledge = Join-Path $OriginalCorpusRoot "knowledge.jsonl"
$OriginalProcedures = Join-Path $OriginalCorpusRoot "procedures.jsonl"
$OriginalTrain = Join-Path $OriginalCorpusRoot "reviewed_splits\train.reviewed.jsonl"
$OriginalDev = Join-Path $OriginalCorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$OriginalTest = Join-Path $OriginalCorpusRoot "reviewed_splits\test.reviewed.jsonl"

if ($RegenerateDataset -or -not (Test-Path (Join-Path $RecoveryCorpusRoot "manifests\dataset_manifest.json"))) {
    Invoke-PythonStep "[v2.1.1 Data] Generate preservation-first real-world procedural corpus" @(
        ".\scripts\phase_9_student_distillation\generate_procedural_recovery_v211.py",
        "--output-root", $RecoveryCorpusRoot,
        "--original-train", $OriginalTrain,
        "--original-dev", $OriginalDev,
        "--seed", "$Seed"
    )
}
Invoke-PythonStep "[v2.1.1 Data] Validate procedural recovery corpus" @(
    ".\scripts\phase_9_student_distillation\validate_rag_reasoning_corpus.py",
    "--corpus-root", $RecoveryCorpusRoot
)
Invoke-PythonStep "[Data audit] Preserve historical v1.7.5 procedure defects as evidence" @(
    ".\scripts\phase_9_student_distillation\audit_procedural_gold_consistency_v21.py",
    "--input", $OriginalTrain,
    "--output", (Join-Path $OutputRoot "historical_procedural_gold_audit.json")
)
Invoke-PythonStep "[Data audit] Require clean v2.1.1 generated latency gold" @(
    ".\scripts\phase_9_student_distillation\audit_procedural_gold_consistency_v21.py",
    "--input", (Join-Path $RecoveryCorpusRoot "reviewed_splits\train.reviewed.jsonl"),
    "--output", (Join-Path $OutputRoot "v211_train_procedural_gold_audit.json"),
    "--require-clean"
)

$RecoveryKnowledge = Join-Path $RecoveryCorpusRoot "knowledge.jsonl"
$RecoveryProcedures = Join-Path $RecoveryCorpusRoot "procedures.jsonl"
$StageATrain = Join-Path $RecoveryCorpusRoot "curriculum_views\train_stage_a_short_binding.reviewed.jsonl"
$StageBTrain = Join-Path $RecoveryCorpusRoot "curriculum_views\train_stage_b_preservation_first.reviewed.jsonl"
$StageBDevProbe = Join-Path $RecoveryCorpusRoot "curriculum_views\dev_preservation_probe.reviewed.jsonl"
$RecoveryDev = Join-Path $RecoveryCorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$RecoveryTest = Join-Path $RecoveryCorpusRoot "reviewed_splits\test.reviewed.jsonl"

$OriginalDb = Join-Path $OutputRoot "original_phase9.sqlite3"
$RecoveryDb = Join-Path $OutputRoot "procedural_v211.sqlite3"
if (-not (Test-Path $OriginalDb)) {
    Invoke-PythonStep "[Retrieval] Build original Phase 9 index" @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py",
        "--knowledge", $OriginalKnowledge,
        "--procedures", $OriginalProcedures,
        "--output", $OriginalDb
    )
}
if (-not (Test-Path $RecoveryDb)) {
    Invoke-PythonStep "[Retrieval] Build v2.1.1 procedural index" @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py",
        "--knowledge", $RecoveryKnowledge,
        "--procedures", $RecoveryProcedures,
        "--output", $RecoveryDb
    )
}

$OriginalCalibration = Join-Path $OutputRoot "original_calibration_query_aware_v21.json"
$RecoveryCalibration = Join-Path $OutputRoot "v211_calibration_query_aware_v21.json"
if (-not (Test-Path $OriginalCalibration)) {
    Invoke-PythonStep "[Retrieval] Freeze original DEV query-aware calibration" @(
        ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py",
        "--benchmark", $OriginalDev,
        "--knowledge-db", $OriginalDb,
        "--output", $OriginalCalibration,
        "--retrieval-selection-policy", "query_aware_v21",
        "--require-research-ready"
    )
}
if (-not (Test-Path $RecoveryCalibration)) {
    Invoke-PythonStep "[Retrieval] Freeze v2.1.1 challenge DEV query-aware calibration" @(
        ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py",
        "--benchmark", $RecoveryDev,
        "--knowledge-db", $RecoveryDb,
        "--output", $RecoveryCalibration,
        "--retrieval-selection-policy", "query_aware_v21",
        "--require-research-ready"
    )
}

$RetrievalAudit = Join-Path $OutputRoot "v211_retrieval_coverage.json"
if (-not (Test-Path $RetrievalAudit)) {
    Invoke-PythonStep "[Retrieval] Verify v2.1.1 DEV support recall" @(
        ".\scripts\phase_9_student_distillation\audit_rag_retrieval_coverage.py",
        "--benchmark", $RecoveryDev,
        "--knowledge-db", $RecoveryDb,
        "--output", $RetrievalAudit,
        "--retrieval-selection-policy", "query_aware_v21",
        "--require-complete-support-recall"
    )
}

$BaselineOriginal = Join-Path $OutputRoot "baseline_original_query_aware"
$BaselineChallenge = Join-Path $OutputRoot "baseline_v211_challenge_query_aware"
Invoke-Eval "[Candidate 0] Unchanged student / original DEV / query-aware" $StudentModel $StudentAdapter $OriginalDev $OriginalDb $OriginalCalibration $BaselineOriginal
Invoke-Eval "[Candidate 0] Unchanged student / v2.1.1 challenge DEV" $StudentModel $StudentAdapter $RecoveryDev $RecoveryDb $RecoveryCalibration $BaselineChallenge

if ($RunLegacyReference) {
    $LegacyCalibration = Join-Path $OutputRoot "original_calibration_legacy.json"
    if (-not (Test-Path $LegacyCalibration)) {
        Invoke-PythonStep "[Reference] Calibrate original DEV legacy retrieval" @(
            ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py",
            "--benchmark", $OriginalDev,
            "--knowledge-db", $OriginalDb,
            "--output", $LegacyCalibration,
            "--retrieval-selection-policy", "legacy",
            "--require-research-ready"
        )
    }
    Invoke-Eval "[Reference] Unchanged student / original DEV / legacy" $StudentModel $StudentAdapter $OriginalDev $OriginalDb $LegacyCalibration (Join-Path $OutputRoot "baseline_original_legacy") "legacy"
}

if ($SkipTraining) {
    Write-Host ""
    Write-Host "v2.1.1 baseline-only workflow complete. No weights were changed."
    Write-Host "Output: $OutputRoot"
    exit 0
}

# Bounded checkpoint ladder. These are sequential micro-stages, not a broad
# hyperparameter sweep. Every checkpoint is evaluated on BOTH the strong
# original query-aware path and the new challenge before final DEV selection.
$StageA = Join-Path $OutputRoot "train_stage_a_short_binding"
Invoke-RecoveryTraining "[Train A] Short residual binding burst" $StageATrain $RecoveryDev $StageA $StageALearningRate $StageASteps $StudentAdapter
$StageAAdapter = Join-Path $StageA "adapter"
Ensure-Path $StageAAdapter "Stage A adapter"
$StageAOriginal = Join-Path $OutputRoot "stage_a_original_query_aware"
$StageAChallenge = Join-Path $OutputRoot "stage_a_v211_challenge"
Invoke-Eval "[Stage A] Original Phase 9 DEV" $StudentModel $StageAAdapter $OriginalDev $OriginalDb $OriginalCalibration $StageAOriginal
Invoke-Eval "[Stage A] v2.1.1 challenge DEV" $StudentModel $StageAAdapter $RecoveryDev $RecoveryDb $RecoveryCalibration $StageAChallenge

$StageB1 = Join-Path $OutputRoot "train_stage_b1_preservation_first"
Invoke-RecoveryTraining "[Train B1] Preservation-first consolidation" $StageBTrain $StageBDevProbe $StageB1 $StageB1LearningRate $StageB1Steps $StageAAdapter
$StageB1Adapter = Join-Path $StageB1 "adapter"
Ensure-Path $StageB1Adapter "Stage B1 adapter"
$StageB1Original = Join-Path $OutputRoot "stage_b1_original_query_aware"
$StageB1Challenge = Join-Path $OutputRoot "stage_b1_v211_challenge"
Invoke-Eval "[Stage B1] Original Phase 9 DEV" $StudentModel $StageB1Adapter $OriginalDev $OriginalDb $OriginalCalibration $StageB1Original
Invoke-Eval "[Stage B1] v2.1.1 challenge DEV" $StudentModel $StageB1Adapter $RecoveryDev $RecoveryDb $RecoveryCalibration $StageB1Challenge

$StageB2 = Join-Path $OutputRoot "train_stage_b2_preservation_first"
Invoke-RecoveryTraining "[Train B2] Second micro-consolidation checkpoint" $StageBTrain $StageBDevProbe $StageB2 $StageB2LearningRate $StageB2Steps $StageB1Adapter
$StageB2Adapter = Join-Path $StageB2 "adapter"
Ensure-Path $StageB2Adapter "Stage B2 adapter"
$StageB2Original = Join-Path $OutputRoot "stage_b2_original_query_aware"
$StageB2Challenge = Join-Path $OutputRoot "stage_b2_v211_challenge"
Invoke-Eval "[Stage B2] Original Phase 9 DEV" $StudentModel $StageB2Adapter $OriginalDev $OriginalDb $OriginalCalibration $StageB2Original
Invoke-Eval "[Stage B2] v2.1.1 challenge DEV" $StudentModel $StageB2Adapter $RecoveryDev $RecoveryDb $RecoveryCalibration $StageB2Challenge

$SelectionJson = Join-Path $OutputRoot "procedural_recovery_v211_selection.json"
$SelectionMd = Join-Path $OutputRoot "PROCEDURAL_RECOVERY_V211_SELECTION.md"
Invoke-PythonStep "[Select] Preservation-adjusted DEV candidate selection" @(
    ".\scripts\phase_9_student_distillation\select_procedural_recovery_v211.py",
    "--config", $ConfigPath,
    "--baseline-original", (Join-Path $BaselineOriginal "summary.json"),
    "--baseline-challenge", (Join-Path $BaselineChallenge "summary.json"),
    "--baseline-adapter", $StudentAdapter,
    "--stage-a-original", (Join-Path $StageAOriginal "summary.json"),
    "--stage-a-challenge", (Join-Path $StageAChallenge "summary.json"),
    "--stage-a-adapter", $StageAAdapter,
    "--stage-b1-original", (Join-Path $StageB1Original "summary.json"),
    "--stage-b1-challenge", (Join-Path $StageB1Challenge "summary.json"),
    "--stage-b1-adapter", $StageB1Adapter,
    "--stage-b2-original", (Join-Path $StageB2Original "summary.json"),
    "--stage-b2-challenge", (Join-Path $StageB2Challenge "summary.json"),
    "--stage-b2-adapter", $StageB2Adapter,
    "--output-json", $SelectionJson,
    "--output-md", $SelectionMd
)

$Selection = Get-Content -Raw $SelectionJson | ConvertFrom-Json
$SelectedName = [string]$Selection.selected_candidate
$SelectedAdapter = [string]$Selection.selected_adapter
Write-Host ""
Write-Host "DEV-selected v2.1.1 candidate: $SelectedName"
if ([string]::IsNullOrWhiteSpace($SelectedAdapter)) {
    Write-Host "Selected path uses the original model weights with no adapter."
} else {
    Write-Host "Selected adapter: $SelectedAdapter"
}

if ($RunSealedTest) {
    Set-Content -Path (Join-Path $OutputRoot "SEALED_TEST_POLICY.txt") -Value @"
Selected on DEV: $SelectedName
Original TEST uses frozen DEV calibration: $OriginalCalibration
v2.1.1 TEST uses frozen DEV calibration: $RecoveryCalibration
No TEST-based recalibration or training is allowed.
"@
    Invoke-Eval "[SEALED] DEV-selected candidate / original Phase 9 TEST" $StudentModel $SelectedAdapter $OriginalTest $OriginalDb $OriginalCalibration (Join-Path $OutputRoot "sealed_selected_original_test") -RequireResearchReady
    Invoke-Eval "[SEALED] DEV-selected candidate / v2.1.1 procedural TEST" $StudentModel $SelectedAdapter $RecoveryTest $RecoveryDb $RecoveryCalibration (Join-Path $OutputRoot "sealed_selected_v211_test") -RequireResearchReady
}

Write-Host ""
Write-Host "v2.1.1 procedural recovery workflow complete."
Write-Host "Output: $OutputRoot"
Write-Host "Review PROCEDURAL_RECOVERY_V211_SELECTION.md before Student Optimization v2."

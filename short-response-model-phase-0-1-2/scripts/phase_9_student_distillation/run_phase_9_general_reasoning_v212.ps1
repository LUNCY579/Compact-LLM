param(
    [Parameter(Mandatory=$true)][string]$StudentModel,
    [string]$StudentAdapter = "",
    [string]$OutputRoot = ".\artifacts\phase_9_general_reasoning_v212_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string]$OriginalCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$V211CorpusRoot = ".\data\phase_9_student_distillation\procedural_recovery_v211",
    [string]$ReasoningCorpusRoot = ".\data\phase_9_student_distillation\general_reasoning_v212",
    [int]$StageASteps = 160,
    [int]$StageB1Steps = 160,
    [int]$StageB2Steps = 120,
    [double]$StageALearningRate = 1e-5,
    [double]$StageB1LearningRate = 7.5e-6,
    [double]$StageB2LearningRate = 5e-6,
    [int]$LoraRank = 16,
    [int]$MaximumEvidenceTokens = 384,
    [int]$Seed = 21242,
    [switch]$RegenerateDataset,
    [switch]$SkipTraining,
    [switch]$RunSealedTest,
    [switch]$RequireCuda,
    [switch]$ReuseExisting,
    [switch]$AllowAggressiveTraining
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root
$ConfigPath = ".\configs\phase_9_general_reasoning_v212.json"

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

function Invoke-RagEval {
    param(
        [string]$Name,
        [string]$Model,
        [string]$Adapter,
        [string]$Benchmark,
        [string]$Database,
        [string]$Calibration,
        [string]$Destination,
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
        "--conditions", "oracle_rag,real_rag",
        "--retrieval-selection-policy", "query_aware_v21",
        "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
        "--max-new-tokens", "32",
        "--minimum-cases-per-condition", "1",
        "--minimum-supported-cases", "1",
        "--minimum-unsupported-cases", "1",
        "--resume"
    )
    if ($RequireResearchReady) { $args += "--require-research-ready" }
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $args += @("--adapter", $Adapter) }
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep $Name $args
}

function Invoke-ReasoningEval {
    param([string]$Name, [string]$Model, [string]$Adapter, [string]$Benchmark, [string]$Destination)
    $summary = Join-Path $Destination "summary.json"
    if (-not (Should-Run $summary)) { return }
    $args = @(
        ".\scripts\phase_9_student_distillation\evaluate_general_reasoning_v212.py",
        "--model", $Model,
        "--benchmark", $Benchmark,
        "--output-dir", $Destination,
        "--max-new-tokens", "32",
        "--minimum-cases", "100",
        "--resume"
    )
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $args += @("--adapter", $Adapter) }
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep $Name $args
}

function Invoke-Training {
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
        "--max-sequence-length", "512",
        "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
        "--max-completion-tokens", "32",
        "--first-answer-token-weight", "1.5",
        "--evidence-anchor-token-weight", "1.5",
        "--abstention-token-weight", "1.75",
        "--eos-loss-weight", "1.3",
        "--eval-steps", "40",
        "--seed", "$Seed",
        "--pipeline-revision", "phase9_general_reasoning_v2_1_2"
    )
    if ([string]::IsNullOrWhiteSpace($InitialAdapter)) {
        $args += @("--lora", "--lora-rank", "$LoraRank")
    } else {
        $args += @("--initial-adapter", $InitialAdapter)
    }
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep $Name $args
}

function Assert-Schedule {
    if ($AllowAggressiveTraining) { return }
    $maxLr = [Math]::Max($StageALearningRate, [Math]::Max($StageB1LearningRate, $StageB2LearningRate))
    $total = $StageASteps + $StageB1Steps + $StageB2Steps
    if ($maxLr -gt 3e-5) {
        throw "v2.1.2 guard: learning rate $maxLr exceeds 3e-5. Use -AllowAggressiveTraining only for an explicit ablation."
    }
    if ($total -gt 600) {
        throw "v2.1.2 guard: total steps $total exceed 600. Use -AllowAggressiveTraining only for an explicit ablation."
    }
}

Assert-Schedule
Ensure-Path $ConfigPath "v2.1.2 config"
Ensure-Path $StudentModel "student/base checkpoint"
if (-not [string]::IsNullOrWhiteSpace($StudentAdapter)) { Ensure-Path $StudentAdapter "input student adapter" }
Ensure-Path $OriginalCorpusRoot "original Phase 9 corpus"
Ensure-Path $V211CorpusRoot "v2.1.1 procedural corpus"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$OriginalKnowledge = Join-Path $OriginalCorpusRoot "knowledge.jsonl"
$OriginalProcedures = Join-Path $OriginalCorpusRoot "procedures.jsonl"
$OriginalTrain = Join-Path $OriginalCorpusRoot "reviewed_splits\train.reviewed.jsonl"
$OriginalDev = Join-Path $OriginalCorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$OriginalTest = Join-Path $OriginalCorpusRoot "reviewed_splits\test.reviewed.jsonl"
$V211Knowledge = Join-Path $V211CorpusRoot "knowledge.jsonl"
$V211Procedures = Join-Path $V211CorpusRoot "procedures.jsonl"
$V211Train = Join-Path $V211CorpusRoot "reviewed_splits\train.reviewed.jsonl"
$V211Dev = Join-Path $V211CorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$V211Test = Join-Path $V211CorpusRoot "reviewed_splits\test.reviewed.jsonl"

if ($RegenerateDataset -or -not (Test-Path (Join-Path $ReasoningCorpusRoot "manifests\dataset_manifest.json"))) {
    Invoke-PythonStep "[v2.1.2 Data] Generate broad general-reasoning corpus" @(
        ".\scripts\phase_9_student_distillation\generate_general_reasoning_v212.py",
        "--output-root", $ReasoningCorpusRoot,
        "--original-train", $OriginalTrain,
        "--v211-train", $V211Train,
        "--seed", "$Seed",
        "--force"
    )
}
Invoke-PythonStep "[v2.1.2 Data] Validate general-reasoning corpus" @(
    ".\scripts\phase_9_student_distillation\validate_general_reasoning_v212.py",
    "--corpus-root", $ReasoningCorpusRoot,
    "--require-rows", "10000"
)

$ReasoningTrainA = Join-Path $ReasoningCorpusRoot "curriculum_views\train_stage_a_reasoning_core.reviewed.jsonl"
$ReasoningTrainB = Join-Path $ReasoningCorpusRoot "curriculum_views\train_stage_b_generalization_mix.reviewed.jsonl"
$ReasoningDev = Join-Path $ReasoningCorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$ReasoningTest = Join-Path $ReasoningCorpusRoot "reviewed_splits\test.reviewed.jsonl"

# Freeze retrieval infrastructure for the two RAG preservation axes.
$OriginalDb = Join-Path $OutputRoot "original_phase9.sqlite3"
$V211Db = Join-Path $OutputRoot "procedural_v211.sqlite3"
if (-not (Test-Path $OriginalDb)) {
    Invoke-PythonStep "[Retrieval] Build original Phase 9 index" @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py", "--knowledge", $OriginalKnowledge, "--procedures", $OriginalProcedures, "--output", $OriginalDb
    )
}
if (-not (Test-Path $V211Db)) {
    Invoke-PythonStep "[Retrieval] Build v2.1.1 procedural index" @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py", "--knowledge", $V211Knowledge, "--procedures", $V211Procedures, "--output", $V211Db
    )
}
$OriginalCalibration = Join-Path $OutputRoot "original_calibration_query_aware_v21.json"
$V211Calibration = Join-Path $OutputRoot "v211_calibration_query_aware_v21.json"
if (-not (Test-Path $OriginalCalibration)) {
    Invoke-PythonStep "[Retrieval] Freeze original DEV calibration" @(
        ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py", "--benchmark", $OriginalDev, "--knowledge-db", $OriginalDb, "--output", $OriginalCalibration, "--retrieval-selection-policy", "query_aware_v21", "--require-research-ready"
    )
}
if (-not (Test-Path $V211Calibration)) {
    Invoke-PythonStep "[Retrieval] Freeze v2.1.1 DEV calibration" @(
        ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py", "--benchmark", $V211Dev, "--knowledge-db", $V211Db, "--output", $V211Calibration, "--retrieval-selection-policy", "query_aware_v21", "--require-research-ready"
    )
}

function Invoke-AllDev {
    param([string]$Label, [string]$Adapter, [string]$Stem)
    Invoke-RagEval "[$Label] Original Phase 9 DEV" $StudentModel $Adapter $OriginalDev $OriginalDb $OriginalCalibration (Join-Path $OutputRoot "${Stem}_original")
    Invoke-RagEval "[$Label] v2.1.1 procedural DEV" $StudentModel $Adapter $V211Dev $V211Db $V211Calibration (Join-Path $OutputRoot "${Stem}_challenge")
    Invoke-ReasoningEval "[$Label] v2.1.2 general reasoning DEV" $StudentModel $Adapter $ReasoningDev (Join-Path $OutputRoot "${Stem}_reasoning")
}

Invoke-AllDev "Candidate 0 / input student" $StudentAdapter "baseline"
if ($SkipTraining) {
    Write-Host ""
    Write-Host "v2.1.2 baseline-only workflow complete. No weights were changed."
    Write-Host "Output: $OutputRoot"
    exit 0
}

$StageA = Join-Path $OutputRoot "train_stage_a_reasoning_core"
Invoke-Training "[Train A] Broad general-reasoning core" $ReasoningTrainA $ReasoningDev $StageA $StageALearningRate $StageASteps $StudentAdapter
$StageAAdapter = Join-Path $StageA "adapter"
Ensure-Path $StageAAdapter "Stage A adapter"
Invoke-AllDev "Stage A" $StageAAdapter "stage_a"

$StageB1 = Join-Path $OutputRoot "train_stage_b1_generalization_mix"
Invoke-Training "[Train B1] Generalization + preservation mix" $ReasoningTrainB $ReasoningDev $StageB1 $StageB1LearningRate $StageB1Steps $StageAAdapter
$StageB1Adapter = Join-Path $StageB1 "adapter"
Ensure-Path $StageB1Adapter "Stage B1 adapter"
Invoke-AllDev "Stage B1" $StageB1Adapter "stage_b1"

$StageB2 = Join-Path $OutputRoot "train_stage_b2_generalization_mix"
Invoke-Training "[Train B2] Lower-rate mixed consolidation" $ReasoningTrainB $ReasoningDev $StageB2 $StageB2LearningRate $StageB2Steps $StageB1Adapter
$StageB2Adapter = Join-Path $StageB2 "adapter"
Ensure-Path $StageB2Adapter "Stage B2 adapter"
Invoke-AllDev "Stage B2" $StageB2Adapter "stage_b2"

$SelectionJson = Join-Path $OutputRoot "general_reasoning_v212_selection.json"
$SelectionMd = Join-Path $OutputRoot "GENERAL_REASONING_V212_SELECTION.md"
Invoke-PythonStep "[Select] Generalization benefit vs preservation cost" @(
    ".\scripts\phase_9_student_distillation\select_general_reasoning_v212.py",
    "--config", $ConfigPath,
    "--baseline-original", (Join-Path $OutputRoot "baseline_original\summary.json"),
    "--baseline-challenge", (Join-Path $OutputRoot "baseline_challenge\summary.json"),
    "--baseline-reasoning", (Join-Path $OutputRoot "baseline_reasoning\summary.json"),
    "--baseline-adapter", $StudentAdapter,
    "--stage-a-original", (Join-Path $OutputRoot "stage_a_original\summary.json"),
    "--stage-a-challenge", (Join-Path $OutputRoot "stage_a_challenge\summary.json"),
    "--stage-a-reasoning", (Join-Path $OutputRoot "stage_a_reasoning\summary.json"),
    "--stage-a-adapter", $StageAAdapter,
    "--stage-b1-original", (Join-Path $OutputRoot "stage_b1_original\summary.json"),
    "--stage-b1-challenge", (Join-Path $OutputRoot "stage_b1_challenge\summary.json"),
    "--stage-b1-reasoning", (Join-Path $OutputRoot "stage_b1_reasoning\summary.json"),
    "--stage-b1-adapter", $StageB1Adapter,
    "--stage-b2-original", (Join-Path $OutputRoot "stage_b2_original\summary.json"),
    "--stage-b2-challenge", (Join-Path $OutputRoot "stage_b2_challenge\summary.json"),
    "--stage-b2-reasoning", (Join-Path $OutputRoot "stage_b2_reasoning\summary.json"),
    "--stage-b2-adapter", $StageB2Adapter,
    "--output-json", $SelectionJson,
    "--output-md", $SelectionMd
)

$Selection = Get-Content -Raw $SelectionJson | ConvertFrom-Json
$SelectedName = [string]$Selection.selected_candidate
$SelectedAdapter = [string]$Selection.selected_adapter
Write-Host ""
Write-Host "DEV-selected v2.1.2 candidate: $SelectedName"
if ([string]::IsNullOrWhiteSpace($SelectedAdapter)) { Write-Host "Selected path uses the input model without an adapter." } else { Write-Host "Selected adapter: $SelectedAdapter" }

if ($RunSealedTest) {
    Set-Content -Path (Join-Path $OutputRoot "SEALED_TEST_POLICY.txt") -Value @"
Selected on DEV: $SelectedName
Original TEST uses frozen DEV calibration: $OriginalCalibration
v2.1.1 TEST uses frozen DEV calibration: $V211Calibration
v2.1.2 reasoning TEST is direct prompt-contained reasoning.
No TEST-based recalibration, training, or candidate switching is allowed.
"@
    Invoke-RagEval "[SEALED] selected / original Phase 9 TEST" $StudentModel $SelectedAdapter $OriginalTest $OriginalDb $OriginalCalibration (Join-Path $OutputRoot "sealed_original_test") -RequireResearchReady
    Invoke-RagEval "[SEALED] selected / v2.1.1 procedural TEST" $StudentModel $SelectedAdapter $V211Test $V211Db $V211Calibration (Join-Path $OutputRoot "sealed_v211_test") -RequireResearchReady
    Invoke-ReasoningEval "[SEALED] selected / v2.1.2 reasoning TEST" $StudentModel $SelectedAdapter $ReasoningTest (Join-Path $OutputRoot "sealed_reasoning_v212_test")
}

Write-Host ""
Write-Host "v2.1.2 general-reasoning workflow complete."
Write-Host "Output: $OutputRoot"
Write-Host "Review GENERAL_REASONING_V212_SELECTION.md before architecture optimization."

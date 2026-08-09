param(
    [Parameter(Mandatory=$true)][string]$StudentModel,
    [string]$StudentAdapter = "",
    [string]$OutputRoot = ".\artifacts\phase_9_procedural_recovery_v21_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string]$OriginalCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$RecoveryCorpusRoot = ".\data\phase_9_student_distillation\procedural_recovery_v21",
    [int]$HardCollisionSteps = 250,
    [int]$ConsolidationSteps = 350,
    [double]$InitialLearningRate = 3e-5,
    [double]$ConsolidationLearningRate = 2e-5,
    [int]$LoraRank = 16,
    [int]$MaximumEvidenceTokens = 384,
    [int]$Seed = 21042,
    [switch]$RegenerateDataset,
    [switch]$SkipTraining,
    [switch]$RunSealedTest,
    [switch]$RequireCuda,
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
        [string]$SelectionPolicy,
        [string]$Conditions = "oracle_rag,real_rag"
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
        "--require-research-ready"
    )
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
        "--first-answer-token-weight", "2.25",
        "--evidence-anchor-token-weight", "2.25",
        "--abstention-token-weight", "1.75",
        "--eos-loss-weight", "1.5",
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
    Invoke-PythonStep "[v2.1 Data] Generate retrieval-collision procedural recovery corpus" @(
        ".\scripts\phase_9_student_distillation\generate_procedural_recovery_v21.py",
        "--output-root", $RecoveryCorpusRoot,
        "--original-train", $OriginalTrain,
        "--seed", "$Seed"
    )
}
Invoke-PythonStep "[v2.1 Data] Validate procedural recovery corpus" @(
    ".\scripts\phase_9_student_distillation\validate_rag_reasoning_corpus.py",
    "--corpus-root", $RecoveryCorpusRoot
)
Invoke-PythonStep "[Data audit] Record historical v1.7.5 procedural gold inconsistencies (informational)" @(
    ".\scripts\phase_9_student_distillation\audit_procedural_gold_consistency_v21.py",
    "--input", $OriginalTrain,
    "--output", (Join-Path $OutputRoot "v175_train_procedural_gold_audit.json")
)
Invoke-PythonStep "[Data audit] Require clean v2.1 procedural recovery gold" @(
    ".\scripts\phase_9_student_distillation\audit_procedural_gold_consistency_v21.py",
    "--input", (Join-Path $RecoveryCorpusRoot "reviewed_splits\train.reviewed.jsonl"),
    "--output", (Join-Path $OutputRoot "v21_train_procedural_gold_audit.json"),
    "--require-clean"
)

$RecoveryKnowledge = Join-Path $RecoveryCorpusRoot "knowledge.jsonl"
$RecoveryProcedures = Join-Path $RecoveryCorpusRoot "procedures.jsonl"
$RecoveryHardTrain = Join-Path $RecoveryCorpusRoot "curriculum_views\train_answerable_hard.reviewed.jsonl"
$RecoveryMixTrain = Join-Path $RecoveryCorpusRoot "curriculum_views\train_recovery_mix.reviewed.jsonl"
$RecoveryDev = Join-Path $RecoveryCorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$RecoveryTest = Join-Path $RecoveryCorpusRoot "reviewed_splits\test.reviewed.jsonl"

$OriginalDb = Join-Path $OutputRoot "original_phase9.sqlite3"
$RecoveryDb = Join-Path $OutputRoot "procedural_v21.sqlite3"
if (-not (Test-Path $OriginalDb)) {
    Invoke-PythonStep "[Retrieval] Build original Phase 9 index" @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py",
        "--knowledge", $OriginalKnowledge,
        "--procedures", $OriginalProcedures,
        "--output", $OriginalDb
    )
}
if (-not (Test-Path $RecoveryDb)) {
    Invoke-PythonStep "[Retrieval] Build v2.1 procedural index" @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py",
        "--knowledge", $RecoveryKnowledge,
        "--procedures", $RecoveryProcedures,
        "--output", $RecoveryDb
    )
}
$CollisionAuditJson = Join-Path $OutputRoot "original_dev_procedural_collision_audit.json"
if (-not (Test-Path $CollisionAuditJson)) {
    Invoke-PythonStep "[Retrieval] Audit original DEV sibling-record collisions" @(
        ".\scripts\phase_9_student_distillation\audit_procedural_retrieval_collisions.py",
        "--benchmark", $OriginalDev,
        "--knowledge-db", $OriginalDb,
        "--output-json", $CollisionAuditJson,
        "--output-md", (Join-Path $OutputRoot "ORIGINAL_DEV_PROCEDURAL_COLLISION_AUDIT.md")
    )
}

$OriginalCalibrationLegacy = Join-Path $OutputRoot "original_calibration_legacy.json"
$OriginalCalibrationV21 = Join-Path $OutputRoot "original_calibration_query_aware_v21.json"
$RecoveryCalibrationV21 = Join-Path $OutputRoot "procedural_calibration_query_aware_v21.json"
foreach ($packet in @(
    @{Name="[Retrieval] Calibrate original DEV / legacy"; Benchmark=$OriginalDev; Db=$OriginalDb; Out=$OriginalCalibrationLegacy; Policy="legacy"},
    @{Name="[Retrieval] Calibrate original DEV / query-aware v2.1"; Benchmark=$OriginalDev; Db=$OriginalDb; Out=$OriginalCalibrationV21; Policy="query_aware_v21"},
    @{Name="[Retrieval] Calibrate procedural challenge DEV / query-aware v2.1"; Benchmark=$RecoveryDev; Db=$RecoveryDb; Out=$RecoveryCalibrationV21; Policy="query_aware_v21"}
)) {
    if (-not (Test-Path $packet.Out)) {
        Invoke-PythonStep $packet.Name @(
            ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py",
            "--benchmark", $packet.Benchmark,
            "--knowledge-db", $packet.Db,
            "--output", $packet.Out,
            "--retrieval-selection-policy", $packet.Policy,
            "--require-research-ready"
        )
    }
}

$RetrievalAudit = Join-Path $OutputRoot "procedural_retrieval_query_aware_v21.json"
if (-not (Test-Path $RetrievalAudit)) {
    Invoke-PythonStep "[Retrieval] Verify v2.1 support recall" @(
        ".\scripts\phase_9_student_distillation\audit_rag_retrieval_coverage.py",
        "--benchmark", $RecoveryDev,
        "--knowledge-db", $RecoveryDb,
        "--output", $RetrievalAudit,
        "--retrieval-selection-policy", "query_aware_v21",
        "--require-complete-support-recall"
    )
}

# A/B the SAME weights before training. This tells us how much of the 0/150
# failure is prompt/retrieval collision versus a learned procedural deficit.
$BaselineLegacy = Join-Path $OutputRoot "baseline_original_legacy"
$BaselineV21 = Join-Path $OutputRoot "baseline_original_query_aware_v21"
$BaselineChallenge = Join-Path $OutputRoot "baseline_procedural_challenge_v21"
Invoke-Eval "[Baseline] Current student / original DEV / legacy retrieval" $StudentModel $StudentAdapter $OriginalDev $OriginalDb $OriginalCalibrationLegacy $BaselineLegacy "legacy"
Invoke-Eval "[Baseline] SAME student / original DEV / query-aware v2.1" $StudentModel $StudentAdapter $OriginalDev $OriginalDb $OriginalCalibrationV21 $BaselineV21 "query_aware_v21"
Invoke-Eval "[Baseline] SAME student / v2.1 challenge DEV" $StudentModel $StudentAdapter $RecoveryDev $RecoveryDb $RecoveryCalibrationV21 $BaselineChallenge "query_aware_v21"

$DiagnosisRoot = Join-Path $OutputRoot "diagnosis"
New-Item -ItemType Directory -Force -Path $DiagnosisRoot | Out-Null
if (-not (Test-Path (Join-Path $DiagnosisRoot "legacy_transfer_gap.json"))) {
    Invoke-PythonStep "[Diagnosis] Classify Oracle -> Real procedural failures under legacy retrieval" @(
        ".\scripts\phase_9_student_distillation\diagnose_procedural_transfer_gap.py",
        "--results", (Join-Path $BaselineLegacy "results.jsonl"),
        "--output-json", (Join-Path $DiagnosisRoot "legacy_transfer_gap.json"),
        "--output-md", (Join-Path $DiagnosisRoot "LEGACY_TRANSFER_GAP.md")
    )
}
if (-not (Test-Path (Join-Path $DiagnosisRoot "query_aware_transfer_gap.json"))) {
    Invoke-PythonStep "[Diagnosis] Reclassify with v2.1 retrieval, same weights" @(
        ".\scripts\phase_9_student_distillation\diagnose_procedural_transfer_gap.py",
        "--results", (Join-Path $BaselineV21 "results.jsonl"),
        "--output-json", (Join-Path $DiagnosisRoot "query_aware_transfer_gap.json"),
        "--output-md", (Join-Path $DiagnosisRoot "QUERY_AWARE_TRANSFER_GAP.md")
    )
}

if (-not $SkipTraining) {
    # One focused rank-16 repair path, not another wide LR/rank sweep.
    # Stage A teaches exact evidence binding under sibling collisions.
    # Stage B consolidates on the full v2.1 curriculum + train-only preservation slice.
    $StageA = Join-Path $OutputRoot "train_stage_a_collision_binding"
    $StageB = Join-Path $OutputRoot "train_stage_b_consolidation"
    Invoke-RecoveryTraining "[Train A] Collision/evidence-binding repair" $RecoveryHardTrain $RecoveryDev $StageA $InitialLearningRate $HardCollisionSteps $StudentAdapter
    $StageAAdapter = Join-Path $StageA "adapter"
    Ensure-Path $StageAAdapter "Stage A recovery adapter"
    Invoke-RecoveryTraining "[Train B] Procedural consolidation + behavior preservation" $RecoveryMixTrain $RecoveryDev $StageB $ConsolidationLearningRate $ConsolidationSteps $StageAAdapter
    $RecoveredAdapter = Join-Path $StageB "adapter"
    Ensure-Path $RecoveredAdapter "final v2.1 recovery adapter"

    $RecoveredOriginal = Join-Path $OutputRoot "recovered_original_query_aware_v21"
    $RecoveredChallenge = Join-Path $OutputRoot "recovered_procedural_challenge_v21"
    Invoke-Eval "[Recovered] Original Phase 9 DEV / query-aware v2.1" $StudentModel $RecoveredAdapter $OriginalDev $OriginalDb $OriginalCalibrationV21 $RecoveredOriginal "query_aware_v21"
    Invoke-Eval "[Recovered] v2.1 procedural challenge DEV" $StudentModel $RecoveredAdapter $RecoveryDev $RecoveryDb $RecoveryCalibrationV21 $RecoveredChallenge "query_aware_v21"

    Invoke-PythonStep "[Compare] Separate retrieval gains from learned training gains" @(
        ".\scripts\phase_9_student_distillation\compare_procedural_recovery_v21.py",
        "--legacy-baseline", (Join-Path $BaselineLegacy "summary.json"),
        "--query-aware-baseline", (Join-Path $BaselineV21 "summary.json"),
        "--recovered", (Join-Path $RecoveredOriginal "summary.json"),
        "--challenge-baseline", (Join-Path $BaselineChallenge "summary.json"),
        "--challenge-recovered", (Join-Path $RecoveredChallenge "summary.json"),
        "--output-json", (Join-Path $OutputRoot "procedural_recovery_comparison.json"),
        "--output-md", (Join-Path $OutputRoot "PROCEDURAL_RECOVERY_COMPARISON.md")
    )

    if ($RunSealedTest) {
        $OriginalTestCalibration = Join-Path $OutputRoot "original_test_uses_dev_calibration.txt"
        Set-Content -Path $OriginalTestCalibration -Value "Sealed test uses the frozen DEV calibration: $OriginalCalibrationV21"
        Invoke-Eval "[SEALED] Original Phase 9 TEST" $StudentModel $RecoveredAdapter $OriginalTest $OriginalDb $OriginalCalibrationV21 (Join-Path $OutputRoot "sealed_original_test") "query_aware_v21"
        Invoke-Eval "[SEALED] v2.1 procedural challenge TEST" $StudentModel $RecoveredAdapter $RecoveryTest $RecoveryDb $RecoveryCalibrationV21 (Join-Path $OutputRoot "sealed_procedural_v21_test") "query_aware_v21"
    }
}

Write-Host ""
Write-Host "v2.1 procedural recovery workflow complete."
Write-Host "Output: $OutputRoot"
Write-Host "Do NOT start Student Optimization v2 depth/width/sparsity work until this recovery is reviewed and approved."

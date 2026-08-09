<#
.SYNOPSIS
Run the Phase 9 grounded-recovery capability ladder with the v1.7.5 large supported/procedural corrective corpus.

v1.7.7 refreshes derived SFT selections under -ReuseExisting and prevents Stage-B baseline decode/prompt-integrity warnings from poisoning Stage-D candidate validity.

.DESCRIPTION
The workflow proves one warm 344M depth-copy student before any narrow-model
sweep. The order is teacher Oracle preflight, untrained warm Oracle baseline,
tiny deliberate overfit, compact warm-SFT sweep, full Oracle/Real evaluation,
and only then optional progressive KD. Narrow architecture comparison remains
locked until the ladder JSON reports architecture_sweep_unlocked=true.
#>
param(
    [ValidateSet("Preflight", "Overfit", "Sweep", "Full")][string]$Mode = "Preflight",
    [string]$RagReasoningCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$TeacherModel = "",
    [string]$ReviewedTeacherData = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\teacher_candidates.auto_verified.jsonl",
    [string]$Knowledge = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\knowledge.jsonl",
    [string]$Procedures = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\procedures.jsonl",
    [string]$FoundationPrompts = ".\data\phase_9_student_distillation\foundation\prompts.foundation.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_9_grounded_recovery_v1_7_5",
    [int]$MaximumEvidenceTokens = 384,
    [int]$Seed = 42,
    [ValidateSet("balanced", "support_procedure_recovery")][string]$BehaviorProfile = "support_procedure_recovery",
    [int]$OverfitRows = 18,
    [int]$OverfitRepeats = 12,
    [int]$OverfitSteps = 300,
    [double]$OverfitMinimumScore = 0.80,
    [double]$OverfitMinimumImprovement = 0.25,
    [switch]$UseLora,
    [switch]$Teacher4Bit,
    [switch]$RunKD,
    [switch]$SkipRetrievalCoverageAudit,
    [switch]$AllowUnreleasedTeacher,
    [switch]$ReuseExisting
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

if (Test-Path (Join-Path $RagReasoningCorpusRoot "manifests\dataset_manifest.json")) {
    Write-Host ""
    Write-Host "[Corpus] Validating the integrated RAG reasoning corpus..."
    & python .\scripts\phase_9_student_distillation\validate_rag_reasoning_corpus.py --corpus-root $RagReasoningCorpusRoot
    if ($LASTEXITCODE -ne 0) { throw "Integrated RAG reasoning corpus validation failed." }
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

function Ensure-File {
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

function Invoke-RagEvaluation {
    param(
        [string]$Name,
        [string]$Model,
        [string]$Adapter,
        [string]$Benchmark,
        [string]$Destination,
        [string]$Conditions,
        [switch]$ResearchReady
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
        "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
        "--max-new-tokens", "32",
        "--minimum-cases-per-condition", "1",
        "--minimum-supported-cases", "1",
        "--minimum-unsupported-cases", "1"
    )
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $args += @("--adapter", $Adapter) }
    if ($ResearchReady) { $args += "--require-research-ready" } else { $args += "--smoke-test" }
    Invoke-PythonStep $Name $args
}

function Invoke-SftTraining {
    param(
        [string]$Name,
        [string]$TrainingFile,
        [string]$DevFile,
        [string]$Destination,
        [double]$LearningRate,
        [int]$MaxSteps,
        [int]$LoraRank = 16
    )
    $manifest = Join-Path $Destination "training_manifest.json"
    if (-not (Should-Run $manifest)) { return }
    $args = @(
        ".\scripts\phase_9_student_distillation\train_student.py",
        "--student", $WarmInitialization,
        "--objective", "sft",
        "--train-file", $TrainingFile,
        "--dev-file", $DevFile,
        "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
        "--output-dir", $Destination,
        "--max-steps", "$MaxSteps",
        "--learning-rate", "$LearningRate",
        "--gradient-accumulation-steps", "32",
        "--lora-rank", "$LoraRank",
        "--first-answer-token-weight", "2.0",
        "--evidence-anchor-token-weight", "1.75",
        "--abstention-token-weight", "2.0",
        "--eos-loss-weight", "1.5",
        "--seed", "$Seed"
    )
    if ($UseLora) { $args += "--lora" }
    Invoke-PythonStep $Name $args
}

function Get-TrainedArtifact {
    param([string]$TrainingRoot)
    if ($UseLora) {
        return [pscustomobject]@{ Model = $WarmInitialization; Adapter = (Join-Path $TrainingRoot "adapter") }
    }
    return [pscustomobject]@{ Model = (Join-Path $TrainingRoot "model"); Adapter = "" }
}

if (-not $AllowUnreleasedTeacher) {
    Invoke-PythonStep "[Gate] Requiring the accepted Phase 8 teacher..." @(
        ".\scripts\common\check_human_gate.py", "--phase", "8"
    )
}
if ([string]::IsNullOrWhiteSpace($TeacherModel)) {
    $TeacherModel = python .\scripts\common\resolve_reference_model.py
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($TeacherModel)) {
        throw "Could not resolve the selected teacher. Supply -TeacherModel explicitly."
    }
}
Ensure-File $TeacherModel "teacher checkpoint"
Ensure-File $ReviewedTeacherData "reviewed teacher data"
Ensure-File $Knowledge "knowledge JSONL"
Ensure-File $Procedures "procedure JSONL"
Ensure-File $FoundationPrompts "foundation prompts"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$WarmInitialization = Join-Path $OutputRoot "warm_344m_initialization"
$Database = Join-Path $OutputRoot "knowledge.sqlite3"
$DatasetRoot = Join-Path $OutputRoot "dataset"
$TrainFile = Join-Path $DatasetRoot "train.jsonl"
$DevFile = Join-Path $DatasetRoot "dev.jsonl"
$TestFile = Join-Path $DatasetRoot "test.jsonl"
$CurriculumFile = Join-Path $DatasetRoot "train_curriculum.jsonl"
$FoundationTrain = Join-Path $DatasetRoot "foundation_train.jsonl"
$FoundationCurriculum = Join-Path $DatasetRoot "foundation_curriculum.jsonl"
$MixedCurriculum = Join-Path $DatasetRoot "train_mixed_curriculum.jsonl"
$BalancedCurriculum = Join-Path $DatasetRoot "train_grounded_balanced.jsonl"
$Calibration = Join-Path $OutputRoot "evidence_gate_calibration.json"
$RetrievalAudit = Join-Path $OutputRoot "retrieval_support_audit.json"
$LadderJson = Join-Path $OutputRoot "grounded_recovery_ladder.json"
$LadderReport = Join-Path $OutputRoot "GROUNDED_RECOVERY_LADDER.md"

if (-not (Test-Path (Join-Path $WarmInitialization "student_creation_manifest.json"))) {
    Invoke-PythonStep "[Student] Creating the warm 344M depth-copy initialization..." @(
        ".\scripts\phase_9_student_distillation\create_student.py",
        "--source", $TeacherModel,
        "--output", $WarmInitialization,
        "--mode", "depth_copy",
        "--num-hidden-layers", "12",
        "--profile-name", "warm_344m"
    )
}
if (-not (Test-Path $Database)) {
    Invoke-PythonStep "[Index] Building the FTS5 knowledge index..." @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py",
        "--knowledge", $Knowledge,
        "--procedures", $Procedures,
        "--output", $Database
    )
}
if (-not (Test-Path $TrainFile)) {
    Invoke-PythonStep "[Data] Building leakage-safe real splits..." @(
        ".\scripts\phase_9_student_distillation\build_pilot_dataset.py",
        "--reviewed-input", $ReviewedTeacherData,
        "--output-dir", $DatasetRoot,
        "--seed", "$Seed",
        "--split-policy", "intended",
        "--require-research-ready"
    )
}
if (-not (Test-Path $CurriculumFile)) {
    Invoke-PythonStep "[Data] Building the reviewed behavior curriculum..." @(
        ".\scripts\phase_9_student_distillation\build_curriculum_dataset.py",
        "--input", $TrainFile,
        "--output", $CurriculumFile,
        "--seed", "$Seed",
        "--canonical-repeats", "2",
        "--prompt-variants", "2",
        "--distractor-variants", "2",
        "--missing-evidence-variants", "2",
        "--injection-variants", "1",
        "--social-repeats", "2"
    )
}
if (-not (Test-Path $FoundationTrain)) {
    Invoke-PythonStep "[Data] Materializing the fictional behavior foundation..." @(
        ".\scripts\phase_9_student_distillation\build_foundation_dataset.py",
        "--prompts", $FoundationPrompts,
        "--output", $FoundationTrain
    )
}
if (-not (Test-Path $FoundationCurriculum)) {
    Invoke-PythonStep "[Data] Expanding the behavior foundation..." @(
        ".\scripts\phase_9_student_distillation\build_curriculum_dataset.py",
        "--input", $FoundationTrain,
        "--output", $FoundationCurriculum,
        "--seed", "$Seed",
        "--canonical-repeats", "1",
        "--prompt-variants", "1",
        "--distractor-variants", "1",
        "--missing-evidence-variants", "1",
        "--injection-variants", "1",
        "--social-repeats", "2"
    )
}
if (-not (Test-Path $MixedCurriculum)) {
    Invoke-PythonStep "[Data] Mixing reviewed and foundation curricula..." @(
        ".\scripts\phase_9_student_distillation\merge_training_data.py",
        "--primary", $CurriculumFile,
        "--foundation", $FoundationCurriculum,
        "--output", $MixedCurriculum,
        "--primary-repeat", "3",
        "--foundation-repeat", "1",
        "--holdout", $DevFile,
        "--holdout", $TestFile,
        "--seed", "$Seed"
    )
}
if (-not (Test-Path $BalancedCurriculum)) {
    Invoke-PythonStep "[Data] Applying the v1.7.5 supported/procedural recovery mixture..." @(
        ".\scripts\phase_9_student_distillation\rebalance_behavior_curriculum.py",
        "--input", $MixedCurriculum,
        "--output", $BalancedCurriculum,
        "--profile", $BehaviorProfile,
        "--seed", "$Seed"
    )
}
if (-not $SkipRetrievalCoverageAudit -and -not (Test-Path $RetrievalAudit)) {
    Invoke-PythonStep "[Data] Auditing that DEV support is retrievable before model evaluation..." @(
        ".\scripts\phase_9_student_distillation\audit_rag_retrieval_coverage.py",
        "--benchmark", $DevFile,
        "--knowledge-db", $Database,
        "--output", $RetrievalAudit,
        "--require-complete-support-recall"
    )
}
if (-not (Test-Path $Calibration)) {
    Invoke-PythonStep "[Data] Calibrating Real-RAG retrieval on dev only..." @(
        ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py",
        "--benchmark", $DevFile,
        "--knowledge-db", $Database,
        "--output", $Calibration,
        "--require-research-ready"
    )
}

$TeacherOracle = Join-Path $OutputRoot "evaluation_teacher_oracle"
$WarmOracle = Join-Path $OutputRoot "evaluation_warm_initialization_oracle"
# Dev is the only selection/preflight benchmark. Test remains sealed until a
# final SFT/KD candidate has already been selected.
Invoke-RagEvaluation "[Stage A] Teacher with Oracle evidence on dev..." $TeacherModel "" $DevFile $TeacherOracle "oracle_rag,adversarial_passthrough_rag"
Invoke-RagEvaluation "[Stage B] Untrained warm student with Oracle evidence on dev..." $WarmInitialization "" $DevFile $WarmOracle "oracle_rag,adversarial_passthrough_rag"
Invoke-PythonStep "[Ladder] Recording teacher and initialization stages..." @(
    ".\scripts\phase_9_student_distillation\check_grounded_recovery_ladder.py",
    "--teacher-summary", (Join-Path $TeacherOracle "summary.json"),
    "--initialization-summary", (Join-Path $WarmOracle "summary.json"),
    "--out", $LadderReport,
    "--json-out", $LadderJson
)
$PreflightPacket = Get-Content -Raw -Path $LadderJson | ConvertFrom-Json
if ([string]$PreflightPacket.stages.A.status -ne "PASS") {
    throw "Teacher Oracle preflight failed. Fix the prompt/evidence/evaluator path before training any student. See $LadderReport"
}
if ($Mode -eq "Preflight") {
    Write-Host "Preflight complete: $LadderReport"
    exit 0
}

$OverfitRoot = Join-Path $OutputRoot "overfit_probe"
$OverfitTrain = Join-Path $OverfitRoot "train.jsonl"
$OverfitBenchmark = Join-Path $OverfitRoot "benchmark.jsonl"
$OverfitDataManifest = Join-Path $OverfitRoot "dataset_manifest.json"
New-Item -ItemType Directory -Force -Path $OverfitRoot | Out-Null
if (-not (Test-Path $OverfitTrain)) {
    Invoke-PythonStep "[Stage C] Building the tiny deliberate overfit probe..." @(
        ".\scripts\phase_9_student_distillation\build_overfit_probe.py",
        "--input", $BalancedCurriculum,
        "--train-output", $OverfitTrain,
        "--benchmark-output", $OverfitBenchmark,
        "--manifest", $OverfitDataManifest,
        "--rows", "$OverfitRows",
        "--repeat", "$OverfitRepeats",
        "--seed", "$Seed"
    )
}
$OverfitInitEval = Join-Path $OverfitRoot "evaluation_initialization"
Invoke-RagEvaluation "[Stage C] Evaluating the pre-training overfit baseline..." $WarmInitialization "" $OverfitBenchmark $OverfitInitEval "oracle_rag,adversarial_passthrough_rag"
$OverfitTraining = Join-Path $OverfitRoot "training"
Invoke-SftTraining "[Stage C] Deliberately overfitting the tiny probe..." $OverfitTrain $OverfitBenchmark $OverfitTraining 0.0001 $OverfitSteps 16
$OverfitArtifact = Get-TrainedArtifact $OverfitTraining
$OverfitTrainedEval = Join-Path $OverfitRoot "evaluation_trained_reloaded"
Invoke-RagEvaluation "[Stage C] Reloading and evaluating the trained overfit artifact..." $OverfitArtifact.Model $OverfitArtifact.Adapter $OverfitBenchmark $OverfitTrainedEval "oracle_rag,adversarial_passthrough_rag"
Invoke-PythonStep "[Stage C] Requiring the wiring test to pass..." @(
    ".\scripts\phase_9_student_distillation\check_grounded_recovery_ladder.py",
    "--teacher-summary", (Join-Path $TeacherOracle "summary.json"),
    "--initialization-summary", (Join-Path $WarmOracle "summary.json"),
    "--overfit-initialization-summary", (Join-Path $OverfitInitEval "summary.json"),
    "--overfit-trained-summary", (Join-Path $OverfitTrainedEval "summary.json"),
    "--overfit-training-manifest", (Join-Path $OverfitTraining "training_manifest.json"),
    "--minimum-overfit-score", "$OverfitMinimumScore",
    "--minimum-overfit-improvement", "$OverfitMinimumImprovement",
    "--out", $LadderReport,
    "--json-out", $LadderJson,
    "--require-overfit-pass"
)
if ($Mode -eq "Overfit") {
    Write-Host "Overfit wiring proof complete: $LadderReport"
    exit 0
}

$SweepRoot = Join-Path $OutputRoot "warm_sft_sweep"
New-Item -ItemType Directory -Force -Path $SweepRoot | Out-Null
$SweepSpecs = @(
    [pscustomobject]@{ Name = "lr5e5_s600"; LearningRate = 0.00005; Steps = 600; Rank = 16 },
    [pscustomobject]@{ Name = "lr5e5_s1200"; LearningRate = 0.00005; Steps = 1200; Rank = 16 },
    [pscustomobject]@{ Name = "lr1e4_s1200"; LearningRate = 0.0001; Steps = 1200; Rank = 16 }
)
$SelectionArgs = @(
    ".\scripts\phase_9_student_distillation\select_grounded_sft_candidate.py",
    "--initialization-summary", (Join-Path $WarmOracle "summary.json"),
    "--out", (Join-Path $SweepRoot "selection.json"),
    "--report", (Join-Path $SweepRoot "SELECTION_REPORT.md")
)
foreach ($Spec in $SweepSpecs) {
    $CandidateRoot = Join-Path $SweepRoot $Spec.Name
    $TrainingRoot = Join-Path $CandidateRoot "training"
    Invoke-SftTraining "[Stage D] SFT sweep $($Spec.Name)..." $BalancedCurriculum $DevFile $TrainingRoot $Spec.LearningRate $Spec.Steps $Spec.Rank
    $Artifact = Get-TrainedArtifact $TrainingRoot
    $EvaluationRoot = Join-Path $CandidateRoot "evaluation_oracle"
    Invoke-RagEvaluation "[Stage D] Dev Oracle evaluation $($Spec.Name)..." $Artifact.Model $Artifact.Adapter $DevFile $EvaluationRoot "oracle_rag,adversarial_passthrough_rag"
    $SelectionArgs += @("--candidate", "$($Spec.Name)=$(Join-Path $EvaluationRoot 'summary.json')")
}
$SweepSelectionPath = Join-Path $SweepRoot "selection.json"
if (Test-Path $SweepSelectionPath) {
    if (-not $ReuseExisting) {
        throw "Artifact already exists: $SweepSelectionPath. Use a fresh -OutputRoot or pass -ReuseExisting deliberately."
    }
    Write-Host "[Stage D] Recomputing the derived SFT selection from existing candidate summaries under the current selector policy..."
    Invoke-PythonStep "[Stage D] Refreshing the non-degenerate Oracle frontier..." $SelectionArgs
}
else {
    Invoke-PythonStep "[Stage D] Selecting only from a non-degenerate Oracle frontier..." $SelectionArgs
}
$Selection = Get-Content -Raw -Path $SweepSelectionPath | ConvertFrom-Json
if ($null -eq $Selection.selected) { throw "No valid warm-SFT Oracle frontier was found." }
$SelectedSftSummary = [string]$Selection.selected.summary
Invoke-PythonStep "[Stage D] Requiring full Oracle capability improvement..." @(
    ".\scripts\phase_9_student_distillation\check_grounded_recovery_ladder.py",
    "--teacher-summary", (Join-Path $TeacherOracle "summary.json"),
    "--initialization-summary", (Join-Path $WarmOracle "summary.json"),
    "--overfit-initialization-summary", (Join-Path $OverfitInitEval "summary.json"),
    "--overfit-trained-summary", (Join-Path $OverfitTrainedEval "summary.json"),
    "--overfit-training-manifest", (Join-Path $OverfitTraining "training_manifest.json"),
    "--sft-summary", $SelectedSftSummary,
    "--minimum-overfit-score", "$OverfitMinimumScore",
    "--minimum-overfit-improvement", "$OverfitMinimumImprovement",
    "--out", $LadderReport,
    "--json-out", $LadderJson,
    "--require-overfit-pass",
    "--require-oracle-improvement"
)
if ($Mode -eq "Sweep") {
    Write-Host "Warm SFT sweep complete: $(Join-Path $SweepRoot 'SELECTION_REPORT.md')"
    exit 0
}

$SelectedModel = [string]$Selection.selected.model
$SelectedAdapter = [string]$Selection.selected.adapter

# Stage E first compares refinements on DEV with the complete RAG path. This is
# the last model-selection point; test remains untouched until after this block.
$SftDevFull = Join-Path $OutputRoot "evaluation_selected_sft_dev_full"
Invoke-RagEvaluation "[Stage E] Full selected-SFT evaluation on DEV including Real RAG..." $SelectedModel $SelectedAdapter $DevFile $SftDevFull "closed_book,oracle_rag,real_rag,adversarial_rag,adversarial_passthrough_rag" -ResearchReady

$FinalDevSelection = Join-Path $OutputRoot "final_dev_selection.json"
$FinalDevReport = Join-Path $OutputRoot "FINAL_DEV_SELECTION.md"
$FinalSelectionArgs = @(
    ".\scripts\phase_9_student_distillation\select_grounded_sft_candidate.py",
    "--initialization-summary", (Join-Path $WarmOracle "summary.json"),
    "--candidate", "sft=$(Join-Path $SftDevFull 'summary.json')",
    "--out", $FinalDevSelection,
    "--report", $FinalDevReport
)

if ($RunKD) {
    $KdRoot = Join-Path $OutputRoot "progressive_kd_from_selected_sft"
    $KdManifest = Join-Path $KdRoot "training_manifest.json"
    if (Should-Run $KdManifest) {
        $args = @(
            ".\scripts\phase_9_student_distillation\train_student.py",
            "--student", $WarmInitialization,
            "--teacher", $TeacherModel,
            "--objective", "kd",
            "--train-file", $BalancedCurriculum,
            "--dev-file", $DevFile,
            "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
            "--output-dir", $KdRoot,
            "--max-steps", "600",
            "--learning-rate", "0.00005",
            "--kd-hard-weight-start", "0.75",
            "--kd-hard-weight-end", "0.25",
            "--kd-warmup-ratio", "0.15",
            "--teacher-confidence-floor", "0.02",
            "--first-answer-token-weight", "2.0",
            "--evidence-anchor-token-weight", "1.75",
            "--abstention-token-weight", "2.0",
            "--eos-loss-weight", "1.5",
            "--seed", "$Seed"
        )
        if (-not [string]::IsNullOrWhiteSpace($SelectedAdapter)) { $args += @("--initial-adapter", $SelectedAdapter) }
        if ($Teacher4Bit) { $args += "--teacher-4bit" }
        Invoke-PythonStep "[KD] Progressive KD from the proven SFT artifact..." $args
    }
    $KdArtifact = Get-TrainedArtifact $KdRoot
    $KdDevFull = Join-Path $OutputRoot "evaluation_progressive_kd_dev_full"
    Invoke-RagEvaluation "[KD] Full DEV KD evaluation..." $KdArtifact.Model $KdArtifact.Adapter $DevFile $KdDevFull "closed_book,oracle_rag,real_rag,adversarial_rag,adversarial_passthrough_rag" -ResearchReady
    $FinalSelectionArgs += @("--candidate", "progressive_kd=$(Join-Path $KdDevFull 'summary.json')")
}

if (Test-Path $FinalDevSelection) {
    if (-not $ReuseExisting) {
        throw "Artifact already exists: $FinalDevSelection. Use a fresh -OutputRoot or pass -ReuseExisting deliberately."
    }
    Write-Host "[Stage E] Recomputing the derived DEV selection under the current selector policy..."
    Invoke-PythonStep "[Stage E] Refreshing the final refinement selection on DEV only..." $FinalSelectionArgs
}
else {
    Invoke-PythonStep "[Stage E] Selecting the final refinement on DEV only..." $FinalSelectionArgs
}
$FinalSelection = Get-Content -Raw -Path $FinalDevSelection | ConvertFrom-Json
if ($null -eq $FinalSelection.selected) { throw "No valid final Phase 9 candidate was selected on dev." }
$FinalName = [string]$FinalSelection.selected.name
$FinalModel = [string]$FinalSelection.selected.model
$FinalAdapter = [string]$FinalSelection.selected.adapter

# SEALED TEST: only now evaluate the already-selected final candidate. Teacher
# and initialization are references, never tuning signals after this point.
$TeacherTest = Join-Path $OutputRoot "evaluation_teacher_test_final"
$WarmTest = Join-Path $OutputRoot "evaluation_warm_initialization_test_final"
$FinalTest = Join-Path $OutputRoot "evaluation_selected_final_test"
Invoke-RagEvaluation "[Sealed Test] Teacher reference..." $TeacherModel "" $TestFile $TeacherTest "closed_book,oracle_rag,real_rag,adversarial_rag,adversarial_passthrough_rag" -ResearchReady
Invoke-RagEvaluation "[Sealed Test] Warm initialization reference..." $WarmInitialization "" $TestFile $WarmTest "closed_book,oracle_rag,real_rag,adversarial_rag,adversarial_passthrough_rag" -ResearchReady
Invoke-RagEvaluation "[Sealed Test] Final selected candidate $FinalName..." $FinalModel $FinalAdapter $TestFile $FinalTest "closed_book,oracle_rag,real_rag,adversarial_rag,adversarial_passthrough_rag" -ResearchReady

$GateArgs = @(
    ".\scripts\phase_9_student_distillation\check_phase_9_gate.py",
    "--teacher-summary", (Join-Path $TeacherTest "summary.json"),
    "--initialization", "warm=$(Join-Path $WarmTest 'summary.json')",
    "--candidate", "$FinalName=$(Join-Path $FinalTest 'summary.json')"
)
Invoke-PythonStep "[Final] Building the Phase 9 assessment and pending human gate..." $GateArgs
Write-Host ""
Write-Host "Grounded recovery Full run completed."
Write-Host "Ladder: $LadderReport"
Write-Host "SFT sweep selection: $(Join-Path $SweepRoot 'SELECTION_REPORT.md')"
Write-Host "Final dev selection: $FinalDevReport"
Write-Host "Phase 9 gate: reports\PHASE_9_STOP_GATE.md"

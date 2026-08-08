<#
.SYNOPSIS
Run Phase 9B contextual memory reconstruction under the v1.7.5 large corrective-corpus contract.

.DESCRIPTION
Phase 9B consumes the human-approved Phase 9 student and teaches source-state-
conditioned memory use. v1.7.5 integrates a 3,600-row source-state reconstruction corpus with all five decisions, 600 counterfactual pairs, 10,800 curated DPO preferences, and 14,400 evaluator candidates.

DEV is used for strategy selection. Exactly one strategy is frozen before sealed
TEST is touched. Prepare remains available for generating project-specific
reconstruction candidates from the Phase 9 RAG corpus, but the integrated
reviewed corpus is the default Train/Evaluate/Full input.
#>
param(
    [ValidateSet("Prepare", "Train", "Evaluate", "Full")][string]$Mode = "Prepare",
    [string]$RagReasoningCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$MemoryReconstructionCorpusRoot = ".\data\phase_9b_memory_reconstruction\reconstruction_corpus_v2_large",
    [string]$Phase9ReferenceManifest = ".\artifacts\references\phase_9_student_reference.json",
    [string]$TeacherModel = "",
    [string]$SourcePrompts = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\prompts.all.jsonl",
    [string]$MemoryFile = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\knowledge.jsonl",
    [string]$ReviewedReconstructions = "",
    [string]$Config = ".\configs\phase_9b_memory_reconstruction.json",
    [string]$OutputRoot = ".\artifacts\phase_9b_memory_reconstruction",
    [int]$MaxSteps = 400,
    [int]$PreferenceMaxSteps = 200,
    [int]$Seed = 42,
    [switch]$UseLora,
    [switch]$Teacher4Bit,
    [switch]$SkipKD,
    [switch]$RunPreferenceAlignment,
    [switch]$SkipTeacherGeneration,
    [switch]$SmokeTest,
    [switch]$AllowUnapprovedPhase9,
    [switch]$ReuseExisting
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

# Backward-compatible manual overrides remain supported for prior experiments:
# .\data\phase_9_student_distillation\rag_reasoning_v2\prompts.all.jsonl
# .\data\phase_9_student_distillation\rag_reasoning_v2\knowledge.jsonl
# .\data\phase_9b_memory_reconstruction\reconstruction_corpus_v1

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

function Resolve-ManifestModel {
    param([string]$Manifest)
    if (-not (Test-Path $Manifest)) { throw "Missing selected-student manifest: $Manifest" }
    $packet = Get-Content -Raw -Path $Manifest | ConvertFrom-Json
    $model = [string]$packet.selected_model
    if ([string]::IsNullOrWhiteSpace($model) -or -not (Test-Path $model)) {
        throw "The manifest does not point to an available selected_model: $Manifest"
    }
    return $model
}

if (-not (Test-Path $Config)) { throw "Missing Phase 9B config: $Config" }

# Validate both first-class corpora before any expensive work.
if (Test-Path (Join-Path $RagReasoningCorpusRoot "manifests\dataset_manifest.json")) {
    Invoke-PythonStep "[Corpus] Validating Phase 9 evidence-reasoning corpus..." @(
        ".\scripts\phase_9_student_distillation\validate_rag_reasoning_corpus.py",
        "--corpus-root", $RagReasoningCorpusRoot
    )
}
if (Test-Path (Join-Path $MemoryReconstructionCorpusRoot "manifests\dataset_manifest.json")) {
    Invoke-PythonStep "[Corpus] Validating dedicated Phase 9B reconstruction corpus..." @(
        ".\scripts\phase_9b_memory_reconstruction\validate_memory_reconstruction_corpus.py",
        "--corpus-root", $MemoryReconstructionCorpusRoot
    )
    Invoke-PythonStep "[Corpus] Verifying reconstruction-evaluator calibration on DEV..." @(
        ".\scripts\phase_9b_memory_reconstruction\verify_memory_reconstruction_evaluator.py",
        "--corpus-root", $MemoryReconstructionCorpusRoot,
        "--split", "dev"
    )
}

if (-not $AllowUnapprovedPhase9) {
    Invoke-PythonStep "[Gate] Requiring the accepted Phase 9 student..." @(
        ".\scripts\common\check_human_gate.py", "--phase", "9"
    )
}
$StudentModel = Resolve-ManifestModel $Phase9ReferenceManifest

# The curated corpus is already reviewed. Train/Evaluate/Full default to it.
$IntegratedReviewed = Join-Path $MemoryReconstructionCorpusRoot "reviewed_memory_reconstruction.jsonl"
$UsingIntegratedCorpus = $false
if ([string]::IsNullOrWhiteSpace($ReviewedReconstructions) -and $Mode -ne "Prepare" -and -not $SmokeTest) {
    if (-not (Test-Path $IntegratedReviewed)) {
        throw "No -ReviewedReconstructions was supplied and the integrated Phase 9B corpus is unavailable: $IntegratedReviewed"
    }
    $ReviewedReconstructions = $IntegratedReviewed
    $UsingIntegratedCorpus = $true
    Write-Host "[Corpus] Using the integrated accepted reconstruction corpus: $ReviewedReconstructions"
} elseif (-not [string]::IsNullOrWhiteSpace($ReviewedReconstructions)) {
    if (-not (Test-Path $ReviewedReconstructions)) { throw "Missing reviewed Phase 9B JSONL: $ReviewedReconstructions" }
    try {
        $UsingIntegratedCorpus = ((Resolve-Path $ReviewedReconstructions).Path -eq (Resolve-Path $IntegratedReviewed).Path)
    } catch { $UsingIntegratedCorpus = $false }
}

$NeedsTeacher = (($Mode -eq "Prepare") -and -not $SkipTeacherGeneration) -or (($Mode -in @("Train", "Full")) -and -not $SkipKD)
if ($NeedsTeacher) {
    if ([string]::IsNullOrWhiteSpace($TeacherModel)) {
        $TeacherModel = python .\scripts\common\resolve_reference_model.py
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($TeacherModel)) {
            throw "Could not resolve the authoritative teacher. Supply -TeacherModel explicitly."
        }
    }
    if (-not (Test-Path $TeacherModel)) { throw "Missing authoritative teacher checkpoint: $TeacherModel" }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Candidates = Join-Path $OutputRoot "reconstruction_candidates.jsonl"
$CandidateReview = Join-Path $OutputRoot "reconstruction_candidates.review.md"
$TeacherCandidates = Join-Path $OutputRoot "teacher_reconstruction_candidates.jsonl"
$TeacherReview = Join-Path $OutputRoot "teacher_reconstruction_candidates.review.md"
$DatasetRoot = Join-Path $OutputRoot "dataset"
$TrainFile = Join-Path $DatasetRoot "train_combined.jsonl"
$DevFile = Join-Path $DatasetRoot "dev_combined.jsonl"
$DevCanonical = Join-Path $DatasetRoot "dev.jsonl"
$TestCanonical = Join-Path $DatasetRoot "test.jsonl"
$SftRoot = Join-Path $OutputRoot "student_reconstruction_sft"
$SftAdapter = Join-Path $SftRoot "adapter"
$KdRoot = Join-Path $OutputRoot "student_reconstruction_kd"
$KdAdapter = Join-Path $KdRoot "adapter"
$DpoRoot = Join-Path $OutputRoot "student_reconstruction_dpo"
$DpoAdapter = Join-Path $DpoRoot "adapter"
$DevEvaluationRoot = Join-Path $OutputRoot "evaluation_dev"
$TestEvaluationRoot = Join-Path $OutputRoot "evaluation_test_sealed"
$DevAssessment = Join-Path $OutputRoot "PHASE_9B_DEV_ASSESSMENT.json"
$DevGate = Join-Path $OutputRoot "PHASE_9B_DEV_SELECTION.md"
$DevSelection = Join-Path $OutputRoot "phase_9b_dev_selection.json"

function Prepare-ReconstructionReview {
    if (-not (Test-Path $SourcePrompts)) { throw "Missing Phase 9 source prompts: $SourcePrompts" }
    if (-not ($ReuseExisting -and (Test-Path $Candidates))) {
        $args = @(
            ".\scripts\phase_9b_memory_reconstruction\build_reconstruction_candidates.py",
            "--source-prompts", $SourcePrompts,
            "--output", $Candidates,
            "--review-packet", $CandidateReview
        )
        if (Test-Path $MemoryFile) { $args += @("--memory-file", $MemoryFile) }
        Invoke-PythonStep "[9B.0] Building project-specific source-state candidates..." $args
    }
    if (-not $SkipTeacherGeneration -and -not ($ReuseExisting -and (Test-Path $TeacherCandidates))) {
        $args = @(
            ".\scripts\phase_9b_memory_reconstruction\generate_teacher_reconstructions.py",
            "--teacher", $TeacherModel,
            "--candidates", $Candidates,
            "--output", $TeacherCandidates,
            "--review-packet", $TeacherReview,
            "--resume"
        )
        if ($Teacher4Bit) { $args += "--load-in-4bit" }
        Invoke-PythonStep "[9B.1] Generating teacher reconstruction candidates..." $args
    }
}

if ($Mode -eq "Prepare") {
    Prepare-ReconstructionReview
    Write-Host ""
    Write-Host "Phase 9B project-specific preparation completed."
    Write-Host "Review packet: $TeacherReview"
    Write-Host "The integrated 600-row corpus remains the default training set; use this packet only as an additional reviewed extension."
    exit 0
}

if ($SmokeTest) {
    $ReviewedReconstructions = ".\data\phase_9b_memory_reconstruction\reviewed.sample.jsonl"
    $UsingIntegratedCorpus = $false
    Write-Warning "Using the bundled Phase 9B smoke packet. It cannot support a research/release claim and will not touch sealed TEST."
}
if (-not (Test-Path $ReviewedReconstructions)) { throw "Missing reviewed Phase 9B JSONL: $ReviewedReconstructions" }

# Materialize canonical/SFT/preference splits for Train/Evaluate/Full.
if (-not ($ReuseExisting -and (Test-Path (Join-Path $DatasetRoot "dataset_manifest.json")))) {
    $args = @(
        ".\scripts\phase_9b_memory_reconstruction\build_reconstruction_dataset.py",
        "--reviewed-input", $ReviewedReconstructions,
        "--output-dir", $DatasetRoot,
        "--seed", "$Seed"
    )
    if ($UsingIntegratedCorpus) {
        $args += @("--split-policy", "intended")
        $args += @("--curated-corpus-root", $MemoryReconstructionCorpusRoot)
    } else {
        $args += @("--split-policy", "auto")
    }
    if ($SmokeTest) { $args += "--smoke-test" } else { $args += "--require-research-ready" }
    Invoke-PythonStep "[9B.2] Building leakage-safe reconstruction curricula..." $args
}

if ($Mode -in @("Train", "Full")) {
    if (-not $UseLora) { throw "Phase 9B Train/Full publishes low-memory PEFT adapters. Add -UseLora." }

    if (-not ($ReuseExisting -and (Test-Path (Join-Path $SftRoot "training_manifest.json")))) {
        $args = @(
            ".\scripts\phase_9_student_distillation\train_student.py",
            "--student", $StudentModel,
            "--objective", "sft",
            "--train-file", $TrainFile,
            "--dev-file", $DevFile,
            "--output-dir", $SftRoot,
            "--max-sequence-length", "768",
            "--max-completion-tokens", "128",
            "--max-steps", "$MaxSteps",
            "--learning-rate", "0.00005",
            "--gradient-accumulation-steps", "32",
            "--eos-loss-weight", "1.5",
            "--lora"
        )
        Invoke-PythonStep "[9B.3] Training reconstruction SFT on canonical approved targets..." $args
    }

    if (-not $SkipKD -and -not ($ReuseExisting -and (Test-Path (Join-Path $KdRoot "training_manifest.json")))) {
        $args = @(
            ".\scripts\phase_9_student_distillation\train_student.py",
            "--student", $StudentModel,
            "--initial-adapter", $SftAdapter,
            "--teacher", $TeacherModel,
            "--objective", "kd",
            "--train-file", $TrainFile,
            "--dev-file", $DevFile,
            "--output-dir", $KdRoot,
            "--max-sequence-length", "768",
            "--max-completion-tokens", "128",
            "--max-steps", "$MaxSteps",
            "--learning-rate", "0.00005",
            "--gradient-accumulation-steps", "32",
            "--kd-hard-weight-start", "0.75",
            "--kd-hard-weight-end", "0.25",
            "--kd-warmup-ratio", "0.15",
            "--teacher-confidence-floor", "0.02",
            "--eos-loss-weight", "1.5"
        )
        if ($Teacher4Bit) { $args += "--teacher-4bit" }
        Invoke-PythonStep "[9B.4] Running optional progressive reconstruction KD..." $args
    }

    $PreDpoAdapter = if (-not $SkipKD -and (Test-Path $KdAdapter)) { $KdAdapter } else { $SftAdapter }
    if ($RunPreferenceAlignment -and -not ($ReuseExisting -and (Test-Path (Join-Path $DpoRoot "training_manifest.json")))) {
        if (-not $UsingIntegratedCorpus) {
            throw "Curated reconstruction preference alignment is enabled only for the integrated dedicated corpus."
        }
        Invoke-PythonStep "[9B.4P] Refining with curated raw-replay/over-reject/wrong-source preferences..." @(
            ".\scripts\phase_3_preference_alignment\train_dpo.py",
            "--model-id", $StudentModel,
            "--sft-adapter", $PreDpoAdapter,
            "--train-file", (Join-Path $DatasetRoot "preferences_train.jsonl"),
            "--dev-file", (Join-Path $DatasetRoot "preferences_dev.jsonl"),
            "--output-dir", $DpoRoot,
            "--max-steps", "$PreferenceMaxSteps",
            "--seed", "$Seed",
            "--lora-rank", "16"
        )
    }

    if ($Mode -eq "Train") {
        Write-Host "Phase 9B training completed. Run -Mode Evaluate next."
        exit 0
    }
}

if ($Mode -in @("Evaluate", "Full")) {
    $CandidateAdapter = if ($RunPreferenceAlignment -and (Test-Path $DpoAdapter)) {
        $DpoAdapter
    } elseif (-not $SkipKD -and (Test-Path $KdAdapter)) {
        $KdAdapter
    } else {
        $SftAdapter
    }
    if (-not (Test-Path $CandidateAdapter)) { throw "Missing trained Phase 9B adapter: $CandidateAdapter" }
    if (-not (Test-Path $DevCanonical)) { throw "Missing Phase 9B DEV file: $DevCanonical" }

    if (-not ($ReuseExisting -and (Test-Path (Join-Path $DevEvaluationRoot "summary.json")))) {
        $args = @(
            ".\scripts\phase_9b_memory_reconstruction\evaluate_memory_reconstruction.py",
            "--baseline-model", $StudentModel,
            "--candidate-model", $StudentModel,
            "--candidate-adapter", $CandidateAdapter,
            "--benchmark", $DevCanonical,
            "--output-dir", $DevEvaluationRoot
        )
        if ($SmokeTest) { $args += "--smoke-test" }
        Invoke-PythonStep "[9B.5 DEV] Comparing raw replay/prompted/trained/direct reconstruction..." $args
    }

    Invoke-PythonStep "[9B.6 DEV] Building proportional DEV selection evidence..." @(
        ".\scripts\phase_9b_memory_reconstruction\check_phase_9b_gate.py",
        "--summary", (Join-Path $DevEvaluationRoot "summary.json"),
        "--config", $Config,
        "--assessment", $DevAssessment,
        "--gate", $DevGate,
        "--candidate", "trained_reconstruction=$StudentModel,$CandidateAdapter",
        "--candidate", "direct_candidate=$StudentModel,$CandidateAdapter"
    )

    if ($SmokeTest) {
        Write-Host "Smoke evaluation completed on DEV only. Sealed TEST was not touched."
        exit 0
    }

    Invoke-PythonStep "[9B.7 DEV] Freezing one strategy before sealed TEST..." @(
        ".\scripts\phase_9b_memory_reconstruction\freeze_phase_9b_dev_selection.py",
        "--assessment", $DevAssessment,
        "--summary", (Join-Path $DevEvaluationRoot "summary.json"),
        "--output", $DevSelection
    )
    $Frozen = Get-Content $DevSelection -Raw | ConvertFrom-Json
    $LockedStrategy = [string]$Frozen.strategy
    if ([string]::IsNullOrWhiteSpace($LockedStrategy)) { throw "Phase 9B DEV selection did not freeze a strategy." }
    if (-not (Test-Path $TestCanonical)) { throw "Missing sealed Phase 9B TEST file: $TestCanonical" }

    $TestStrategies = "memory_off,raw_replay,prompted_reconstruction"
    if ($LockedStrategy -notin @("prompted_reconstruction")) { $TestStrategies += ",$LockedStrategy" }
    if (-not ($ReuseExisting -and (Test-Path (Join-Path $TestEvaluationRoot "summary.json")))) {
        Invoke-PythonStep "[9B.8 SEALED TEST] Evaluating baselines plus the already-frozen strategy..." @(
            ".\scripts\phase_9b_memory_reconstruction\evaluate_memory_reconstruction.py",
            "--baseline-model", $StudentModel,
            "--candidate-model", $StudentModel,
            "--candidate-adapter", $CandidateAdapter,
            "--benchmark", $TestCanonical,
            "--output-dir", $TestEvaluationRoot,
            "--strategies", $TestStrategies
        )
    }

    $FinalGateArgs = @(
        ".\scripts\phase_9b_memory_reconstruction\check_phase_9b_gate.py",
        "--summary", (Join-Path $TestEvaluationRoot "summary.json"),
        "--config", $Config,
        "--locked-strategy", $LockedStrategy,
        "--candidate", "trained_reconstruction=$StudentModel,$CandidateAdapter",
        "--candidate", "direct_candidate=$StudentModel,$CandidateAdapter"
    )
    Invoke-PythonStep "[9B.9] Building sealed-test assessment and pending human gate..." $FinalGateArgs
}

Write-Host ""
Write-Host "Phase 9B v1.7.5 completed."
Write-Host "DEV selection: $DevSelection"
Write-Host "Assessment: reports\PHASE_9B_ASSESSMENT.json"
Write-Host "Human gate: reports\PHASE_9B_STOP_GATE.md"
Write-Host "After human PASS, run finalize_phase_9b_selection.py with the exact locked strategy."

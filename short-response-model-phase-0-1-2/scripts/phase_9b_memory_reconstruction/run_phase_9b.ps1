<#
.SYNOPSIS
Run Phase 9B contextual memory reconstruction.

.DESCRIPTION
Phase 9B consumes the exact human-approved Phase 9 student. It upgrades external
RAG records with source-state metadata, creates a human-reviewed reconstruction
corpus, trains the student to ACCEPT/ADAPT/REJECT/CONFLICT/ABSTAIN, and compares
raw replay with prompted, trained two-pass, and single-pass distilled behavior.

Prepare generates candidates and a review packet. Train requires accepted rows.
Full intentionally stops after Prepare when no reviewed packet is supplied; it
never converts unreviewed teacher output into training labels.
#>
param(
    [ValidateSet("Prepare", "Train", "Evaluate", "Full")][string]$Mode = "Prepare",
    [string]$Phase9ReferenceManifest = ".\artifacts\references\phase_9_student_reference.json",
    [string]$TeacherModel = "",
    [string]$SourcePrompts = ".\artifacts\phase_9_student_distillation\pilot_dataset_v2\train.jsonl",
    [string]$MemoryFile = ".\data\phase_9b_memory_reconstruction\memories.sample.jsonl",
    [string]$ReviewedReconstructions = "",
    [string]$Config = ".\configs\phase_9b_memory_reconstruction.json",
    [string]$OutputRoot = ".\artifacts\phase_9b_memory_reconstruction",
    [int]$MaxSteps = 400,
    [int]$Seed = 42,
    [switch]$UseLora,
    [switch]$Teacher4Bit,
    [switch]$SkipKD,
    [switch]$SkipTeacherGeneration,
    [switch]$SmokeTest,
    [switch]$AllowUnapprovedPhase9,
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
if (-not $AllowUnapprovedPhase9) {
    Invoke-PythonStep "[Gate] Requiring the accepted Phase 9 student..." @(
        ".\scripts\common\check_human_gate.py", "--phase", "9"
    )
}
$StudentModel = Resolve-ManifestModel $Phase9ReferenceManifest
$NeedsTeacher = (($Mode -in @("Prepare", "Full")) -and -not $SkipTeacherGeneration) -or (($Mode -in @("Train", "Full")) -and -not $SkipKD)
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
$TestFile = Join-Path $DatasetRoot "test.jsonl"
$SftRoot = Join-Path $OutputRoot "student_reconstruction_sft"
$SftAdapter = Join-Path $SftRoot "adapter"
$KdRoot = Join-Path $OutputRoot "student_reconstruction_kd"
$KdAdapter = Join-Path $KdRoot "adapter"
$EvaluationRoot = Join-Path $OutputRoot "evaluation"

function Prepare-ReconstructionReview {
    if (-not (Test-Path $SourcePrompts)) {
        if ($SmokeTest) {
            Write-Warning "Phase 9 prompt artifacts are unavailable; use the bundled reviewed sample for a smoke test."
            return
        }
        throw "Missing Phase 9 source prompts: $SourcePrompts"
    }
    if (-not ($ReuseExisting -and (Test-Path $Candidates))) {
        $args = @(
            ".\scripts\phase_9b_memory_reconstruction\build_reconstruction_candidates.py",
            "--source-prompts", $SourcePrompts,
            "--output", $Candidates,
            "--review-packet", $CandidateReview
        )
        if (Test-Path $MemoryFile) { $args += @("--memory-file", $MemoryFile) }
        Invoke-PythonStep "[9B.0] Building source-state memory candidates..." $args
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

if ($Mode -in @("Prepare", "Full")) {
    Prepare-ReconstructionReview
    if ($Mode -eq "Prepare" -or ($Mode -eq "Full" -and [string]::IsNullOrWhiteSpace($ReviewedReconstructions) -and -not $SmokeTest)) {
        Write-Host ""
        Write-Host "Phase 9B preparation completed."
        Write-Host "Review packet: $TeacherReview"
        Write-Host "After review, save accepted JSONL and rerun with -Mode Train -ReviewedReconstructions <path> -UseLora."
        exit 0
    }
}

if ([string]::IsNullOrWhiteSpace($ReviewedReconstructions)) {
    if ($SmokeTest) {
        $ReviewedReconstructions = ".\data\phase_9b_memory_reconstruction\reviewed.sample.jsonl"
        Write-Warning "Using the bundled Phase 9B smoke packet. It cannot support a research or release claim."
    } else {
        throw "Phase 9B training requires -ReviewedReconstructions. Unreviewed teacher output is never used automatically."
    }
}
if (-not (Test-Path $ReviewedReconstructions)) { throw "Missing reviewed Phase 9B JSONL: $ReviewedReconstructions" }

if ($Mode -in @("Train", "Full") -and -not $UseLora) {
    throw "Phase 9B Train/Full currently publishes LoRA adapters for low-memory continuation. Add -UseLora."
}

if ($Mode -in @("Train", "Full")) {
    if (-not ($ReuseExisting -and (Test-Path (Join-Path $DatasetRoot "dataset_manifest.json")))) {
        $args = @(
            ".\scripts\phase_9b_memory_reconstruction\build_reconstruction_dataset.py",
            "--reviewed-input", $ReviewedReconstructions,
            "--output-dir", $DatasetRoot,
            "--seed", "$Seed"
        )
        if ($SmokeTest) { $args += "--smoke-test" } else { $args += "--require-research-ready" }
        Invoke-PythonStep "[9B.2] Building leakage-safe reconstruction curricula..." $args
    }

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
            "--eos-loss-weight", "1.5"
        )
        if ($UseLora) { $args += "--lora" }
        Invoke-PythonStep "[9B.3] Training memory-reconstruction SFT..." $args
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
    if ($Mode -eq "Train") {
        Write-Host "Phase 9B training completed. Run -Mode Evaluate next."
        exit 0
    }
}

if ($Mode -in @("Evaluate", "Full")) {
    $CandidateAdapter = if (-not $SkipKD -and (Test-Path $KdAdapter)) { $KdAdapter } else { $SftAdapter }
    if (-not (Test-Path $CandidateAdapter)) { throw "Missing trained Phase 9B adapter: $CandidateAdapter" }
    if (-not (Test-Path $TestFile)) { throw "Missing Phase 9B test file: $TestFile" }
    if (-not ($ReuseExisting -and (Test-Path (Join-Path $EvaluationRoot "summary.json")))) {
        $args = @(
            ".\scripts\phase_9b_memory_reconstruction\evaluate_memory_reconstruction.py",
            "--baseline-model", $StudentModel,
            "--candidate-model", $StudentModel,
            "--candidate-adapter", $CandidateAdapter,
            "--benchmark", $TestFile,
            "--output-dir", $EvaluationRoot
        )
        if ($SmokeTest) { $args += "--smoke-test" }
        Invoke-PythonStep "[9B.5] Comparing raw replay and reconstruction strategies..." $args
    }
    Invoke-PythonStep "[9B.6] Building the proportional assessment and human gate..." @(
        ".\scripts\phase_9b_memory_reconstruction\check_phase_9b_gate.py",
        "--summary", (Join-Path $EvaluationRoot "summary.json"),
        "--config", $Config,
        "--candidate", "trained_reconstruction=$StudentModel,$CandidateAdapter",
        "--candidate", "direct_candidate=$StudentModel,$CandidateAdapter"
    )
}

Write-Host ""
Write-Host "Phase 9B completed."
Write-Host "Assessment: reports\PHASE_9B_ASSESSMENT.json"
Write-Host "Human gate: reports\PHASE_9B_STOP_GATE.md"
Write-Host "After human PASS, run finalize_phase_9b_selection.py with the exact selected strategy."

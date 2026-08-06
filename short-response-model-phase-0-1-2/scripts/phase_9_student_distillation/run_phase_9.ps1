<#
.SYNOPSIS
Run the refined Phase 9 tiny-student/RAG distillation workflow.

.DESCRIPTION
Bootstrap creates student initializations, the FTS5 knowledge index, and an
explicit human-review packet. Pilot builds leakage-safe splits, refuses to call
a tiny six-row packet research-ready unless -SmokeTest is explicit, creates a
controlled behavior curriculum from human-approved targets, trains curriculum
SFT, then performs confidence-filtered progressive KD from the SFT checkpoint.
Direct KD from initialization is retained only as an optional ablation.

Single is the recommended production-foundation track. Hybrid adds the warm
student and narrow random-initialized research controls; those controls can never
be recommended for production automatically.
#>
param(
    [ValidateSet("Bootstrap", "Pilot", "Evaluate")][string]$Mode = "Bootstrap",
    [ValidateSet("Single", "Hybrid")][string]$StudentSuite = "Single",
    [string]$TeacherModel = "",
    [string]$StudentSource = "",
    [ValidateSet("depth_copy", "custom_qwen_random", "pretrained_clone")][string]$StudentMode = "depth_copy",
    [int]$StudentLayers = 12,
    [int]$HiddenSize = 384,
    [int]$IntermediateSize = 1024,
    [int]$AttentionHeads = 6,
    [int]$KvHeads = 2,
    [int]$HeadDim = 64,
    [string]$Config = ".\configs\phase_9_student_distillation.json",
    [string]$Knowledge = ".\data\phase_9_student_distillation\knowledge.sample.jsonl",
    [string]$Procedures = ".\data\phase_9_student_distillation\procedures.sample.jsonl",
    [string]$Prompts = ".\data\phase_9_student_distillation\prompts.sample.jsonl",
    [string]$FoundationPrompts = ".\data\phase_9_student_distillation\foundation\prompts.foundation.jsonl",
    [string]$ReviewedTeacherData = "",
    [string]$OutputRoot = ".\artifacts\phase_9_student_distillation",
    [int]$MaxSteps = 600,
    [int]$Seed = 42,
    [switch]$UseLora,
    [switch]$Teacher4Bit,
    [switch]$SkipKD,
    [switch]$RunDirectKD,
    [switch]$SkipCurriculum,
    [switch]$SkipFoundation,
    [int]$PrimaryCurriculumRepeat = 3,
    [int]$FoundationRepeat = 1,
    [switch]$SmokeTest,
    [switch]$RunOnPolicy,
    [double]$OnPolicyStudentFraction = 0.20,
    [switch]$AllowUnreleasedTeacher,
    [switch]$AcknowledgeRandomControl,
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

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    return Get-Content -Raw -Path $Path | ConvertFrom-Json
}

function Get-FileSha256 {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "" }
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

function Get-TreeSha256 {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "" }
    $value = & python -c "from pathlib import Path; import sys; from scripts.common.experiment_utils import sha256_tree; print(sha256_tree(Path(sys.argv[1])))" $Path
    if ($LASTEXITCODE -ne 0) { return "" }
    return ([string]($value | Select-Object -Last 1)).Trim()
}

function Test-CurrentDerivedArtifact {
    param([string]$Artifact, [string]$ManifestPath, [string]$Source)
    if (-not $ReuseExisting -or -not (Test-Path $Artifact) -or -not (Test-Path $ManifestPath)) { return $false }
    $manifest = Read-JsonFile $ManifestPath
    if ($null -eq $manifest -or [string]$manifest.pipeline_revision -ne "phase9_behavior_curriculum_v2") { return $false }
    return [string]$manifest.source_sha256 -eq (Get-FileSha256 $Source)
}

function Test-CurrentMixedCurriculum {
    param([string]$Artifact, [string]$ManifestPath, [string]$Primary, [string]$Foundation)
    if (-not $ReuseExisting -or -not (Test-Path $Artifact) -or -not (Test-Path $ManifestPath)) { return $false }
    $manifest = Read-JsonFile $ManifestPath
    if ($null -eq $manifest -or [string]$manifest.pipeline_revision -ne "phase9_behavior_curriculum_v2") { return $false }
    return (
        [string]$manifest.primary_sha256 -eq (Get-FileSha256 $Primary) -and
        [string]$manifest.foundation_sha256 -eq (Get-FileSha256 $Foundation) -and
        [int]$manifest.primary_repeat -eq $PrimaryCurriculumRepeat -and
        [int]$manifest.foundation_repeat -eq $FoundationRepeat -and
        [int]$manifest.seed -eq $Seed
    )
}

function Test-CurrentKnowledgeIndex {
    if (-not $ReuseExisting -or -not (Test-Path $Database)) { return $false }
    $manifest = Read-JsonFile "$Database.manifest.json"
    if ($null -eq $manifest) { return $false }
    $knowledgeEntry = @($manifest.knowledge)[0]
    $procedureEntry = @($manifest.procedures)[0]
    if ($null -eq $knowledgeEntry -or $null -eq $procedureEntry) { return $false }
    return (
        [string]$knowledgeEntry.sha256 -eq (Get-FileSha256 $Knowledge) -and
        [string]$procedureEntry.sha256 -eq (Get-FileSha256 $Procedures)
    )
}

function Test-CurrentTrainingArtifact {
    param(
        [string]$Directory,
        [string]$TrainingSource,
        [string]$Objective,
        [string]$ExpectedStudent,
        [string]$ExpectedTeacher = "",
        [string]$ExpectedInitialAdapter = ""
    )
    if (-not $ReuseExisting) { return $false }
    $manifest = Read-JsonFile (Join-Path $Directory "training_manifest.json")
    if ($null -eq $manifest -or [string]$manifest.pipeline_revision -ne "phase9_behavior_curriculum_v2") { return $false }
    if ([string]$manifest.objective -ne $Objective) { return $false }
    if ([string]$manifest.train_sha256 -ne (Get-FileSha256 $TrainingSource)) { return $false }
    if ($null -eq $manifest.requested_max_steps -or [int]$manifest.requested_max_steps -ne $MaxSteps) { return $false }
    if ([string]$manifest.student_initialization_tree_sha256 -ne (Get-TreeSha256 $ExpectedStudent)) { return $false }
    if ([string]::IsNullOrWhiteSpace($ExpectedInitialAdapter)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$manifest.initial_adapter)) { return $false }
    } elseif ([string]$manifest.initial_adapter_tree_sha256 -ne (Get-TreeSha256 $ExpectedInitialAdapter)) {
        return $false
    }
    if ($Objective -eq "kd") {
        if ([string]$manifest.teacher_tree_sha256 -ne (Get-TreeSha256 $ExpectedTeacher)) { return $false }
    }
    return (Test-Path ([string]$manifest.artifact_path))
}

function Test-CurrentEvaluation {
    param([string]$SummaryPath, [string]$Model, [string]$Adapter)
    if (-not $ReuseExisting -or -not (Test-Path $SummaryPath)) { return $false }
    $summary = Read-JsonFile $SummaryPath
    if ($null -eq $summary -or [string]$summary.schema_version -ne "1.1") { return $false }
    if ([string]$summary.benchmark_sha256 -ne (Get-FileSha256 $TestFile)) { return $false }
    if ([string]$summary.model_tree_sha256 -ne (Get-TreeSha256 $Model)) { return $false }
    if ([string]::IsNullOrWhiteSpace($Adapter)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$summary.adapter)) { return $false }
    } elseif ([string]$summary.adapter_tree_sha256 -ne (Get-TreeSha256 $Adapter)) {
        return $false
    }
    $conditionNames = @($summary.conditions.PSObject.Properties.Name | Sort-Object)
    $expectedConditions = @("adversarial_passthrough_rag", "adversarial_rag", "closed_book", "oracle_rag", "real_rag")
    if (($conditionNames -join ",") -ne ($expectedConditions -join ",")) { return $false }
    if ([bool]$summary.smoke_test -ne [bool]$SmokeTest) { return $false }
    return $true
}

function Get-HybridProfiles {
    param([string]$ConfigPath)
    if (-not (Test-Path $ConfigPath)) { throw "Missing Phase 9 config: $ConfigPath" }
    $packet = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
    $profiles = $packet.student_tracks.hybrid_compare.profiles
    if ($null -eq $profiles) { throw "Config is missing student_tracks.hybrid_compare.profiles" }
    $ordered = @("warm_344m", "narrow_100m", "narrow_130m", "narrow_170m")
    $result = @()
    foreach ($name in $ordered) {
        $profile = $profiles.$name
        if ($null -eq $profile) { throw "Hybrid suite is missing required profile '$name'" }
        $result += [pscustomobject]@{
            Name = $name
            Mode = [string]$profile.mode
            NumHiddenLayers = [int]$profile.num_hidden_layers
            HiddenSize = if ($null -ne $profile.hidden_size) { [int]$profile.hidden_size } else { 0 }
            IntermediateSize = if ($null -ne $profile.intermediate_size) { [int]$profile.intermediate_size } else { 0 }
            AttentionHeads = if ($null -ne $profile.num_attention_heads) { [int]$profile.num_attention_heads } else { 0 }
            KvHeads = if ($null -ne $profile.num_key_value_heads) { [int]$profile.num_key_value_heads } else { 0 }
            HeadDim = if ($null -ne $profile.head_dim) { [int]$profile.head_dim } else { 64 }
            MaxPositionEmbeddings = if ($null -ne $profile.max_position_embeddings) { [int]$profile.max_position_embeddings } else { 1024 }
            TargetParams = if ($null -ne $profile.target_params) { [long]$profile.target_params } else { 0 }
            TargetParamsTolerance = if ($null -ne $profile.target_params_tolerance) { [double]$profile.target_params_tolerance } else { 0.12 }
            ResearchControl = [bool]$profile.research_control
        }
    }
    return $result
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
if (-not (Test-Path $TeacherModel)) { throw "Phase 9 requires a local teacher checkpoint: $TeacherModel" }
if ([string]::IsNullOrWhiteSpace($StudentSource)) { $StudentSource = $TeacherModel }

$HybridProfiles = @()
if ($StudentSuite -eq "Hybrid") {
    $HybridProfiles = @(Get-HybridProfiles -ConfigPath $Config)
    if (-not $AcknowledgeRandomControl) {
        throw "Hybrid includes random narrow controls. Pass -AcknowledgeRandomControl explicitly."
    }
    if ($RunOnPolicy) { throw "-RunOnPolicy is supported only for -StudentSuite Single." }
} elseif ($StudentMode -eq "custom_qwen_random" -and -not $AcknowledgeRandomControl) {
    throw "custom_qwen_random is a research control. Pass -AcknowledgeRandomControl explicitly."
}
if ($StudentMode -eq "pretrained_clone" -and -not $SkipKD) {
    Write-Warning "Direct vocabulary KD requires an identical tokenizer. Use -SkipKD for a cross-tokenizer pretrained challenger."
}

$Database = Join-Path $OutputRoot "knowledge.sqlite3"
$TeacherCandidates = Join-Path $OutputRoot "teacher_candidates.jsonl"
$ReviewPacket = Join-Path $OutputRoot "teacher_candidates.review.md"
$DatasetRoot = Join-Path $OutputRoot "pilot_dataset_v2"
$TrainFile = Join-Path $DatasetRoot "train.jsonl"
$DevFile = Join-Path $DatasetRoot "dev.jsonl"
$TestFile = Join-Path $DatasetRoot "test.jsonl"
$CurriculumTrain = Join-Path $DatasetRoot "train_curriculum.jsonl"
$FoundationTrain = Join-Path $DatasetRoot "foundation_train.jsonl"
$FoundationCurriculum = Join-Path $DatasetRoot "foundation_curriculum.jsonl"
$MixedCurriculum = Join-Path $DatasetRoot "train_mixed_curriculum.jsonl"
$Calibration = Join-Path $OutputRoot "evidence_gate_calibration_v2.json"
$EvalTeacher = Join-Path $OutputRoot "evaluation_teacher_v2"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Ensure-KnowledgeIndex {
    if (Test-CurrentKnowledgeIndex) { return }
    if (-not (Test-Path $Knowledge)) { throw "Missing knowledge JSONL: $Knowledge" }
    if (-not (Test-Path $Procedures)) { throw "Missing procedure JSONL: $Procedures" }
    Invoke-PythonStep "[Index] Building SQLite FTS5 knowledge/procedure index..." @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py",
        "--knowledge", $Knowledge,
        "--procedures", $Procedures,
        "--output", $Database
    )
}

function Ensure-SingleInitialization {
    param([string]$Destination)
    if (Test-Path (Join-Path $Destination "student_creation_manifest.json")) { return }
    $args = @(
        ".\scripts\phase_9_student_distillation\create_student.py",
        "--source", $StudentSource,
        "--output", $Destination,
        "--mode", $StudentMode,
        "--num-hidden-layers", "$StudentLayers",
        "--hidden-size", "$HiddenSize",
        "--intermediate-size", "$IntermediateSize",
        "--num-attention-heads", "$AttentionHeads",
        "--num-key-value-heads", "$KvHeads",
        "--head-dim", "$HeadDim"
    )
    if ($AcknowledgeRandomControl) { $args += "--acknowledge-random-control" }
    if (Test-Path $Destination) { $args += "--overwrite" }
    Invoke-PythonStep "[Student] Creating the explicit student initialization..." $args
}

function Ensure-ProfileInitialization {
    param([pscustomobject]$Profile, [string]$Destination, [string]$Label)
    if (Test-Path (Join-Path $Destination "student_creation_manifest.json")) { return }
    $args = @(
        ".\scripts\phase_9_student_distillation\create_student.py",
        "--source", $StudentSource,
        "--output", $Destination,
        "--mode", $Profile.Mode,
        "--num-hidden-layers", "$($Profile.NumHiddenLayers)",
        "--profile-name", $Profile.Name
    )
    if ($Profile.Mode -eq "custom_qwen_random") {
        $args += @(
            "--hidden-size", "$($Profile.HiddenSize)",
            "--intermediate-size", "$($Profile.IntermediateSize)",
            "--num-attention-heads", "$($Profile.AttentionHeads)",
            "--num-key-value-heads", "$($Profile.KvHeads)",
            "--head-dim", "$($Profile.HeadDim)",
            "--max-position-embeddings", "$($Profile.MaxPositionEmbeddings)",
            "--acknowledge-random-control"
        )
    }
    if ($Profile.TargetParams -gt 0) {
        $args += @(
            "--target-params", "$($Profile.TargetParams)",
            "--target-params-tolerance", "$($Profile.TargetParamsTolerance)"
        )
    }
    if (Test-Path $Destination) { $args += "--overwrite" }
    Invoke-PythonStep $Label $args
}

function Invoke-RagEvaluation {
    param([string]$Name, [string]$Model, [string]$Adapter, [string]$Destination)
    $summaryPath = Join-Path $Destination "summary.json"
    if (Test-CurrentEvaluation -SummaryPath $summaryPath -Model $Model -Adapter $Adapter) { return }
    if ($ReuseExisting -and (Test-Path $Destination)) {
        Write-Warning "Deleting stale or incompatible Phase 9 evaluation: $Destination"
        Remove-Item -Recurse -Force $Destination
    }
    $args = @(
        ".\scripts\phase_9_student_distillation\evaluate_rag.py",
        "--model", $Model,
        "--benchmark", $TestFile,
        "--output-dir", $Destination,
        "--knowledge-db", $Database,
        "--gate-calibration", $Calibration,
        "--conditions", "closed_book,oracle_rag,real_rag,adversarial_rag,adversarial_passthrough_rag",
        "--max-new-tokens", "32",
        "--minimum-cases-per-condition", "1",
        "--minimum-supported-cases", "1",
        "--minimum-unsupported-cases", "1"
    )
    if ($SmokeTest) { $args += "--smoke-test" } else { $args += "--require-research-ready" }
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $args += @("--adapter", $Adapter) }
    Invoke-PythonStep $Name $args
}

function Train-Profile {
    param([pscustomobject]$Profile, [string]$ProfileRoot, [string]$TrainingFile, [string]$Label)
    $studentRoot = Join-Path $ProfileRoot "initialization"
    $sftRoot = Join-Path $ProfileRoot "sft_v2"
    $progressiveRoot = Join-Path $ProfileRoot "progressive_kd_v2"
    $directRoot = Join-Path $ProfileRoot "direct_kd_v2"
    Ensure-ProfileInitialization -Profile $Profile -Destination $studentRoot -Label "$Label Creating $($Profile.Name)..."

    if ($Mode -eq "Pilot" -and -not (Test-CurrentTrainingArtifact -Directory $sftRoot -TrainingSource $TrainingFile -Objective "sft" -ExpectedStudent $studentRoot)) {
        $args = @(
            ".\scripts\phase_9_student_distillation\train_student.py",
            "--student", $studentRoot,
            "--objective", "sft",
            "--train-file", $TrainingFile,
            "--dev-file", $DevFile,
            "--output-dir", $sftRoot,
            "--max-steps", "$MaxSteps",
            "--seed", "$Seed",
            "--eos-loss-weight", "1.5"
        )
        if ($UseLora) { $args += "--lora" }
        if ($Profile.ResearchControl) { $args += "--allow-random-control" }
        Invoke-PythonStep "$Label Curriculum SFT for $($Profile.Name)..." $args
    }

    $sftModel = if ($UseLora) { $studentRoot } else { Join-Path $sftRoot "model" }
    $sftAdapter = if ($UseLora) { Join-Path $sftRoot "adapter" } else { "" }
    if ($Mode -eq "Pilot" -and -not $SkipKD -and -not (Test-CurrentTrainingArtifact -Directory $progressiveRoot -TrainingSource $TrainingFile -Objective "kd" -ExpectedStudent $sftModel -ExpectedTeacher $TeacherModel -ExpectedInitialAdapter $sftAdapter)) {
        $args = @(
            ".\scripts\phase_9_student_distillation\train_student.py",
            "--student", $sftModel,
            "--teacher", $TeacherModel,
            "--objective", "kd",
            "--train-file", $TrainingFile,
            "--dev-file", $DevFile,
            "--output-dir", $progressiveRoot,
            "--max-steps", "$MaxSteps",
            "--seed", "$Seed",
            "--kd-hard-weight-start", "0.75",
            "--kd-hard-weight-end", "0.25",
            "--kd-warmup-ratio", "0.15",
            "--teacher-confidence-floor", "0.02",
            "--eos-loss-weight", "1.5"
        )
        if (-not [string]::IsNullOrWhiteSpace($sftAdapter)) { $args += @("--initial-adapter", $sftAdapter) }
        if ($Teacher4Bit) { $args += "--teacher-4bit" }
        if ($Profile.ResearchControl) { $args += "--allow-random-control" }
        Invoke-PythonStep "$Label Progressive SFT-to-KD for $($Profile.Name)..." $args
    }

    if ($Mode -eq "Pilot" -and $RunDirectKD -and -not $SkipKD -and -not (Test-CurrentTrainingArtifact -Directory $directRoot -TrainingSource $TrainingFile -Objective "kd" -ExpectedStudent $studentRoot -ExpectedTeacher $TeacherModel)) {
        $args = @(
            ".\scripts\phase_9_student_distillation\train_student.py",
            "--student", $studentRoot,
            "--teacher", $TeacherModel,
            "--objective", "kd",
            "--train-file", $TrainingFile,
            "--dev-file", $DevFile,
            "--output-dir", $directRoot,
            "--max-steps", "$MaxSteps",
            "--seed", "$Seed",
            "--kd-hard-weight-start", "0.85",
            "--kd-hard-weight-end", "0.35",
            "--kd-warmup-ratio", "0.25",
            "--teacher-confidence-floor", "0.02",
            "--eos-loss-weight", "1.5"
        )
        if ($UseLora) { $args += "--lora" }
        if ($Teacher4Bit) { $args += "--teacher-4bit" }
        if ($Profile.ResearchControl) { $args += "--allow-random-control" }
        Invoke-PythonStep "$Label Direct-KD ablation for $($Profile.Name)..." $args
    }

    $evalSft = Join-Path $ProfileRoot "evaluation_sft_v2"
    Invoke-RagEvaluation "$Label Evaluating SFT for $($Profile.Name)..." $sftModel $sftAdapter $evalSft
    $result = [ordered]@{
        Name = $Profile.Name
        ResearchControl = [bool]$Profile.ResearchControl
        SftSummary = Join-Path $evalSft "summary.json"
    }
    if (-not $SkipKD) {
        $progressiveModel = if ($UseLora) { $studentRoot } else { Join-Path $progressiveRoot "model" }
        $progressiveAdapter = if ($UseLora) { Join-Path $progressiveRoot "adapter" } else { "" }
        $evalProgressive = Join-Path $ProfileRoot "evaluation_progressive_kd_v2"
        Invoke-RagEvaluation "$Label Evaluating progressive KD for $($Profile.Name)..." $progressiveModel $progressiveAdapter $evalProgressive
        $result.ProgressiveKdSummary = Join-Path $evalProgressive "summary.json"
    }
    if ($RunDirectKD -and -not $SkipKD) {
        $directModel = if ($UseLora) { $studentRoot } else { Join-Path $directRoot "model" }
        $directAdapter = if ($UseLora) { Join-Path $directRoot "adapter" } else { "" }
        $evalDirect = Join-Path $ProfileRoot "evaluation_direct_kd_v2"
        Invoke-RagEvaluation "$Label Evaluating direct KD for $($Profile.Name)..." $directModel $directAdapter $evalDirect
        $result.DirectKdSummary = Join-Path $evalDirect "summary.json"
    }
    return [pscustomobject]$result
}

if ($Mode -eq "Bootstrap") {
    if ($StudentSuite -eq "Hybrid") {
        $index = 1
        foreach ($profile in $HybridProfiles) {
            $destination = Join-Path $OutputRoot (Join-Path "students" (Join-Path $profile.Name "initialization"))
            Ensure-ProfileInitialization -Profile $profile -Destination $destination -Label "[Hybrid $index/$($HybridProfiles.Count)] Creating $($profile.Name)..."
            $index += 1
        }
    } else {
        Ensure-SingleInitialization -Destination (Join-Path $OutputRoot "student_initialization")
    }
    Ensure-KnowledgeIndex
    if (-not (Test-Path $Prompts)) { throw "Missing prompt JSONL: $Prompts" }
    $args = @(
        ".\scripts\phase_9_student_distillation\generate_teacher_data.py",
        "--teacher", $TeacherModel,
        "--prompts", $Prompts,
        "--knowledge-db", $Database,
        "--output", $TeacherCandidates,
        "--review-packet", $ReviewPacket,
        "--retrieval-mode", "oracle",
        "--samples-per-question", "2",
        "--max-new-tokens", "32",
        "--resume"
    )
    Invoke-PythonStep "[Teacher] Generating the candidate/review packet..." $args
    Write-Host ""
    Write-Host "Bootstrap completed. Human-review $TeacherCandidates before Pilot mode."
    Write-Host "Review packet: $ReviewPacket"
    exit 0
}

Ensure-KnowledgeIndex
if ([string]::IsNullOrWhiteSpace($ReviewedTeacherData)) {
    throw "Pilot/Evaluate requires -ReviewedTeacherData containing review_status=accepted rows."
}
if (-not (Test-Path $ReviewedTeacherData)) { throw "Reviewed teacher data does not exist: $ReviewedTeacherData" }

if ($Mode -eq "Pilot") {
    if (-not (Test-CurrentDerivedArtifact -Artifact $TrainFile -ManifestPath (Join-Path $DatasetRoot "dataset_manifest.json") -Source $ReviewedTeacherData)) {
        $args = @(
            ".\scripts\phase_9_student_distillation\build_pilot_dataset.py",
            "--reviewed-input", $ReviewedTeacherData,
            "--output-dir", $DatasetRoot,
            "--seed", "$Seed"
        )
        if ($SmokeTest) { $args += "--smoke-test" } else { $args += "--require-research-ready" }
        Invoke-PythonStep "[Data] Building document/topic-disjoint, class-aware splits..." $args
    }
    $TrainingFile = $TrainFile
    if (-not $SkipCurriculum) {
        $curriculumManifest = [System.IO.Path]::ChangeExtension($CurriculumTrain, ".manifest.json")
        if (-not (Test-CurrentDerivedArtifact -Artifact $CurriculumTrain -ManifestPath $curriculumManifest -Source $TrainFile)) {
            Invoke-PythonStep "[Data] Building the human-approved behavior curriculum..." @(
                ".\scripts\phase_9_student_distillation\build_curriculum_dataset.py",
                "--input", $TrainFile,
                "--output", $CurriculumTrain,
                "--seed", "$Seed",
                "--canonical-repeats", "2",
                "--prompt-variants", "2",
                "--distractor-variants", "2",
                "--missing-evidence-variants", "2",
                "--injection-variants", "1",
                "--social-repeats", "2"
            )
        }
        $TrainingFile = $CurriculumTrain
    }
    if (-not $SkipFoundation) {
        if (-not (Test-Path $FoundationPrompts)) { throw "Missing synthetic foundation prompts: $FoundationPrompts" }
        $foundationManifest = [System.IO.Path]::ChangeExtension($FoundationTrain, ".manifest.json")
        if (-not (Test-CurrentDerivedArtifact -Artifact $FoundationTrain -ManifestPath $foundationManifest -Source $FoundationPrompts)) {
            Invoke-PythonStep "[Data] Materializing the versioned synthetic behavior foundation..." @(
                ".\scripts\phase_9_student_distillation\build_foundation_dataset.py",
                "--prompts", $FoundationPrompts,
                "--output", $FoundationTrain
            )
        }
        $foundationCurriculumManifest = [System.IO.Path]::ChangeExtension($FoundationCurriculum, ".manifest.json")
        if (-not (Test-CurrentDerivedArtifact -Artifact $FoundationCurriculum -ManifestPath $foundationCurriculumManifest -Source $FoundationTrain)) {
            Invoke-PythonStep "[Data] Expanding the synthetic behavior foundation..." @(
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
        $mixedManifest = [System.IO.Path]::ChangeExtension($MixedCurriculum, ".manifest.json")
        if (-not (Test-CurrentMixedCurriculum -Artifact $MixedCurriculum -ManifestPath $mixedManifest -Primary $TrainingFile -Foundation $FoundationCurriculum)) {
            Invoke-PythonStep "[Data] Mixing human-reviewed and synthetic curricula..." @(
                ".\scripts\phase_9_student_distillation\merge_training_data.py",
                "--primary", $TrainingFile,
                "--foundation", $FoundationCurriculum,
                "--output", $MixedCurriculum,
                "--primary-repeat", "$PrimaryCurriculumRepeat",
                "--foundation-repeat", "$FoundationRepeat",
                "--seed", "$Seed"
            )
        }
        $TrainingFile = $MixedCurriculum
    }
    $calibrationArgs = @(
        ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py",
        "--benchmark", $DevFile,
        "--knowledge-db", $Database,
        "--output", $Calibration
    )
    if ($SmokeTest) { $calibrationArgs += "--smoke-test" } else { $calibrationArgs += "--require-research-ready" }
    Invoke-PythonStep "[Data] Calibrating the evidence pre-gate on dev only..." $calibrationArgs
} else {
    if (-not $SkipFoundation -and (Test-Path $MixedCurriculum)) {
        $TrainingFile = $MixedCurriculum
    } elseif (-not $SkipCurriculum -and (Test-Path $CurriculumTrain)) {
        $TrainingFile = $CurriculumTrain
    } else {
        $TrainingFile = $TrainFile
    }
    foreach ($required in @($TrainFile, $DevFile, $TestFile, $Calibration)) {
        if (-not (Test-Path $required)) { throw "Evaluate mode is missing required artifact: $required" }
    }
}

Invoke-RagEvaluation "[Teacher] Evaluating the teacher under identical conditions..." $TeacherModel "" $EvalTeacher
$GateArgs = @(
    ".\scripts\phase_9_student_distillation\check_phase_9_gate.py",
    "--teacher-summary", (Join-Path $EvalTeacher "summary.json")
)

if ($StudentSuite -eq "Hybrid") {
    $index = 1
    foreach ($profile in $HybridProfiles) {
        $profileRoot = Join-Path $OutputRoot (Join-Path "students" $profile.Name)
        $result = Train-Profile -Profile $profile -ProfileRoot $profileRoot -TrainingFile $TrainingFile -Label "[Hybrid $index/$($HybridProfiles.Count)]"
        $GateArgs += @("--candidate", "sft_$($result.Name)=$($result.SftSummary)")
        if (-not $SkipKD) { $GateArgs += @("--candidate", "progressive_kd_$($result.Name)=$($result.ProgressiveKdSummary)") }
        if ($RunDirectKD -and -not $SkipKD) { $GateArgs += @("--candidate", "direct_kd_$($result.Name)=$($result.DirectKdSummary)") }
        if ($result.ResearchControl) {
            $GateArgs += @("--research-control", "sft_$($result.Name)")
            if (-not $SkipKD) { $GateArgs += @("--research-control", "progressive_kd_$($result.Name)") }
            if ($RunDirectKD -and -not $SkipKD) { $GateArgs += @("--research-control", "direct_kd_$($result.Name)") }
        }
        $index += 1
    }
} else {
    $studentRoot = Join-Path $OutputRoot "student_initialization"
    Ensure-SingleInitialization -Destination $studentRoot
    $profile = [pscustomobject]@{
        Name = "single"
        ResearchControl = ($StudentMode -eq "custom_qwen_random")
    }
    # Reuse the profile trainer by pointing it at the already-created initialization.
    $profileRoot = Join-Path $OutputRoot "single_v2"
    $profileInit = Join-Path $profileRoot "initialization"
    if (-not (Test-Path $profileInit)) {
        New-Item -ItemType Directory -Force -Path $profileRoot | Out-Null
        try {
            New-Item -ItemType Junction -Path $profileInit -Target (Resolve-Path $studentRoot).Path | Out-Null
        } catch {
            Copy-Item -Recurse -Force $studentRoot $profileInit
        }
    }
    # A small inline equivalent avoids recreating the single initialization.
    $singleProfile = [pscustomobject]@{
        Name = "single"
        Mode = $StudentMode
        NumHiddenLayers = $StudentLayers
        HiddenSize = $HiddenSize
        IntermediateSize = $IntermediateSize
        AttentionHeads = $AttentionHeads
        KvHeads = $KvHeads
        HeadDim = $HeadDim
        MaxPositionEmbeddings = 1024
        TargetParams = 0
        TargetParamsTolerance = 0.12
        ResearchControl = ($StudentMode -eq "custom_qwen_random")
    }
    $result = Train-Profile -Profile $singleProfile -ProfileRoot $profileRoot -TrainingFile $TrainingFile -Label "[Single]"
    $GateArgs += @("--candidate", "sft=$($result.SftSummary)")
    if (-not $SkipKD) { $GateArgs += @("--candidate", "progressive_kd=$($result.ProgressiveKdSummary)") }
    if ($RunDirectKD -and -not $SkipKD) { $GateArgs += @("--candidate", "direct_kd=$($result.DirectKdSummary)") }
    if ($singleProfile.ResearchControl) {
        $GateArgs += @("--research-control", "sft")
        if (-not $SkipKD) { $GateArgs += @("--research-control", "progressive_kd") }
        if ($RunDirectKD -and -not $SkipKD) { $GateArgs += @("--research-control", "direct_kd") }
    }

    if ($RunOnPolicy) {
        if ($SkipKD) { throw "-RunOnPolicy requires progressive KD." }
        $progressiveRoot = Join-Path $profileRoot "progressive_kd_v2"
        $progressiveModel = if ($UseLora) { $profileInit } else { Join-Path $progressiveRoot "model" }
        $progressiveAdapter = if ($UseLora) { Join-Path $progressiveRoot "adapter" } else { "" }
        $onPolicyData = Join-Path $OutputRoot "on_policy_train_v2.jsonl"
        $onPolicyRoot = Join-Path $OutputRoot "student_on_policy_v2"
        $inputModel = if ($UseLora) { $profileInit } else { $progressiveModel }
        if ($Mode -eq "Pilot" -and -not ($ReuseExisting -and (Test-Path $onPolicyData))) {
            $args = @(
                ".\scripts\phase_9_student_distillation\generate_on_policy_data.py",
                "--student", $progressiveModel,
                "--input", $TrainingFile,
                "--output", $onPolicyData,
                "--student-fraction", "$OnPolicyStudentFraction",
                "--seed", "$Seed"
            )
            if (-not [string]::IsNullOrWhiteSpace($progressiveAdapter)) { $args += @("--adapter", $progressiveAdapter) }
            Invoke-PythonStep "[On-policy] Generating mixed trajectories..." $args
        }
        if ($Mode -eq "Pilot" -and -not (Test-CurrentTrainingArtifact -Directory $onPolicyRoot -TrainingSource $onPolicyData -Objective "kd" -ExpectedStudent $inputModel -ExpectedTeacher $TeacherModel -ExpectedInitialAdapter $progressiveAdapter)) {
            $args = @(
                ".\scripts\phase_9_student_distillation\train_student.py",
                "--student", $inputModel,
                "--teacher", $TeacherModel,
                "--objective", "kd",
                "--train-file", $onPolicyData,
                "--dev-file", $DevFile,
                "--output-dir", $onPolicyRoot,
                "--max-steps", "$MaxSteps",
                "--seed", "$Seed",
                "--kd-hard-weight-start", "0.50",
                "--kd-hard-weight-end", "0.10",
                "--kd-warmup-ratio", "0.05",
                "--teacher-confidence-floor", "0.02",
                "--eos-loss-weight", "1.5"
            )
            if (-not [string]::IsNullOrWhiteSpace($progressiveAdapter)) { $args += @("--initial-adapter", $progressiveAdapter) }
            if ($Teacher4Bit) { $args += "--teacher-4bit" }
            if ($singleProfile.ResearchControl) { $args += "--allow-random-control" }
            Invoke-PythonStep "[On-policy] Training recovery candidate..." $args
        }
        $onPolicyModel = if ($UseLora) { $profileInit } else { Join-Path $onPolicyRoot "model" }
        $onPolicyAdapter = if ($UseLora) { Join-Path $onPolicyRoot "adapter" } else { "" }
        $evalOnPolicy = Join-Path $OutputRoot "evaluation_on_policy_v2"
        Invoke-RagEvaluation "[On-policy] Evaluating recovery candidate..." $onPolicyModel $onPolicyAdapter $evalOnPolicy
        $GateArgs += @("--candidate", "on_policy=$(Join-Path $evalOnPolicy 'summary.json')")
    }
}

Invoke-PythonStep "[Final] Building the proportional Phase 9 assessment and human gate..." $GateArgs
Write-Host ""
Write-Host "Phase 9 $Mode ($StudentSuite) completed."
Write-Host "Assessment: reports\PHASE_9_ASSESSMENT.json"
Write-Host "Human gate: reports\PHASE_9_STOP_GATE.md"
Write-Host "Smoke packets prove execution only; they cannot establish student readiness."

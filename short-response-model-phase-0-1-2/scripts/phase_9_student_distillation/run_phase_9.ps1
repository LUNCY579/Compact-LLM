<#
.SYNOPSIS
Run the appended Phase 9 tiny-student/RAG distillation workflow.

.DESCRIPTION
Bootstrap creates student initialization(s), a low-memory SQLite FTS5 index, and a
teacher-candidate human review packet. Pilot consumes only reviewed teacher rows,
builds document/topic-disjoint splits, calibrates the evidence pre-gate, trains
matched SFT and SFT+KD candidates, evaluates teacher and students under
closed/oracle/real/adversarial RAG, and writes a pending human stop gate.

-StudentSuite Single keeps the original one-student layout.
-StudentSuite Hybrid trains the warm depth-copy intermediate plus three narrow
same-tokenizer research controls and compares them all in one gate.
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
    [string]$ReviewedTeacherData = "",
    [string]$OutputRoot = ".\artifacts\phase_9_student_distillation",
    [int]$MaxSteps = 200,
    [int]$Seed = 42,
    [switch]$UseLora,
    [switch]$Teacher4Bit,
    [switch]$SkipKD,
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

function Get-HybridProfiles {
    param([string]$ConfigPath)
    if (-not (Test-Path $ConfigPath)) {
        throw "Missing Phase 9 config: $ConfigPath"
    }
    $packet = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
    $profiles = $packet.student_tracks.hybrid_compare.profiles
    if ($null -eq $profiles) {
        throw "Config is missing student_tracks.hybrid_compare.profiles: $ConfigPath"
    }
    $ordered = @("warm_344m", "narrow_100m", "narrow_130m", "narrow_170m")
    $result = @()
    foreach ($name in $ordered) {
        $profile = $profiles.$name
        if ($null -eq $profile) {
            throw "Hybrid suite is missing required profile '$name' in $ConfigPath"
        }
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
if (-not (Test-Path $TeacherModel)) {
    throw "Phase 9 currently requires a local teacher checkpoint: $TeacherModel"
}
if ([string]::IsNullOrWhiteSpace($StudentSource)) { $StudentSource = $TeacherModel }

$HybridProfiles = @()
if ($StudentSuite -eq "Hybrid") {
    $HybridProfiles = @(Get-HybridProfiles -ConfigPath $Config)
    if (-not $AcknowledgeRandomControl) {
        throw "Hybrid suite includes random narrow controls. Pass -AcknowledgeRandomControl explicitly."
    }
    if ($RunOnPolicy) {
        throw "-RunOnPolicy is supported only for -StudentSuite Single in this revision."
    }
} else {
    if ($StudentMode -eq "custom_qwen_random" -and -not $AcknowledgeRandomControl) {
        throw "custom_qwen_random is a research control. Pass -AcknowledgeRandomControl explicitly."
    }
    if ($StudentMode -eq "pretrained_clone" -and -not $SkipKD) {
        Write-Warning "A pretrained challenger may use a different tokenizer. Direct KD will reject tokenizer mismatch. Use -SkipKD for sequence-level SFT, or supply an exactly tokenizer-compatible source."
    }
}

$Database = Join-Path $OutputRoot "knowledge.sqlite3"
$TeacherCandidates = Join-Path $OutputRoot "teacher_candidates.jsonl"
$ReviewPacket = Join-Path $OutputRoot "teacher_candidates.review.md"
$DatasetRoot = Join-Path $OutputRoot "pilot_dataset"
$Calibration = Join-Path $OutputRoot "evidence_gate_calibration.json"
$EvalTeacher = Join-Path $OutputRoot "evaluation_teacher"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Ensure-StudentInitializationFromProfile {
    param(
        [pscustomobject]$Profile,
        [string]$StudentRoot,
        [string]$StepLabel
    )
    if ($ReuseExisting -and (Test-Path (Join-Path $StudentRoot "student_creation_manifest.json"))) { return }
    $Args = @(
        ".\scripts\phase_9_student_distillation\create_student.py",
        "--source", $StudentSource,
        "--output", $StudentRoot,
        "--mode", $Profile.Mode,
        "--num-hidden-layers", "$($Profile.NumHiddenLayers)",
        "--profile-name", $Profile.Name
    )
    if ($Profile.Mode -eq "custom_qwen_random") {
        $Args += @(
            "--hidden-size", "$($Profile.HiddenSize)",
            "--intermediate-size", "$($Profile.IntermediateSize)",
            "--num-attention-heads", "$($Profile.AttentionHeads)",
            "--num-key-value-heads", "$($Profile.KvHeads)",
            "--head-dim", "$($Profile.HeadDim)",
            "--max-position-embeddings", "$($Profile.MaxPositionEmbeddings)",
            "--acknowledge-random-control"
        )
    } else {
        $Args += @(
            "--hidden-size", "$HiddenSize",
            "--intermediate-size", "$IntermediateSize",
            "--num-attention-heads", "$AttentionHeads",
            "--num-key-value-heads", "$KvHeads",
            "--head-dim", "$HeadDim"
        )
    }
    if ($Profile.TargetParams -gt 0) {
        $Args += @(
            "--target-params", "$($Profile.TargetParams)",
            "--target-params-tolerance", "$($Profile.TargetParamsTolerance)"
        )
    }
    if (Test-Path $StudentRoot) { $Args += "--overwrite" }
    Invoke-PythonStep $StepLabel $Args
}

function Ensure-SingleStudentInitialization {
    $StudentRoot = Join-Path $OutputRoot "student_initialization"
    if ($ReuseExisting -and (Test-Path (Join-Path $StudentRoot "student_creation_manifest.json"))) { return }
    $Args = @(
        ".\scripts\phase_9_student_distillation\create_student.py",
        "--source", $StudentSource,
        "--output", $StudentRoot,
        "--mode", $StudentMode,
        "--num-hidden-layers", "$StudentLayers",
        "--hidden-size", "$HiddenSize",
        "--intermediate-size", "$IntermediateSize",
        "--num-attention-heads", "$AttentionHeads",
        "--num-key-value-heads", "$KvHeads",
        "--head-dim", "$HeadDim"
    )
    if ($AcknowledgeRandomControl) { $Args += "--acknowledge-random-control" }
    if (Test-Path $StudentRoot) { $Args += "--overwrite" }
    Invoke-PythonStep "[1] Creating the explicit student initialization..." $Args
}

function Ensure-KnowledgeIndex {
    if ($ReuseExisting -and (Test-Path $Database)) { return }
    if (-not (Test-Path $Knowledge)) { throw "Missing knowledge JSONL: $Knowledge" }
    if (-not (Test-Path $Procedures)) { throw "Missing procedure JSONL: $Procedures" }
    Invoke-PythonStep "[2] Building the SQLite FTS5 knowledge/procedure index..." @(
        ".\scripts\phase_9_student_distillation\build_knowledge_index.py",
        "--knowledge", $Knowledge,
        "--procedures", $Procedures,
        "--output", $Database
    )
}

function Invoke-RagEvaluation {
    param(
        [string]$Name,
        [string]$Model,
        [string]$Adapter,
        [string]$Destination,
        [string]$TestFile
    )
    if ($ReuseExisting -and (Test-Path (Join-Path $Destination "summary.json"))) { return }
    $Args = @(
        ".\scripts\phase_9_student_distillation\evaluate_rag.py",
        "--model", $Model,
        "--benchmark", $TestFile,
        "--output-dir", $Destination,
        "--knowledge-db", $Database,
        "--gate-calibration", $Calibration,
        "--conditions", "closed_book,oracle_rag,real_rag,adversarial_rag",
        "--max-new-tokens", "32"
    )
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $Args += @("--adapter", $Adapter) }
    Invoke-PythonStep $Name $Args
}

function Invoke-ProfileTrainAndEval {
    param(
        [pscustomobject]$Profile,
        [string]$ProfileRoot,
        [string]$TrainFile,
        [string]$DevFile,
        [string]$TestFile,
        [string]$StepPrefix
    )
    $StudentRoot = Join-Path $ProfileRoot "initialization"
    $SftRoot = Join-Path $ProfileRoot "sft"
    $KdRoot = Join-Path $ProfileRoot "kd"
    $EvalSft = Join-Path $ProfileRoot "evaluation_sft"
    $EvalKd = Join-Path $ProfileRoot "evaluation_kd"

    Ensure-StudentInitializationFromProfile -Profile $Profile -StudentRoot $StudentRoot -StepLabel "$StepPrefix Creating student $($Profile.Name)..."

    if ($Mode -eq "Pilot") {
        if (-not ($ReuseExisting -and (Test-Path (Join-Path $SftRoot "training_manifest.json")))) {
            $SftArgs = @(
                ".\scripts\phase_9_student_distillation\train_student.py",
                "--student", $StudentRoot,
                "--objective", "sft",
                "--train-file", $TrainFile,
                "--dev-file", $DevFile,
                "--output-dir", $SftRoot,
                "--max-steps", "$MaxSteps",
                "--seed", "$Seed"
            )
            if ($UseLora) { $SftArgs += "--lora" }
            if ($Profile.ResearchControl -or $AcknowledgeRandomControl) { $SftArgs += "--allow-random-control" }
            Invoke-PythonStep "$StepPrefix Training SFT for $($Profile.Name)..." $SftArgs
        }
        if (-not $SkipKD -and -not ($ReuseExisting -and (Test-Path (Join-Path $KdRoot "training_manifest.json")))) {
            $KdArgs = @(
                ".\scripts\phase_9_student_distillation\train_student.py",
                "--student", $StudentRoot,
                "--teacher", $TeacherModel,
                "--objective", "kd",
                "--train-file", $TrainFile,
                "--dev-file", $DevFile,
                "--output-dir", $KdRoot,
                "--max-steps", "$MaxSteps",
                "--seed", "$Seed"
            )
            if ($UseLora) { $KdArgs += "--lora" }
            if ($Teacher4Bit) { $KdArgs += "--teacher-4bit" }
            if ($Profile.ResearchControl -or $AcknowledgeRandomControl) { $KdArgs += "--allow-random-control" }
            Invoke-PythonStep "$StepPrefix Training KD for $($Profile.Name)..." $KdArgs
        }
    }

    $SftModel = if ($UseLora) { $StudentRoot } else { Join-Path $SftRoot "model" }
    $SftAdapter = if ($UseLora) { Join-Path $SftRoot "adapter" } else { "" }
    $KdModel = if ($UseLora) { $StudentRoot } else { Join-Path $KdRoot "model" }
    $KdAdapter = if ($UseLora) { Join-Path $KdRoot "adapter" } else { "" }

    Invoke-RagEvaluation "$StepPrefix Evaluating SFT for $($Profile.Name)..." $SftModel $SftAdapter $EvalSft $TestFile
    if (-not $SkipKD) {
        Invoke-RagEvaluation "$StepPrefix Evaluating KD for $($Profile.Name)..." $KdModel $KdAdapter $EvalKd $TestFile
    }

    return [pscustomobject]@{
        Name = $Profile.Name
        ResearchControl = [bool]$Profile.ResearchControl
        SftSummary = Join-Path $EvalSft "summary.json"
        KdSummary = Join-Path $EvalKd "summary.json"
    }
}

if ($Mode -eq "Bootstrap") {
    if ($StudentSuite -eq "Hybrid") {
        $index = 1
        foreach ($profile in $HybridProfiles) {
            $studentRoot = Join-Path $OutputRoot (Join-Path "students" (Join-Path $profile.Name "initialization"))
            Ensure-StudentInitializationFromProfile -Profile $profile -StudentRoot $studentRoot -StepLabel "[$index] Creating hybrid student $($profile.Name)..."
            $index += 1
        }
    } else {
        Ensure-SingleStudentInitialization
    }
    Ensure-KnowledgeIndex
    if (-not (Test-Path $Prompts)) { throw "Missing prompt JSONL: $Prompts" }
    $TeacherArgs = @(
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
    if ($ReuseExisting -and (Test-Path $TeacherCandidates)) {
        Write-Host "Resuming teacher-candidate generation from existing $TeacherCandidates"
    }
    Invoke-PythonStep "[Teacher] Generating the teacher candidate/review packet..." $TeacherArgs
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

$TrainFile = Join-Path $DatasetRoot "train.jsonl"
$DevFile = Join-Path $DatasetRoot "dev.jsonl"
$TestFile = Join-Path $DatasetRoot "test.jsonl"
if ($Mode -eq "Pilot") {
    if (-not ($ReuseExisting -and (Test-Path (Join-Path $DatasetRoot "dataset_manifest.json")))) {
        Invoke-PythonStep "[3] Building document/topic-disjoint pilot splits..." @(
            ".\scripts\phase_9_student_distillation\build_pilot_dataset.py",
            "--reviewed-input", $ReviewedTeacherData,
            "--output-dir", $DatasetRoot,
            "--seed", "$Seed"
        )
    }
    Invoke-PythonStep "[4] Calibrating the external evidence pre-gate on dev only..." @(
        ".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py",
        "--benchmark", $DevFile,
        "--knowledge-db", $Database,
        "--output", $Calibration
    )
}

Invoke-RagEvaluation "[Teacher] Evaluating the teacher under the same RAG conditions..." $TeacherModel "" $EvalTeacher $TestFile

$GateArgs = @(
    ".\scripts\phase_9_student_distillation\check_phase_9_gate.py",
    "--teacher-summary", (Join-Path $EvalTeacher "summary.json")
)

if ($StudentSuite -eq "Hybrid") {
    $profileIndex = 1
    foreach ($profile in $HybridProfiles) {
        $profileRoot = Join-Path $OutputRoot (Join-Path "students" $profile.Name)
        $result = Invoke-ProfileTrainAndEval `
            -Profile $profile `
            -ProfileRoot $profileRoot `
            -TrainFile $TrainFile `
            -DevFile $DevFile `
            -TestFile $TestFile `
            -StepPrefix "[Hybrid $profileIndex/$($HybridProfiles.Count)]"
        $GateArgs += @("--candidate", "sft_$($result.Name)=$($result.SftSummary)")
        if (-not $SkipKD) {
            $GateArgs += @("--candidate", "kd_$($result.Name)=$($result.KdSummary)")
        }
        if ($result.ResearchControl) {
            $GateArgs += @("--research-control", "sft_$($result.Name)")
            if (-not $SkipKD) { $GateArgs += @("--research-control", "kd_$($result.Name)") }
        }
        $profileIndex += 1
    }
} else {
    $StudentRoot = Join-Path $OutputRoot "student_initialization"
    $SftRoot = Join-Path $OutputRoot "student_sft"
    $KdRoot = Join-Path $OutputRoot "student_kd"
    $OnPolicyData = Join-Path $OutputRoot "on_policy_train.jsonl"
    $OnPolicyRoot = Join-Path $OutputRoot "student_on_policy"
    $EvalSft = Join-Path $OutputRoot "evaluation_sft"
    $EvalKd = Join-Path $OutputRoot "evaluation_kd"
    $EvalOnPolicy = Join-Path $OutputRoot "evaluation_on_policy"

    Ensure-SingleStudentInitialization

    if ($Mode -eq "Pilot") {
        if (-not ($ReuseExisting -and (Test-Path (Join-Path $SftRoot "training_manifest.json")))) {
            $SftArgs = @(
                ".\scripts\phase_9_student_distillation\train_student.py",
                "--student", $StudentRoot,
                "--objective", "sft",
                "--train-file", $TrainFile,
                "--dev-file", $DevFile,
                "--output-dir", $SftRoot,
                "--max-steps", "$MaxSteps",
                "--seed", "$Seed"
            )
            if ($UseLora) { $SftArgs += "--lora" }
            if ($AcknowledgeRandomControl) { $SftArgs += "--allow-random-control" }
            Invoke-PythonStep "[5] Training the matched ordinary RAG-SFT candidate..." $SftArgs
        }

        if (-not $SkipKD -and -not ($ReuseExisting -and (Test-Path (Join-Path $KdRoot "training_manifest.json")))) {
            $KdArgs = @(
                ".\scripts\phase_9_student_distillation\train_student.py",
                "--student", $StudentRoot,
                "--teacher", $TeacherModel,
                "--objective", "kd",
                "--train-file", $TrainFile,
                "--dev-file", $DevFile,
                "--output-dir", $KdRoot,
                "--max-steps", "$MaxSteps",
                "--seed", "$Seed"
            )
            if ($UseLora) { $KdArgs += "--lora" }
            if ($Teacher4Bit) { $KdArgs += "--teacher-4bit" }
            if ($AcknowledgeRandomControl) { $KdArgs += "--allow-random-control" }
            Invoke-PythonStep "[6] Training the same-initialization completion-only logit-KD candidate..." $KdArgs
        }
    }

    $SftModel = if ($UseLora) { $StudentRoot } else { Join-Path $SftRoot "model" }
    $SftAdapter = if ($UseLora) { Join-Path $SftRoot "adapter" } else { "" }
    $KdModel = if ($UseLora) { $StudentRoot } else { Join-Path $KdRoot "model" }
    $KdAdapter = if ($UseLora) { Join-Path $KdRoot "adapter" } else { "" }

    Invoke-RagEvaluation "[8] Evaluating the ordinary SFT student..." $SftModel $SftAdapter $EvalSft $TestFile
    $GateArgs += @("--candidate", "sft=$(Join-Path $EvalSft 'summary.json')")
    if (-not $SkipKD) {
        Invoke-RagEvaluation "[9] Evaluating the logit-KD student..." $KdModel $KdAdapter $EvalKd $TestFile
        $GateArgs += @("--candidate", "kd=$(Join-Path $EvalKd 'summary.json')")
    }

    if ($RunOnPolicy) {
        if ($SkipKD) { throw "-RunOnPolicy requires the KD candidate." }
        if (-not ($ReuseExisting -and (Test-Path $OnPolicyData))) {
            $TrajectoryArgs = @(
                ".\scripts\phase_9_student_distillation\generate_on_policy_data.py",
                "--student", $KdModel,
                "--input", $TrainFile,
                "--output", $OnPolicyData,
                "--student-fraction", "$OnPolicyStudentFraction",
                "--seed", "$Seed"
            )
            if (-not [string]::IsNullOrWhiteSpace($KdAdapter)) { $TrajectoryArgs += @("--adapter", $KdAdapter) }
            Invoke-PythonStep "[10] Generating mixed student/teacher trajectories..." $TrajectoryArgs
        }
        if (-not ($ReuseExisting -and (Test-Path (Join-Path $OnPolicyRoot "training_manifest.json")))) {
            $OnPolicyInputModel = if (-not [string]::IsNullOrWhiteSpace($KdAdapter)) { $StudentRoot } else { $KdModel }
            $OnPolicyTrainArgs = @(
                ".\scripts\phase_9_student_distillation\train_student.py",
                "--student", $OnPolicyInputModel,
                "--teacher", $TeacherModel,
                "--objective", "kd",
                "--train-file", $OnPolicyData,
                "--dev-file", $DevFile,
                "--output-dir", $OnPolicyRoot,
                "--max-steps", "$MaxSteps",
                "--seed", "$Seed"
            )
            if (-not [string]::IsNullOrWhiteSpace($KdAdapter)) {
                $OnPolicyTrainArgs += @("--initial-adapter", $KdAdapter)
            }
            if ($Teacher4Bit) { $OnPolicyTrainArgs += "--teacher-4bit" }
            if ($AcknowledgeRandomControl) { $OnPolicyTrainArgs += "--allow-random-control" }
            Invoke-PythonStep "[11] Running the on-policy KD recovery round..." $OnPolicyTrainArgs
        }
        $OnPolicyModel = if ($UseLora) { $StudentRoot } else { Join-Path $OnPolicyRoot "model" }
        $OnPolicyAdapter = if ($UseLora) { Join-Path $OnPolicyRoot "adapter" } else { "" }
        Invoke-RagEvaluation "[12] Evaluating the on-policy KD candidate..." $OnPolicyModel $OnPolicyAdapter $EvalOnPolicy $TestFile
        $GateArgs += @("--candidate", "on_policy=$(Join-Path $EvalOnPolicy 'summary.json')")
    }
}

Invoke-PythonStep "[Final] Building the proportional Phase 9 assessment and human gate..." $GateArgs

Write-Host ""
Write-Host "Phase 9 $Mode ($StudentSuite) completed."
Write-Host "Assessment: reports\PHASE_9_ASSESSMENT.json"
Write-Host "Human gate: reports\PHASE_9_STOP_GATE.md"
Write-Host "Do not promote a random-control or unreviewed-data student as a release model."

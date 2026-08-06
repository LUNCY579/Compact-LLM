<#
.SYNOPSIS
Run the appended Phase 9 tiny-student/RAG distillation workflow.

.DESCRIPTION
Bootstrap creates an explicit student initialization, a low-memory SQLite FTS5
index, and a teacher-candidate human review packet. Pilot consumes only reviewed
teacher rows, builds document/topic-disjoint splits, calibrates the evidence
pre-gate, trains matched SFT and SFT+KD candidates, evaluates teacher and students
under closed/oracle/real/adversarial RAG, and writes a pending human stop gate.
#>
param(
    [ValidateSet("Bootstrap", "Pilot", "Evaluate")][string]$Mode = "Bootstrap",
    [string]$TeacherModel = "",
    [string]$StudentSource = "",
    [ValidateSet("depth_copy", "custom_qwen_random", "pretrained_clone")][string]$StudentMode = "depth_copy",
    [int]$StudentLayers = 12,
    [int]$HiddenSize = 384,
    [int]$IntermediateSize = 1024,
    [int]$AttentionHeads = 6,
    [int]$KvHeads = 2,
    [int]$HeadDim = 64,
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
if ($StudentMode -eq "custom_qwen_random" -and -not $AcknowledgeRandomControl) {
    throw "custom_qwen_random is a research control. Pass -AcknowledgeRandomControl explicitly."
}
if ($StudentMode -eq "pretrained_clone" -and -not $SkipKD) {
    Write-Warning "A pretrained challenger may use a different tokenizer. Direct KD will reject tokenizer mismatch. Use -SkipKD for sequence-level SFT, or supply an exactly tokenizer-compatible source."
}

$StudentRoot = Join-Path $OutputRoot "student_initialization"
$Database = Join-Path $OutputRoot "knowledge.sqlite3"
$TeacherCandidates = Join-Path $OutputRoot "teacher_candidates.jsonl"
$ReviewPacket = Join-Path $OutputRoot "teacher_candidates.review.md"
$DatasetRoot = Join-Path $OutputRoot "pilot_dataset"
$Calibration = Join-Path $OutputRoot "evidence_gate_calibration.json"
$SftRoot = Join-Path $OutputRoot "student_sft"
$KdRoot = Join-Path $OutputRoot "student_kd"
$OnPolicyData = Join-Path $OutputRoot "on_policy_train.jsonl"
$OnPolicyRoot = Join-Path $OutputRoot "student_on_policy"
$EvalTeacher = Join-Path $OutputRoot "evaluation_teacher"
$EvalSft = Join-Path $OutputRoot "evaluation_sft"
$EvalKd = Join-Path $OutputRoot "evaluation_kd"
$EvalOnPolicy = Join-Path $OutputRoot "evaluation_on_policy"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Ensure-StudentInitialization {
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

if ($Mode -eq "Bootstrap") {
    Ensure-StudentInitialization
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
        "--max-new-tokens", "32"
    )
    if ($ReuseExisting) { $TeacherArgs += "--resume" }
    Invoke-PythonStep "[3] Generating the teacher candidate/review packet..." $TeacherArgs
    Write-Host ""
    Write-Host "Bootstrap completed. Human-review $TeacherCandidates before Pilot mode."
    Write-Host "Review packet: $ReviewPacket"
    exit 0
}

Ensure-StudentInitialization
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

function Invoke-RagEvaluation {
    param([string]$Name, [string]$Model, [string]$Adapter, [string]$Destination)
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

Invoke-RagEvaluation "[7] Evaluating the teacher under the same RAG conditions..." $TeacherModel "" $EvalTeacher
Invoke-RagEvaluation "[8] Evaluating the ordinary SFT student..." $SftModel $SftAdapter $EvalSft
if (-not $SkipKD) {
    Invoke-RagEvaluation "[9] Evaluating the logit-KD student..." $KdModel $KdAdapter $EvalKd
}

$OnPolicySummary = ""
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
    Invoke-RagEvaluation "[12] Evaluating the on-policy KD candidate..." $OnPolicyModel $OnPolicyAdapter $EvalOnPolicy
    $OnPolicySummary = Join-Path $EvalOnPolicy "summary.json"
}

$GateArgs = @(
    ".\scripts\phase_9_student_distillation\check_phase_9_gate.py",
    "--teacher-summary", (Join-Path $EvalTeacher "summary.json"),
    "--sft-summary", (Join-Path $EvalSft "summary.json")
)
if (-not $SkipKD) { $GateArgs += @("--kd-summary", (Join-Path $EvalKd "summary.json")) }
if (-not [string]::IsNullOrWhiteSpace($OnPolicySummary)) { $GateArgs += @("--on-policy-summary", $OnPolicySummary) }
Invoke-PythonStep "[Final] Building the proportional Phase 9 assessment and human gate..." $GateArgs

Write-Host ""
Write-Host "Phase 9 $Mode completed."
Write-Host "Assessment: reports\PHASE_9_ASSESSMENT.json"
Write-Host "Human gate: reports\PHASE_9_STOP_GATE.md"
Write-Host "Do not promote a random-control or unreviewed-data student as a release model."

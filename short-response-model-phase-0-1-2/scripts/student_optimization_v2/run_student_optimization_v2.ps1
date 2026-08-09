<#
.SYNOPSIS
Fully orchestrate the accepted student's post-distillation optimization ladder.

.DESCRIPTION
Student Optimization v2.0 is a cross-phase state machine, not a replacement for
Phases 9-11. It adds the missing student-side layer sensitivity, DEV-only context
budget selection, contextual-sparsity/runtime measurement, and DEV-only
quantization backend sweep, while delegating physical structural reduction and
recovery to Phase 10 and final production RAG/release to Phase 11.

The runner writes durable state before/after every expensive stage and exits
cleanly at human checkpoints. Re-run the identical command after human PASS; it
resumes from the checkpoint. It is only marked COMPLETE after the approved Phase
11 package exists, so an intermediate process exit is never misreported as done.
#>
param(
    [ValidateSet("Auto", "Plan", "Report")][string]$Mode = "Auto",
    [string]$Config = ".\configs\student_optimization_v2.json",
    [string]$TeacherModel = "",
    [string]$StudentModel = "",
    [string]$Phase9ReferenceManifest = ".\artifacts\references\phase_9_student_reference.json",
    [string]$Phase9BReferenceManifest = ".\artifacts\references\phase_9b_student_reference.json",
    [string]$Phase10ReferenceManifest = ".\artifacts\references\phase_10_student_reference.json",
    [string]$DatasetRoot = ".\artifacts\phase_9_grounded_recovery_v1_7_5_qwen\dataset",
    [string]$KnowledgeDb = ".\artifacts\phase_9_grounded_recovery_v1_7_5_qwen\knowledge.sqlite3",
    [string]$GateCalibration = ".\artifacts\phase_9_grounded_recovery_v1_7_5_qwen\evidence_gate_calibration.json",
    [string]$Knowledge = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\knowledge.jsonl",
    [string]$Procedures = ".\data\phase_9_student_distillation\rag_reasoning_v3_large\procedures.jsonl",
    [string]$RagReasoningCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$MemoryReconstructionCorpusRoot = ".\data\phase_9b_memory_reconstruction\reconstruction_corpus_v2_large",
    [string]$WorkflowRoot = ".\artifacts\student_optimization_v2",
    [string]$Phase10OutputRoot = ".\artifacts\phase_10_student_architecture_v2",
    [string]$Phase11OutputRoot = ".\artifacts\phase_11_production_rag_v2",
    [string]$MasterReport = ".\reports\STUDENT_OPTIMIZATION_V2_MASTER_REPORT.md",
    [int]$Phase10MaxSteps = 0,
    [int]$Phase10RecoveryTopK = 0,
    [string]$ChallengerSource = "",
    [string]$ChallengerName = "pretrained_compact_challenger",
    [string]$Phase10Candidate = "",
    [ValidateSet("deployment_candidate", "intermediate_teacher", "research_control")][string]$Phase10Role = "deployment_candidate",
    [ValidateSet("off", "prompted", "trained")][string]$MemoryReconstructionMode = "off",
    [int]$ReconstructionMaxNewTokens = 128,
    [switch]$Teacher4Bit,
    [switch]$SkipLayerSensitivity,
    [switch]$SkipContextSweep,
    [switch]$SkipSparseRuntime,
    [switch]$SkipQuantizationSweep,
    [switch]$EnableGGUF,
    [string]$LlamaCppConvertScript = "",
    [string]$LlamaCppQuantizeExe = "",
    [string]$LlamaCli = "",
    [string]$GgufType = "Q4_K_M",
    [string]$PackageCandidate = "",
    [ValidateSet("manifest", "copy")][string]$PackageMode = "copy",
    [switch]$ReuseExisting,
    [switch]$AllowUnapprovedPhase9,
    [switch]$ForceRefinalizePhase10,
    [switch]$ForcePhase10Rerun,
    [switch]$ForcePhase11Rerun
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

$StateFile = Join-Path $WorkflowRoot "WORKFLOW_STATE.json"
$HadExistingState = Test-Path $StateFile
$EffectiveReuse = $ReuseExisting -or $HadExistingState
$StateTool = ".\scripts\student_optimization_v2\state_manager.py"
$CurrentStage = $null

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

function Update-State {
    param(
        [string]$Action,
        [string]$Stage = "",
        [string]$Message = "",
        [string]$Checkpoint = "",
        [string]$Command = "",
        [string[]]$Artifacts = @(),
        [string[]]$Notes = @()
    )
    $Args = @($StateTool, "--state", $StateFile, "--report", $MasterReport, "--action", $Action)
    if (-not [string]::IsNullOrWhiteSpace($Stage)) { $Args += @("--stage", $Stage) }
    if (-not [string]::IsNullOrWhiteSpace($Message)) { $Args += @("--message", $Message) }
    if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) { $Args += @("--checkpoint", $Checkpoint) }
    if (-not [string]::IsNullOrWhiteSpace($Command)) { $Args += @("--command", $Command) }
    foreach ($Artifact in $Artifacts) { if (-not [string]::IsNullOrWhiteSpace($Artifact)) { $Args += @("--artifact", $Artifact) } }
    foreach ($Note in $Notes) { if (-not [string]::IsNullOrWhiteSpace($Note)) { $Args += @("--note", $Note) } }
    & python @Args | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Workflow state update failed." }
}

function Start-Stage {
    param([string]$Stage, [string]$Message, [string]$Command = "")
    $script:CurrentStage = $Stage
    Update-State -Action start -Stage $Stage -Message $Message -Command $Command
}

function Finish-Stage {
    param([string]$Stage, [string]$Message, [string[]]$Artifacts = @(), [string[]]$Notes = @())
    Update-State -Action finish -Stage $Stage -Message $Message -Artifacts $Artifacts -Notes $Notes
    $script:CurrentStage = $null
}

function Skip-Stage {
    param([string]$Stage, [string]$Message)
    Update-State -Action skip -Stage $Stage -Message $Message
}

function Stop-AtCheckpoint {
    param([string]$Stage, [string]$Checkpoint, [string]$Message, [string[]]$Artifacts = @(), [string[]]$Notes = @())
    Update-State -Action checkpoint -Stage $Stage -Checkpoint $Checkpoint -Message $Message -Artifacts $Artifacts -Notes $Notes
    Write-Host ""
    Write-Host "Student Optimization v2.0 stopped at checkpoint: $Checkpoint"
    Write-Host $Message
    Write-Host "Re-run the same command after the required human/materialization action is complete."
    exit 0
}

function Test-HumanGate {
    param([string]$Phase)
    & python .\scripts\common\check_human_gate.py --phase $Phase *> $null
    return ($LASTEXITCODE -eq 0)
}

function Safe-Name {
    param([string]$Value)
    return ($Value -replace '[^A-Za-z0-9_.-]', '_')
}

if (-not (Test-Path $Config)) { throw "Missing Student Optimization v2.0 config: $Config" }
$WorkflowConfig = Get-Content $Config -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $WorkflowRoot | Out-Null

$InputPacket = @{
    config = $Config
    teacher_model = $TeacherModel
    explicit_student_model = $StudentModel
    dataset_root = $DatasetRoot
    knowledge_db = $KnowledgeDb
    gate_calibration = $GateCalibration
    workflow_root = $WorkflowRoot
    phase10_output_root = $Phase10OutputRoot
    phase11_output_root = $Phase11OutputRoot
} | ConvertTo-Json -Compress
& python $StateTool --state $StateFile --report $MasterReport --action init --inputs-json $InputPacket --message "Student Optimization v2.0 initialized/resumed." | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Could not initialize workflow state." }

if ($Mode -eq "Report") {
    & python $StateTool --state $StateFile --report $MasterReport --action render | Out-Host
    exit 0
}
if ($Mode -eq "Plan") {
    Write-Host ""
    Write-Host "Student Optimization v2.0 plan:"
    Write-Host "  S0 accepted-student intake"
    Write-Host "  S1 empirical every-layer deletion sensitivity scan"
    Write-Host "  S2 Phase 10 importance-guided depth/width/GQA reduction + recovery KD"
    Write-Host "  S3 human Phase 10 checkpoint + exact checkpoint materialization"
    Write-Host "  S4 DEV-only context/KV budget sweep"
    Write-Host "  S5 DEV-only contextual sparsity simulation + measured runtime"
    Write-Host "  S6 DEV-only quantization/runtime backend sweep"
    Write-Host "  S7 final Phase 11 frozen backend/context evaluation on sealed TEST"
    Write-Host "  S8 human Phase 11 checkpoint + reproducible release package"
    Write-Host ""
    Write-Host "Live report: $MasterReport"
    exit 0
}

try {
    # ------------------------------------------------------------------ S0
    Start-Stage "S0" "Resolving the exact accepted Phase 9/9B student and validating prerequisite artifacts."

    $UsePhase9B = (Test-Path $Phase9BReferenceManifest)
    $PriorPhase = if ($UsePhase9B) { "9B" } else { "9" }
    $PriorManifest = if ($UsePhase9B) { $Phase9BReferenceManifest } else { $Phase9ReferenceManifest }
    if (-not $AllowUnapprovedPhase9 -and -not (Test-HumanGate $PriorPhase)) {
        Stop-AtCheckpoint "S0" "WAITING_FOR_PHASE_${PriorPhase}_HUMAN_PASS" "The accepted student must have an explicit Phase $PriorPhase human PASS before official post-distillation optimization begins."
    }
    if ([string]::IsNullOrWhiteSpace($StudentModel)) {
        if (-not (Test-Path $PriorManifest)) {
            Stop-AtCheckpoint "S0" "WAITING_FOR_ACCEPTED_STUDENT_MATERIALIZATION" "Human approval exists or research override is active, but the materialized Phase $PriorPhase student reference manifest is missing." -Artifacts @($PriorManifest)
        }
        $StudentModel = (& python .\scripts\phase_10_student_architecture\resolve_phase_9_student.py --manifest $PriorManifest).Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($StudentModel)) { throw "Could not resolve accepted student from $PriorManifest" }
    }
    if (-not (Test-Path $StudentModel)) { throw "Accepted student checkpoint does not exist: $StudentModel" }
    if ([string]::IsNullOrWhiteSpace($TeacherModel)) {
        $TeacherModel = (& python .\scripts\common\resolve_reference_model.py).Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($TeacherModel)) { throw "Could not resolve authoritative teacher; supply -TeacherModel." }
    }
    if (-not (Test-Path $TeacherModel)) { throw "Teacher checkpoint does not exist: $TeacherModel" }
    $TrainFile = Join-Path $DatasetRoot "train.jsonl"
    $DevFile = Join-Path $DatasetRoot "dev.jsonl"
    $TestFile = Join-Path $DatasetRoot "test.jsonl"
    foreach ($Path in @($TrainFile, $DevFile, $TestFile, $KnowledgeDb, $GateCalibration, $Knowledge, $Procedures)) {
        if (-not (Test-Path $Path)) { throw "Missing Student Optimization input: $Path" }
    }
    Finish-Stage "S0" "Accepted student and authoritative teacher resolved." -Artifacts @($PriorManifest, $StudentModel) -Notes @("Prior accepted phase: $PriorPhase", "Sealed TEST is not used for S1/S4/S5/S6 selection.")

    # ------------------------------------------------------------------ S1
    $SensitivityRoot = Join-Path $WorkflowRoot "layer_sensitivity"
    $SensitivityJson = Join-Path $SensitivityRoot "layer_sensitivity.json"
    $StructureProfile = Join-Path $SensitivityRoot "student_structure_profile.pt"
    if ($SkipLayerSensitivity -or -not [bool]$WorkflowConfig.layer_sensitivity.enabled) {
        Skip-Stage "S1" "Layer sensitivity was explicitly disabled. Phase 10 will use its legacy uniform-depth controls/config."
        $Phase10Config = ".\configs\phase_10_student_architecture.json"
        $SensitivityJson = ""
        $StructureProfile = ""
    } else {
        $Phase10Config = [string]$WorkflowConfig.phase10.config
        Start-Stage "S1" "Scanning every student decoder layer as a physical single-layer deletion on DEV." -Command "scan_student_layer_sensitivity.py --model <accepted-student> --benchmark <dev>"
        $ScanArgs = @(
            ".\scripts\student_optimization_v2\scan_student_layer_sensitivity.py",
            "--model", $StudentModel,
            "--profile-data", $TrainFile,
            "--benchmark", $DevFile,
            "--knowledge-db", $KnowledgeDb,
            "--gate-calibration", $GateCalibration,
            "--output-root", $SensitivityRoot,
            "--limit", "$([int]$WorkflowConfig.layer_sensitivity.dev_limit)",
            "--max-profile-records", "$([int]$WorkflowConfig.layer_sensitivity.profile_records)",
            "--conditions", ([string]$WorkflowConfig.layer_sensitivity.conditions)
        )
        if ($EffectiveReuse) { $ScanArgs += "--reuse-existing" }
        if ([bool]$WorkflowConfig.layer_sensitivity.keep_temporary_candidates) { $ScanArgs += "--keep-candidates" }
        Invoke-PythonStep "[S1] Student every-layer sensitivity scan..." $ScanArgs
        Finish-Stage "S1" "Every decoder layer was deletion-tested and ranked without opening sealed TEST." -Artifacts @($SensitivityJson, (Join-Path $SensitivityRoot "LAYER_SENSITIVITY_REPORT.md"), $StructureProfile)
    }

    # ------------------------------------------------------------------ S2
    # A completed Phase 10 registry is the durable evidence that all automated
    # Phase 10 work reached its human checkpoint. On resume we MUST NOT call
    # Phase 10 Full again, because its gate builder intentionally writes a fresh
    # PENDING human gate and would erase a reviewer-recorded PASS.
    $Phase10Registry = Join-Path $Phase10OutputRoot "candidate_registry.json"
    Start-Stage "S2" "Running Phase 10 structural reduction, raw DEV evaluation, bounded recovery KD, final DEV freeze, and Phase 10 sealed comparison." -Command "run_phase_10.ps1 -Mode Full"
    if ((Test-Path $Phase10Registry) -and -not $ForcePhase10Rerun) {
        Finish-Stage "S2" "Existing completed Phase 10 registry reused; automated Phase 10 work was not rerun, preserving the human gate." -Artifacts @($Phase10Registry, ".\reports\PHASE_10_CANDIDATE_REPORT.md", ".\reports\PHASE_10_STOP_GATE.md") -Notes @("Use -ForcePhase10Rerun only when intentionally rebuilding Phase 10 evidence; doing so will require a fresh human review.")
    } else {
        if ($Phase10MaxSteps -le 0) { $Phase10MaxSteps = [int]$WorkflowConfig.phase10.max_steps }
        if ($Phase10RecoveryTopK -le 0) { $Phase10RecoveryTopK = [int]$WorkflowConfig.phase10.recovery_top_k }
        $Phase10Args = @(
        ".\scripts\phase_10_student_architecture\run_phase_10.ps1",
        "-Mode", "Full",
        "-StudentModel", $StudentModel,
        "-TeacherModel", $TeacherModel,
        "-Config", $Phase10Config,
        "-DatasetRoot", $DatasetRoot,
        "-KnowledgeDb", $KnowledgeDb,
        "-GateCalibration", $GateCalibration,
        "-OutputRoot", $Phase10OutputRoot,
        "-MaxSteps", "$Phase10MaxSteps",
        "-RecoveryTopK", "$Phase10RecoveryTopK",
        "-UseLora"
    )
    if (-not [string]::IsNullOrWhiteSpace($StructureProfile)) { $Phase10Args += @("-StructureProfile", $StructureProfile) }
    if (-not [string]::IsNullOrWhiteSpace($SensitivityJson)) { $Phase10Args += @("-LayerSensitivityScan", $SensitivityJson) }
    if ($Teacher4Bit) { $Phase10Args += "-Teacher4Bit" }
    if (-not [string]::IsNullOrWhiteSpace($ChallengerSource)) { $Phase10Args += @("-ChallengerSource", $ChallengerSource, "-ChallengerName", $ChallengerName) }
    if ($EffectiveReuse) { $Phase10Args += "-ReuseExisting" }
    if ($AllowUnapprovedPhase9) { $Phase10Args += "-AllowUnapprovedPhase9" }
    Write-Host ""
    Write-Host "[S2] Calling official Phase 10..."
    $Phase10Command = $Phase10Args[0]
    $Phase10Rest = @($Phase10Args | Select-Object -Skip 1)
    & $Phase10Command @Phase10Rest
        if ($LASTEXITCODE -ne 0) { throw "Phase 10 runner failed with exit code $LASTEXITCODE." }
        if (-not (Test-Path $Phase10Registry)) { throw "Phase 10 completed without candidate registry: $Phase10Registry" }
        Finish-Stage "S2" "Phase 10 completed to its human checkpoint." -Artifacts @($Phase10Registry, ".\reports\PHASE_10_CANDIDATE_REPORT.md", ".\reports\PHASE_10_STOP_GATE.md")
    }

    # ------------------------------------------------------------------ S3
    Start-Stage "S3" "Checking the Phase 10 human decision and materializing the exact approved structural student."
    if (-not (Test-HumanGate "10")) {
        Stop-AtCheckpoint "S3" "WAITING_FOR_PHASE10_HUMAN_PASS" "All Phase 10 automated structural/recovery work is finished. Review the registry/raw outputs/sealed comparison, record an explicit PASS with the exact accepted candidate, then rerun this master workflow." -Artifacts @($Phase10Registry, ".\reports\PHASE_10_STOP_GATE.md")
    }
    if (-not (Test-Path $Phase10ReferenceManifest) -or $ForceRefinalizePhase10) {
        $RegistryJson = Get-Content $Phase10Registry -Raw | ConvertFrom-Json
        $CandidateToFinalize = if (-not [string]::IsNullOrWhiteSpace($Phase10Candidate)) { $Phase10Candidate } else { [string]$RegistryJson.recommended_candidate }
        if ([string]::IsNullOrWhiteSpace($CandidateToFinalize)) {
            Stop-AtCheckpoint "S3" "WAITING_FOR_PHASE10_CANDIDATE_NAME" "Phase 10 has human PASS but no automated recommended candidate. Re-run with -Phase10Candidate <exact registry key> matching the human decision."
        }
        $FinalizeArgs = @(
            ".\scripts\phase_10_student_architecture\finalize_phase_10_selection.py",
            "--registry", $Phase10Registry,
            "--candidate", $CandidateToFinalize,
            "--role", $Phase10Role,
            "--reason", "Approved Student Optimization v2.0 Phase 10 structural/recovery candidate after governed DEV selection and sealed comparison."
        )
        if ($ForceRefinalizePhase10) { $FinalizeArgs += "--overwrite" }
        Invoke-PythonStep "[S3] Materializing approved Phase 10 student..." $FinalizeArgs
    }
    if (-not (Test-Path $Phase10ReferenceManifest)) { throw "Phase 10 reference manifest still missing after finalization." }
    $Phase10Model = (& python .\scripts\phase_10_student_architecture\resolve_phase_10_student.py --manifest $Phase10ReferenceManifest).Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Phase10Model)) { throw "Could not resolve materialized Phase 10 model." }
    Finish-Stage "S3" "Human-approved Phase 10 student materialized for deployment optimization." -Artifacts @($Phase10ReferenceManifest, $Phase10Model)

    # ------------------------------------------------------------------ Shared runtime snapshot for S4/S6
    $RuntimeLabRoot = Join-Path $WorkflowRoot "runtime_lab"
    $SnapshotRoot = Join-Path $RuntimeLabRoot "knowledge_snapshot"
    $SnapshotManifest = Join-Path $SnapshotRoot "snapshot_manifest.json"
    $RuntimeKnowledgeDb = Join-Path $SnapshotRoot "knowledge.sqlite3"
    if (-not ($EffectiveReuse -and (Test-Path $SnapshotManifest) -and (Test-Path $RuntimeKnowledgeDb))) {
        $SnapshotArgs = @(
            ".\scripts\phase_11_production_rag\build_knowledge_snapshot.py",
            "--knowledge", $Knowledge,
            "--procedures", $Procedures,
            "--output-dir", $SnapshotRoot,
            "--snapshot-name", "student_optimization_v2"
        )
        if (Test-Path $SnapshotRoot) { $SnapshotArgs += "--overwrite" }
        Invoke-PythonStep "[Runtime lab] Building immutable DEV-selection knowledge snapshot..." $SnapshotArgs
    }

    # ------------------------------------------------------------------ S4
    $ContextRoot = Join-Path $WorkflowRoot "context_optimization"
    $ContextSelection = Join-Path $ContextRoot "context_selection.json"
    if ($SkipContextSweep -or -not [bool]$WorkflowConfig.context_sweep.enabled) {
        Start-Stage "S4" "Context sweep disabled; creating default full-context calibration for downstream DEV runtime experiments."
        $DefaultCalibration = Join-Path $ContextRoot "default_answerability_calibration.json"
        New-Item -ItemType Directory -Force -Path $ContextRoot | Out-Null
        if (-not ($EffectiveReuse -and (Test-Path $DefaultCalibration))) {
            Invoke-PythonStep "[S4] Calibrating default answerability profile..." @(
                ".\scripts\phase_11_production_rag\calibrate_answerability.py",
                "--benchmark", $DevFile,
                "--knowledge-db", $RuntimeKnowledgeDb,
                "--output", $DefaultCalibration,
                "--retrieval-top-k", "20",
                "--selected-top-k", "3",
                "--false-answer-cost", "3.0",
                "--false-abstention-cost", "1.0"
            )
        }
        $DefaultPacket = @{
            schema_version = "2.0"; selected_profile = "reference";
            selected = @{ name="reference"; retrieval_top_k=20; selected_top_k=3; maximum_evidence_tokens=512; calibration=$DefaultCalibration };
            selection_benchmark = $DevFile; calibration_benchmark = $DevFile; sealed_test_used = $false
        } | ConvertTo-Json -Depth 6
        $DefaultPacket | Set-Content -Encoding UTF8 $ContextSelection
        Finish-Stage "S4" "Context sweep skipped; reference context profile frozen." -Artifacts @($ContextSelection, $DefaultCalibration)
    } else {
        Start-Stage "S4" "Sweeping retrieval/evidence budgets on disjoint DEV calibration and selection groups." -Command "context_budget_sweep.py"
        $ContextArgs = @(
            ".\scripts\student_optimization_v2\context_budget_sweep.py",
            "--model", $Phase10Model,
            "--dev-benchmark", $DevFile,
            "--knowledge-db", $RuntimeKnowledgeDb,
            "--config", $Config,
            "--output-root", $ContextRoot,
            "--memory-reconstruction-mode", $MemoryReconstructionMode,
            "--reconstruction-max-new-tokens", "$ReconstructionMaxNewTokens"
        )
        if ($EffectiveReuse) { $ContextArgs += "--reuse-existing" }
        Invoke-PythonStep "[S4] Context/KV-budget optimization..." $ContextArgs
        Finish-Stage "S4" "A context profile was frozen using non-sealed DEV evidence." -Artifacts @($ContextSelection, (Join-Path $ContextRoot "CONTEXT_BUDGET_REPORT.md"))
    }
    $ContextJson = Get-Content $ContextSelection -Raw | ConvertFrom-Json
    $SelectedContext = $ContextJson.selected
    $RetrievalTopK = [int]$SelectedContext.retrieval_top_k
    $SelectedTopK = [int]$SelectedContext.selected_top_k
    $MaximumEvidenceTokens = [int]$SelectedContext.maximum_evidence_tokens
    $ContextCalibration = [string]$SelectedContext.calibration
    $ContextSelectionBenchmark = [string]$ContextJson.selection_benchmark

    # ------------------------------------------------------------------ S5
    $SparseRoot = Join-Path $WorkflowRoot "sparse_runtime"
    $SparseSelection = Join-Path $SparseRoot "sparse_selection.json"
    if ($SkipSparseRuntime -or -not [bool]$WorkflowConfig.sparse_runtime.enabled) {
        Skip-Stage "S5" "Contextual sparsity/runtime research was explicitly disabled; dense runtime remains the deployment baseline."
    } else {
        Start-Stage "S5" "Simulating student contextual MLP sparsity, then measuring router runtime on DEV." -Command "sparse_student_pipeline.py"
        $Ratios = @($WorkflowConfig.sparse_runtime.ratios | ForEach-Object { "$_" })
        $SparseArgs = @(
            ".\scripts\student_optimization_v2\sparse_student_pipeline.py",
            "--model", $Phase10Model,
            "--calibration", $TrainFile,
            "--benchmark", $ContextSelectionBenchmark,
            "--output-root", $SparseRoot,
            "--block-size", "$([int]$WorkflowConfig.sparse_runtime.block_size)",
            "--oracle-label-ratio", "$([double]$WorkflowConfig.sparse_runtime.oracle_label_ratio)",
            "--max-calibration-records", "$([int]$WorkflowConfig.sparse_runtime.max_calibration_records)",
            "--max-benchmark-cases", "$([int]$WorkflowConfig.sparse_runtime.max_benchmark_cases)",
            "--router-epochs", "$([int]$WorkflowConfig.sparse_runtime.router_epochs)",
            "--router-width", "$([int]$WorkflowConfig.sparse_runtime.router_width)",
            "--runtime-repeats", "$([int]$WorkflowConfig.sparse_runtime.runtime_repeats)",
            "--ratios"
        ) + $Ratios
        if ($EffectiveReuse) { $SparseArgs += "--reuse-existing" }
        Invoke-PythonStep "[S5] Student sparse simulation/runtime..." $SparseArgs
        $SparseJson = Get-Content $SparseSelection -Raw | ConvertFrom-Json
        Finish-Stage "S5" "Sparse runtime research completed; dense fallback remains mandatory." -Artifacts @($SparseSelection, (Join-Path $SparseRoot "SPARSE_RUNTIME_REPORT.md")) -Notes @("Recommendation: $($SparseJson.runtime_recommendation)", "Sparse feature-flag evidence is not silently inserted into the Phase 11 package.")
    }

    # ------------------------------------------------------------------ S6
    $QuantRoot = Join-Path $WorkflowRoot "quantization_sweep"
    $QuantSelection = Join-Path $QuantRoot "quantization_selection.json"
    if ($SkipQuantizationSweep -or -not [bool]$WorkflowConfig.quantization_sweep.enabled) {
        Start-Stage "S6" "Quantization sweep disabled; dense_copy frozen for the final Phase 11 experiment."
        New-Item -ItemType Directory -Force -Path $QuantRoot | Out-Null
        @{schema_version="2.0"; selected_backend="dense_copy"; sealed_test_used=$false; candidates=@(); failures=@()} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $QuantSelection
        Finish-Stage "S6" "Dense backend frozen by configuration." -Artifacts @($QuantSelection)
    } else {
        Start-Stage "S6" "Sweeping supported deployment quantization backends on the frozen DEV context profile." -Command "quantization_sweep.py"
        $Backends = @($WorkflowConfig.quantization_sweep.backends | ForEach-Object { [string]$_ })
        if ($EnableGGUF) { $Backends += "llamacpp_gguf" }
        $QuantArgs = @(
            ".\scripts\student_optimization_v2\quantization_sweep.py",
            "--model", $Phase10Model,
            "--benchmark", $ContextSelectionBenchmark,
            "--knowledge-db", $RuntimeKnowledgeDb,
            "--answerability-calibration", $ContextCalibration,
            "--snapshot-manifest", $SnapshotManifest,
            "--output-root", $QuantRoot,
            "--retrieval-top-k", "$RetrievalTopK",
            "--selected-top-k", "$SelectedTopK",
            "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
            "--memory-reconstruction-mode", $MemoryReconstructionMode,
            "--reconstruction-max-new-tokens", "$ReconstructionMaxNewTokens",
            "--gguf-type", $GgufType,
            "--backends"
        ) + $Backends
        if ($EnableGGUF) {
            $QuantArgs += @("--llama-cpp-convert-script", $LlamaCppConvertScript, "--llama-cpp-quantize-exe", $LlamaCppQuantizeExe, "--llama-cli", $LlamaCli)
        }
        if ($EffectiveReuse) { $QuantArgs += "--reuse-existing" }
        Invoke-PythonStep "[S6] Quantization/runtime DEV sweep..." $QuantArgs
        $QuantJson = Get-Content $QuantSelection -Raw | ConvertFrom-Json
        Finish-Stage "S6" "One deployment backend was frozen on DEV; dense remains rollback." -Artifacts @($QuantSelection, (Join-Path $QuantRoot "QUANTIZATION_SWEEP_REPORT.md")) -Notes @("Selected backend: $($QuantJson.selected_backend)")
    }
    $QuantJson = Get-Content $QuantSelection -Raw | ConvertFrom-Json
    $SelectedBackend = [string]$QuantJson.selected_backend
    if ([string]::IsNullOrWhiteSpace($SelectedBackend)) { $SelectedBackend = "dense_copy" }

    # ------------------------------------------------------------------ S7
    $ContextName = Safe-Name ([string]$ContextJson.selected_profile)
    $BackendName = Safe-Name $SelectedBackend
    $FinalPhase11Root = Join-Path $Phase11OutputRoot "final_${ContextName}_${BackendName}"
    $Phase11Registry = Join-Path $FinalPhase11Root "release_candidate_registry.json"
    Start-Stage "S7" "Running the frozen context + DEV-selected backend through the official Phase 11 dense-vs-deployment sealed runtime comparison." -Command "run_phase_11.ps1 -Mode Full"
    # As with Phase 10, a completed Phase 11 registry means the automated sealed
    # comparison has already reached the human checkpoint. Re-running Full here
    # would regenerate a PENDING gate and could overwrite a human PASS.
    if ((Test-Path $Phase11Registry) -and -not $ForcePhase11Rerun) {
        Finish-Stage "S7" "Existing completed Phase 11 registry reused; sealed runtime evaluation was not rerun and the human gate was preserved." -Artifacts @($Phase11Registry, ".\reports\PHASE_11_RELEASE_CANDIDATE_REPORT.md", ".\reports\PHASE_11_STOP_GATE.md") -Notes @("Use -ForcePhase11Rerun only when intentionally creating a fresh sealed-evaluation cycle and fresh human review.")
    } else {
        $Phase11Args = @(
        ".\scripts\phase_11_production_rag\run_phase_11.ps1",
        "-Mode", "Full",
        "-RagReasoningCorpusRoot", $RagReasoningCorpusRoot,
        "-MemoryReconstructionCorpusRoot", $MemoryReconstructionCorpusRoot,
        "-StudentModel", $Phase10Model,
        "-Phase10ReferenceManifest", $Phase10ReferenceManifest,
        "-Knowledge", $Knowledge,
        "-Procedures", $Procedures,
        "-DevBenchmark", $DevFile,
        "-Benchmark", $TestFile,
        "-OutputRoot", $FinalPhase11Root,
        "-QuantizationBackend", $SelectedBackend,
        "-RetrievalTopK", "$RetrievalTopK",
        "-SelectedTopK", "$SelectedTopK",
        "-MaximumEvidenceTokens", "$MaximumEvidenceTokens",
        "-MemoryReconstructionMode", $MemoryReconstructionMode,
        "-ReconstructionMaxNewTokens", "$ReconstructionMaxNewTokens",
        "-GgufType", $GgufType
    )
    if ($SelectedBackend -eq "llamacpp_gguf") {
        $Phase11Args += @("-LlamaCppConvertScript", $LlamaCppConvertScript, "-LlamaCppQuantizeExe", $LlamaCppQuantizeExe, "-LlamaCli", $LlamaCli)
    }
    if ($EffectiveReuse) { $Phase11Args += "-ReuseExisting" }
    Write-Host ""
    Write-Host "[S7] Calling official Phase 11 final frozen experiment..."
    $Phase11Command = $Phase11Args[0]
    $Phase11Rest = @($Phase11Args | Select-Object -Skip 1)
    & $Phase11Command @Phase11Rest
        if ($LASTEXITCODE -ne 0) { throw "Phase 11 final runner failed with exit code $LASTEXITCODE." }
        if (-not (Test-Path $Phase11Registry)) { throw "Phase 11 final registry missing: $Phase11Registry" }
        Finish-Stage "S7" "Final Phase 11 sealed runtime comparison completed; no tuning may use these TEST results." -Artifacts @($Phase11Registry, ".\reports\PHASE_11_RELEASE_CANDIDATE_REPORT.md", ".\reports\PHASE_11_STOP_GATE.md") -Notes @("Frozen context: $($ContextJson.selected_profile)", "Frozen backend from DEV: $SelectedBackend")
    }

    # ------------------------------------------------------------------ S8
    Start-Stage "S8" "Checking Phase 11 human commercial/release decision and packaging the exact approved runtime."
    if (-not (Test-HumanGate "11")) {
        Stop-AtCheckpoint "S8" "WAITING_FOR_PHASE11_HUMAN_PASS" "All automated optimization and final sealed runtime evidence is complete. Review Phase 11 outputs, record explicit human PASS with the exact deployment candidate, then rerun the master workflow to package it." -Artifacts @($Phase11Registry, ".\reports\PHASE_11_STOP_GATE.md")
    }
    $RegistryJson = Get-Content $Phase11Registry -Raw | ConvertFrom-Json
    $CandidateToPackage = if (-not [string]::IsNullOrWhiteSpace($PackageCandidate)) { $PackageCandidate } elseif (-not [string]::IsNullOrWhiteSpace([string]$RegistryJson.recommended_candidate)) { [string]$RegistryJson.recommended_candidate } else { "dense_reference" }
    $PackageArgs = @(
        ".\scripts\phase_11_production_rag\run_phase_11.ps1",
        "-Mode", "Package",
        "-OutputRoot", $FinalPhase11Root,
        "-Phase10ReferenceManifest", $Phase10ReferenceManifest,
        "-PackageCandidate", $CandidateToPackage,
        "-PackageMode", $PackageMode,
        "-RagReasoningCorpusRoot", $RagReasoningCorpusRoot,
        "-MemoryReconstructionCorpusRoot", $MemoryReconstructionCorpusRoot
    )
    if ($EffectiveReuse) { $PackageArgs += "-ReuseExisting" }
    Write-Host ""
    Write-Host "[S8] Packaging human-approved Phase 11 candidate $CandidateToPackage..."
    $PackageCommand = $PackageArgs[0]
    $PackageRest = @($PackageArgs | Select-Object -Skip 1)
    & $PackageCommand @PackageRest
    if ($LASTEXITCODE -ne 0) { throw "Phase 11 package step failed with exit code $LASTEXITCODE." }
    $ReleaseZip = ".\artifacts\phase_11_release.zip"
    if (-not (Test-Path $ReleaseZip)) { throw "Release package command completed but expected ZIP is missing: $ReleaseZip" }
    Finish-Stage "S8" "Approved release package created and rollback provenance retained." -Artifacts @($ReleaseZip, ".\artifacts\phase_11_release") -Notes @("Packaged candidate: $CandidateToPackage")
    Update-State -Action complete -Message "Student Optimization v2.0 has no further automated work: Phase 11 release package exists after human PASS."

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "STUDENT OPTIMIZATION v2.0 COMPLETE"
    Write-Host "============================================================"
    Write-Host "Master report: $MasterReport"
    Write-Host "State: $StateFile"
    Write-Host "Release package: $ReleaseZip"
}
catch {
    if ($null -ne $CurrentStage) {
        try { Update-State -Action fail -Stage $CurrentStage -Message $_.Exception.Message } catch { }
    }
    throw
}

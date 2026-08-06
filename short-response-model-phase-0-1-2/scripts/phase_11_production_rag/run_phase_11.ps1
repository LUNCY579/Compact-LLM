<#
.SYNOPSIS
Run Phase 11 production RAG preparation, quantization, runtime benchmarking, gate, and packaging.

.DESCRIPTION
Phase 11 consumes the exact Phase 10 human-approved student. It creates an
immutable knowledge/procedure snapshot, calibrates a question-plus-evidence
answerability gate, benchmarks the complete route/retrieve/gate/generate/ground
request path, optionally quantizes the model, compares dense and deployment
runtimes using measured process memory and quality, and writes a pending human
release gate. Packaging is a separate post-approval action.
#>
param(
    [ValidateSet("Prepare", "Benchmark", "Full", "Package")][string]$Mode = "Full",
    [string]$StudentModel = "",
    [string]$Phase10ReferenceManifest = ".\artifacts\references\phase_10_student_reference.json",
    [string]$Config = ".\configs\phase_11_production_rag.json",
    [string]$Knowledge = ".\data\phase_9_student_distillation\knowledge.sample.jsonl",
    [string]$Procedures = ".\data\phase_9_student_distillation\procedures.sample.jsonl",
    [string]$DevBenchmark = ".\artifacts\phase_9_student_distillation\pilot_dataset\dev.jsonl",
    [string]$Benchmark = ".\artifacts\phase_9_student_distillation\pilot_dataset\test.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_11_production_rag",
    [ValidateSet("dense_copy", "bitsandbytes_nf4", "bitsandbytes_int8", "llamacpp_gguf")][string]$QuantizationBackend = "dense_copy",
    [string]$LlamaCppConvertScript = "",
    [string]$LlamaCppQuantizeExe = "",
    [string]$LlamaCli = "",
    [string]$GgufType = "Q4_K_M",
    [string]$FallbackModel = "",
    [ValidateSet("abstain", "fallback")][string]$HighRiskPolicy = "abstain",
    [ValidateSet("clarify", "abstain", "fallback")][string]$ComplexPolicy = "clarify",
    [int]$RetrievalTopK = 0,
    [int]$SelectedTopK = 0,
    [int]$MaximumEvidenceTokens = 0,
    [int]$MaxSequenceLength = 0,
    [int]$MaxNewTokens = 0,
    [string]$ResponseControl = "",
    [double]$UnsupportedConfidentAnswerCap = -1.0,
    [int]$Limit = 0,
    [string]$PackageCandidate = "",
    [ValidateSet("manifest", "copy")][string]$PackageMode = "manifest",
    [switch]$ReuseExisting,
    [switch]$AllowUnapprovedPhase10
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

if (-not $AllowUnapprovedPhase10) {
    Invoke-PythonStep "[Gate] Requiring the accepted Phase 10 student..." @(
        ".\scripts\common\check_human_gate.py", "--phase", "10"
    )
}
if (-not (Test-Path $Config)) { throw "Missing Phase 11 config: $Config" }
$Phase11Config = Get-Content $Config -Raw | ConvertFrom-Json
if ($RetrievalTopK -le 0) { $RetrievalTopK = [int]$Phase11Config.retrieval.retrieval_top_k }
if ($SelectedTopK -le 0) { $SelectedTopK = [int]$Phase11Config.retrieval.selected_top_k }
if ($MaximumEvidenceTokens -le 0) { $MaximumEvidenceTokens = [int]$Phase11Config.retrieval.maximum_evidence_tokens }
if ($MaxSequenceLength -le 0) { $MaxSequenceLength = [int]$Phase11Config.generation.max_sequence_length }
if ($MaxNewTokens -le 0) { $MaxNewTokens = [int]$Phase11Config.generation.max_new_tokens }
if ([string]::IsNullOrWhiteSpace($ResponseControl)) { $ResponseControl = [string]$Phase11Config.generation.response_control }
if ($ResponseControl -notin @("eos_only", "one_sentence")) { throw "Unsupported response-control mode: $ResponseControl" }
if ($UnsupportedConfidentAnswerCap -gt 1.0) { throw "UnsupportedConfidentAnswerCap must be in [0,1] or negative to remain unset." }

# Packaging consumes the already-approved registry/runtime evidence. It does not
# require the original training datasets or a live model load.
if ($Mode -eq "Package") {
    $Registry = Join-Path $OutputRoot "release_candidate_registry.json"
    if ([string]::IsNullOrWhiteSpace($PackageCandidate)) {
        if (-not (Test-Path $Registry)) { throw "Missing release registry: $Registry" }
        $RegistryJson = Get-Content $Registry -Raw | ConvertFrom-Json
        $PackageCandidate = [string]$RegistryJson.recommended_candidate
        if ([string]::IsNullOrWhiteSpace($PackageCandidate)) {
            throw "No automated recommendation exists. Supply -PackageCandidate explicitly, such as dense_reference after a human-approved negative quantization result."
        }
    }
    $PackageArgs = @(
        ".\scripts\phase_11_production_rag\package_phase_11_release.py",
        "--registry", $Registry,
        "--candidate", $PackageCandidate,
        "--phase10-reference", $Phase10ReferenceManifest,
        "--mode", $PackageMode,
        "--output-dir", ".\artifacts\phase_11_release",
        "--zip", ".\artifacts\phase_11_release.zip"
    )
    if ($ReuseExisting) { $PackageArgs += "--overwrite" }
    Invoke-PythonStep "[Package] Building the approved Phase 11 release package..." $PackageArgs
    exit 0
}
if ([string]::IsNullOrWhiteSpace($StudentModel)) {
    $StudentModel = python .\scripts\phase_10_student_architecture\resolve_phase_10_student.py --manifest $Phase10ReferenceManifest
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($StudentModel)) {
        throw "Could not resolve the accepted Phase 10 student. Finalize Phase 10 or pass -StudentModel."
    }
}
if (-not (Test-Path $StudentModel)) { throw "Missing Phase 10 student checkpoint: $StudentModel" }
foreach ($Path in @($Knowledge, $Procedures, $DevBenchmark, $Benchmark)) {
    if (-not (Test-Path $Path)) { throw "Missing Phase 11 input: $Path" }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$SnapshotRoot = Join-Path $OutputRoot "knowledge_snapshot"
$SnapshotManifest = Join-Path $SnapshotRoot "snapshot_manifest.json"
$KnowledgeDb = Join-Path $SnapshotRoot "knowledge.sqlite3"
$AnswerabilityCalibration = Join-Path $OutputRoot "answerability_calibration.json"
$DenseEval = Join-Path $OutputRoot "runtime_dense_reference"
$DeploymentArtifact = Join-Path $OutputRoot "deployment_artifact"
$QuantizationManifest = Join-Path $DeploymentArtifact "phase_11_quantization_manifest.json"
$DeploymentEval = Join-Path $OutputRoot "runtime_deployment_candidate"
$Registry = Join-Path $OutputRoot "release_candidate_registry.json"


if ($Mode -in @("Prepare", "Full")) {
    if (-not ($ReuseExisting -and (Test-Path $SnapshotManifest))) {
        $SnapshotArgs = @(
            ".\scripts\phase_11_production_rag\build_knowledge_snapshot.py",
            "--knowledge", $Knowledge,
            "--procedures", $Procedures,
            "--output-dir", $SnapshotRoot,
            "--snapshot-name", "phase11_production"
        )
        if (Test-Path $SnapshotRoot) { $SnapshotArgs += "--overwrite" }
        Invoke-PythonStep "[1] Building immutable production knowledge/procedure snapshot..." $SnapshotArgs
    }
    if (-not ($ReuseExisting -and (Test-Path $AnswerabilityCalibration))) {
        Invoke-PythonStep "[2] Calibrating question-plus-evidence answerability..." @(
            ".\scripts\phase_11_production_rag\calibrate_answerability.py",
            "--benchmark", $DevBenchmark,
            "--knowledge-db", $KnowledgeDb,
            "--output", $AnswerabilityCalibration,
            "--retrieval-top-k", "$RetrievalTopK",
            "--selected-top-k", "$SelectedTopK",
            "--false-answer-cost", "$($Phase11Config.answerability.false_answer_cost)",
            "--false-abstention-cost", "$($Phase11Config.answerability.false_abstention_cost)"
        )
    }
}

if ($Mode -eq "Prepare") {
    Write-Host ""
    Write-Host "Phase 11 preparation completed. Snapshot and calibration are ready; no model benchmark or gate was run."
    exit 0
}

foreach ($Path in @($SnapshotManifest, $KnowledgeDb, $AnswerabilityCalibration)) {
    if (-not (Test-Path $Path)) { throw "Missing prepared Phase 11 artifact: $Path" }
}

function Invoke-ProductionBenchmark {
    param(
        [string]$Name,
        [string]$Model,
        [string]$RuntimeBackend,
        [string]$TokenizerSource,
        [string]$Destination
    )
    if ($ReuseExisting -and (Test-Path (Join-Path $Destination "summary.json"))) { return }
    $Args = @(
        ".\scripts\phase_11_production_rag\benchmark_production_runtime.py",
        "--model", $Model,
        "--runtime-backend", $RuntimeBackend,
        "--benchmark", $Benchmark,
        "--knowledge-db", $KnowledgeDb,
        "--answerability-calibration", $AnswerabilityCalibration,
        "--output-dir", $Destination,
        "--high-risk-policy", $HighRiskPolicy,
        "--complex-policy", $ComplexPolicy,
        "--retrieval-top-k", "$RetrievalTopK",
        "--selected-top-k", "$SelectedTopK",
        "--maximum-evidence-tokens", "$MaximumEvidenceTokens",
        "--max-sequence-length", "$MaxSequenceLength",
        "--max-new-tokens", "$MaxNewTokens",
        "--response-control", $ResponseControl
    )
    if (-not [string]::IsNullOrWhiteSpace($TokenizerSource)) { $Args += @("--tokenizer-source", $TokenizerSource) }
    if (-not [string]::IsNullOrWhiteSpace($FallbackModel)) { $Args += @("--fallback-model", $FallbackModel) }
    if ($Limit -gt 0) { $Args += @("--limit", "$Limit") }
    if ($RuntimeBackend -eq "llama_cpp") {
        if ([string]::IsNullOrWhiteSpace($LlamaCli)) { throw "llama_cpp benchmark requires -LlamaCli" }
        $Args += @("--llama-cli", $LlamaCli)
    }
    if ($ReuseExisting) { $Args += "--resume" }
    Invoke-PythonStep $Name $Args
}

Invoke-ProductionBenchmark "[3] Benchmarking the dense Phase 10 reference in the complete production RAG path..." $StudentModel "transformers" "" $DenseEval

if (-not ($ReuseExisting -and (Test-Path $QuantizationManifest))) {
    $QuantArgs = @(
        ".\scripts\phase_11_production_rag\quantize_student.py",
        "--model", $StudentModel,
        "--output", $DeploymentArtifact,
        "--backend", $QuantizationBackend,
        "--gguf-type", $GgufType
    )
    if ($QuantizationBackend -eq "llamacpp_gguf") {
        if ([string]::IsNullOrWhiteSpace($LlamaCppConvertScript) -or [string]::IsNullOrWhiteSpace($LlamaCppQuantizeExe)) {
            throw "llamacpp_gguf requires -LlamaCppConvertScript and -LlamaCppQuantizeExe"
        }
        $QuantArgs += @(
            "--llama-cpp-convert-script", $LlamaCppConvertScript,
            "--llama-cpp-quantize-exe", $LlamaCppQuantizeExe
        )
    }
    if (Test-Path $DeploymentArtifact) { $QuantArgs += "--overwrite" }
    Invoke-PythonStep "[4] Creating the measured deployment artifact..." $QuantArgs
}

$QuantManifestJson = Get-Content $QuantizationManifest -Raw | ConvertFrom-Json
$DeploymentModel = [string]$QuantManifestJson.model_path
$RuntimeBackend = if ($QuantizationBackend -eq "llamacpp_gguf") { "llama_cpp" } else { "transformers" }
$TokenizerSource = if ($RuntimeBackend -eq "llama_cpp") { $StudentModel } else { "" }
Invoke-ProductionBenchmark "[5] Benchmarking the deployment artifact in the complete production RAG path..." $DeploymentModel $RuntimeBackend $TokenizerSource $DeploymentEval

$GateArgs = @(
    ".\scripts\phase_11_production_rag\check_phase_11_gate.py",
    "--reference-summary", (Join-Path $DenseEval "summary.json"),
    "--candidate", "dense_reference=$(Join-Path $DenseEval 'summary.json')",
    "--candidate", "deployment_${QuantizationBackend}=$(Join-Path $DeploymentEval 'summary.json')",
    "--snapshot-manifest", $SnapshotManifest,
    "--answerability-calibration", $AnswerabilityCalibration
)
if ($UnsupportedConfidentAnswerCap -ge 0.0) {
    $GateArgs += @("--unsupported-confident-answer-cap", "$UnsupportedConfidentAnswerCap")
}
Invoke-PythonStep "[Final] Building the Phase 11 production registry and human release gate..." $GateArgs

Write-Host ""
Write-Host "Phase 11 completed."
Write-Host "Registry: artifacts\phase_11_production_rag\release_candidate_registry.json"
Write-Host "Report: reports\PHASE_11_RELEASE_CANDIDATE_REPORT.md"
Write-Host "Gate: reports\PHASE_11_STOP_GATE.md"
Write-Host "After human PASS, rerun with -Mode Package -PackageCandidate <exact registry name>."

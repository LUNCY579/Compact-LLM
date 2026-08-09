<#
.SYNOPSIS
Run Phase 10 student architecture reduction and compact-model challengers.

.DESCRIPTION
Phase 10 starts only after Phase 9 has an explicit human PASS and an exact
selected-student manifest. It profiles the accepted student on the reviewed RAG
data, creates physically smaller dense candidates, evaluates unrecovered models,
recovery-trains a bounded promising subset under the original teacher, optionally
trains a pretrained compact challenger, reevaluates every candidate, and writes a
relative-opportunity registry plus pending human gate.
#>
param(
    [ValidateSet("Plan", "Pilot", "Full", "Evaluate")][string]$Mode = "Pilot",
    [string]$RagReasoningCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$MemoryReconstructionCorpusRoot = ".\data\phase_9b_memory_reconstruction\reconstruction_corpus_v2_large",
    [string]$StudentModel = "",
    [string]$Phase9ReferenceManifest = ".\artifacts\references\phase_9_student_reference.json",
    [string]$Phase9BReferenceManifest = ".\artifacts\references\phase_9b_student_reference.json",
    [string]$TeacherModel = "",
    [string]$Config = ".\configs\phase_10_student_architecture.json",
    [string]$StructureProfile = "",
    [string]$LayerSensitivityScan = "",
    [string]$DatasetRoot = ".\artifacts\phase_9_grounded_recovery_v1_7_5\dataset",
    [string]$KnowledgeDb = ".\artifacts\phase_9_grounded_recovery_v1_7_5\knowledge.sqlite3",
    [string]$GateCalibration = ".\artifacts\phase_9_grounded_recovery_v1_7_5\evidence_gate_calibration.json",
    [string]$OutputRoot = ".\artifacts\phase_10_student_architecture",
    [string]$Registry = "",
    [string]$CandidateReport = ".\reports\PHASE_10_CANDIDATE_REPORT.md",
    [string]$GateReport = ".\reports\PHASE_10_STOP_GATE.md",
    [int]$MaxSteps = 0,
    [int]$RecoveryTopK = 0,
    [int]$Seed = 42,
    [switch]$UseLora,
    [switch]$Teacher4Bit,
    [switch]$SkipRecovery,
    [string]$ChallengerSource = "",
    [string]$ChallengerName = "pretrained_compact_challenger",
    [switch]$ReuseExisting,
    [switch]$AllowUnapprovedPhase9
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

if ((Test-Path $Phase9BReferenceManifest) -and (Test-Path (Join-Path $MemoryReconstructionCorpusRoot "manifests\dataset_manifest.json"))) {
    Write-Host ""
    Write-Host "[Corpus] Validating the Phase 9B reconstruction corpus retained by this student..."
    & python .\scripts\phase_9b_memory_reconstruction\validate_memory_reconstruction_corpus.py --corpus-root $MemoryReconstructionCorpusRoot
    if ($LASTEXITCODE -ne 0) { throw "Integrated memory-reconstruction corpus validation failed." }
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

function Get-PropertyOrDefault {
    param($Object, [string]$Name, $Default)
    $Property = $Object.PSObject.Properties[$Name]
    if ($null -eq $Property -or $null -eq $Property.Value) { return $Default }
    return $Property.Value
}

$UsePhase9BReference = [string]::IsNullOrWhiteSpace($StudentModel) -and (Test-Path $Phase9BReferenceManifest)
if (-not $AllowUnapprovedPhase9) {
    if ($UsePhase9BReference) {
        Invoke-PythonStep "[Gate] Requiring the accepted Phase 9B reconstruction student..." @(
            ".\scripts\common\check_human_gate.py", "--phase", "9B"
        )
    } else {
        Invoke-PythonStep "[Gate] Requiring the accepted Phase 9 student..." @(
            ".\scripts\common\check_human_gate.py", "--phase", "9"
        )
    }
}
if (-not (Test-Path $Config)) { throw "Missing Phase 10 config: $Config" }
$Phase10Config = Get-Content $Config -Raw | ConvertFrom-Json
$CandidateMode = if ($Mode -eq "Full") { "full" } else { "pilot" }
$CandidateRows = $Phase10Config.candidate_sets.$CandidateMode
if (-not $CandidateRows) { throw "No candidate set found for mode $CandidateMode in $Config" }

if ([string]::IsNullOrWhiteSpace($StudentModel)) {
    $SelectedStudentManifest = if ($UsePhase9BReference) { $Phase9BReferenceManifest } else { $Phase9ReferenceManifest }
    $StudentModel = python .\scripts\phase_10_student_architecture\resolve_phase_9_student.py --manifest $SelectedStudentManifest
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($StudentModel)) {
        throw "Could not resolve the selected Phase 9/9B student. Finalize the prior phase or pass -StudentModel."
    }
}
if (-not (Test-Path $StudentModel)) { throw "Missing selected Phase 9/9B student checkpoint: $StudentModel" }
if ([string]::IsNullOrWhiteSpace($TeacherModel)) {
    $TeacherModel = python .\scripts\common\resolve_reference_model.py
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($TeacherModel)) {
        throw "Could not resolve the authoritative teacher. Supply -TeacherModel explicitly."
    }
}
if (-not (Test-Path $TeacherModel)) { throw "Phase 10 requires a local authoritative teacher checkpoint: $TeacherModel" }

$TrainFile = Join-Path $DatasetRoot "train.jsonl"
$DevFile = Join-Path $DatasetRoot "dev.jsonl"
$TestFile = Join-Path $DatasetRoot "test.jsonl"
foreach ($Path in @($TrainFile, $DevFile, $TestFile, $KnowledgeDb, $GateCalibration)) {
    if (-not (Test-Path $Path)) { throw "Missing Phase 10 input: $Path" }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if ([string]::IsNullOrWhiteSpace($Registry)) { $Registry = Join-Path $OutputRoot "candidate_registry.json" }

$ReferenceDevEval = Join-Path $OutputRoot "evaluation_reference_dev"
$ReferenceTestEval = Join-Path $OutputRoot "evaluation_reference_test_final"
$FinalDevSelection = Join-Path $OutputRoot "final_dev_selection.json"
$Profile = if ([string]::IsNullOrWhiteSpace($StructureProfile)) { Join-Path $OutputRoot "student_structure_profile.pt" } else { $StructureProfile }
$ProfileSummary = if ([string]::IsNullOrWhiteSpace($StructureProfile)) { Join-Path $OutputRoot "student_structure_profile.json" } else { [System.IO.Path]::ChangeExtension($StructureProfile, ".json") }
if (-not [string]::IsNullOrWhiteSpace($StructureProfile) -and -not (Test-Path $StructureProfile)) { throw "Missing external student structure profile: $StructureProfile" }
if (-not [string]::IsNullOrWhiteSpace($LayerSensitivityScan) -and -not (Test-Path $LayerSensitivityScan)) { throw "Missing student layer-sensitivity scan: $LayerSensitivityScan" }
$RecoverySelection = Join-Path $OutputRoot "recovery_selection.json"

function Invoke-RagEvaluation {
    param([string]$Name, [string]$Model, [string]$Adapter, [string]$Benchmark, [string]$Destination)
    if ($ReuseExisting -and (Test-Path (Join-Path $Destination "summary.json"))) { return }
    $Args = @(
        ".\scripts\phase_9_student_distillation\evaluate_rag.py",
        "--model", $Model,
        "--benchmark", $Benchmark,
        "--output-dir", $Destination,
        "--knowledge-db", $KnowledgeDb,
        "--gate-calibration", $GateCalibration,
        "--conditions", "closed_book,oracle_rag,real_rag,adversarial_rag",
        "--response-control", "one_sentence",
        "--max-new-tokens", "32"
    )
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $Args += @("--adapter", $Adapter) }
    if ($ReuseExisting) { $Args += "--resume" }
    Invoke-PythonStep $Name $Args
}

Invoke-RagEvaluation "[1] Evaluating the selected Phase 9/9B reference student on DEV..." $StudentModel "" $DevFile $ReferenceDevEval

if ($Mode -ne "Evaluate") {
    if ([string]::IsNullOrWhiteSpace($StructureProfile) -and -not ($ReuseExisting -and (Test-Path $Profile))) {
        $MaxProfileRecords = if ($Mode -eq "Full") {
            [int]$Phase10Config.profiling.max_records_full
        } else {
            [int]$Phase10Config.profiling.max_records_pilot
        }
        Invoke-PythonStep "[2] Profiling residual, MLP, and attention usage..." @(
            ".\scripts\phase_10_student_architecture\profile_student_structure.py",
            "--model", $StudentModel,
            "--data", $TrainFile,
            "--output", $Profile,
            "--summary", $ProfileSummary,
            "--max-records", "$MaxProfileRecords",
            "--max-sequence-length", "$($Phase10Config.profiling.max_sequence_length)"
        )
    }

    foreach ($Candidate in $CandidateRows) {
        $Name = [string]$Candidate.name
        $RawPath = Join-Path $OutputRoot "candidates\$Name\raw"
        if (-not ($ReuseExisting -and (Test-Path (Join-Path $RawPath "phase_10_candidate_manifest.json")))) {
            $DepthDrop = [int](Get-PropertyOrDefault $Candidate "depth_drop" 0)
            $RequestedDepthMode = [string](Get-PropertyOrDefault $Candidate "depth_mode" "uniform-drop")
            $CreateDepthMode = $RequestedDepthMode
            $ExplicitDropIds = @()
            if ($RequestedDepthMode -eq "importance-drop") {
                if ([string]::IsNullOrWhiteSpace($LayerSensitivityScan)) {
                    throw "Candidate $Name requests importance-drop but -LayerSensitivityScan was not supplied."
                }
                $Sensitivity = Get-Content $LayerSensitivityScan -Raw | ConvertFrom-Json
                $Order = @($Sensitivity.recommended_drop_order)
                if ($DepthDrop -le 0 -or $Order.Count -lt $DepthDrop) {
                    throw "Layer-sensitivity scan does not provide enough ranked layers for $Name (depth drop $DepthDrop)."
                }
                $ExplicitDropIds = @($Order | Select-Object -First $DepthDrop)
                $CreateDepthMode = "explicit-drop"
            }
            $Args = @(
                ".\scripts\phase_10_student_architecture\create_structural_candidate.py",
                "--model", $StudentModel,
                "--profile", $Profile,
                "--output", $RawPath,
                "--name", $Name,
                "--description", ([string]$Candidate.description),
                "--target-role", ([string](Get-PropertyOrDefault $Candidate "target_role" "experimental_candidate")),
                "--depth-drop", "$DepthDrop",
                "--depth-mode", $CreateDepthMode,
                "--hidden-prune-ratio", "$([double](Get-PropertyOrDefault $Candidate "hidden_prune_ratio" 0.0))",
                "--mlp-prune-ratio", "$([double](Get-PropertyOrDefault $Candidate "mlp_prune_ratio" 0.0))",
                "--attention-group-prune-ratio", "$([double](Get-PropertyOrDefault $Candidate "attention_group_prune_ratio" 0.0))"
            )
            if ($ExplicitDropIds.Count -gt 0) {
                $Args += "--drop-layer-ids"
                foreach ($LayerId in $ExplicitDropIds) { $Args += "$LayerId" }
            }
            $TargetMin = Get-PropertyOrDefault $Candidate "target_parameter_min" $null
            $TargetMax = Get-PropertyOrDefault $Candidate "target_parameter_max" $null
            if ($null -ne $TargetMin) { $Args += @("--target-parameter-min", "$([int64]$TargetMin)") }
            if ($null -ne $TargetMax) { $Args += @("--target-parameter-max", "$([int64]$TargetMax)") }
            if (Test-Path $RawPath) { $Args += "--overwrite" }
            Invoke-PythonStep "[3] Creating structural candidate $Name..." $Args
        }
    }
}

$RawCandidateSpecs = New-Object System.Collections.Generic.List[string]
$FinalCandidateSpecs = New-Object System.Collections.Generic.List[string]
foreach ($Candidate in $CandidateRows) {
    $Name = [string]$Candidate.name
    $RawPath = Join-Path $OutputRoot "candidates\$Name\raw"
    if (-not (Test-Path $RawPath)) { throw "Missing raw Phase 10 candidate: $RawPath" }
    $RawEval = Join-Path $OutputRoot "candidates\$Name\evaluation_raw_dev"
    Invoke-RagEvaluation "[4] Evaluating raw candidate $Name on DEV..." $RawPath "" $DevFile $RawEval
    $RawSummary = Join-Path $RawEval "summary.json"
    $RawCandidateSpecs.Add("$Name=$RawSummary") | Out-Null
    $FinalCandidateSpecs.Add("raw_${Name}=$RawSummary") | Out-Null
}

if ($Mode -eq "Evaluate") {
    foreach ($Candidate in $CandidateRows) {
        $Name = [string]$Candidate.name
        $RecoveredSummary = Join-Path $OutputRoot "candidates\$Name\evaluation_recovered_dev\summary.json"
        if (Test-Path $RecoveredSummary) {
            $FinalCandidateSpecs.Add("recovered_${Name}=$RecoveredSummary") | Out-Null
        }
    }
    $ChallengerRoot = Join-Path $OutputRoot "challengers"
    if (Test-Path $ChallengerRoot) {
        foreach ($Directory in Get-ChildItem $ChallengerRoot -Directory) {
            $ChallengerSummary = Join-Path $Directory.FullName "evaluation_dev\summary.json"
            if (Test-Path $ChallengerSummary) {
                $FinalCandidateSpecs.Add("$($Directory.Name)=$ChallengerSummary") | Out-Null
            }
        }
    }
}

if ($Mode -eq "Plan") {
    Write-Host ""
    Write-Host "Phase 10 planning/screening completed. Raw candidates were built and evaluated; recovery and final gate were not run."
    exit 0
}

if (-not $SkipRecovery -and $Mode -ne "Evaluate") {
    if ($RecoveryTopK -le 0) {
        $RecoveryTopK = if ($Mode -eq "Full") {
            [int]$Phase10Config.recovery.full_top_k
        } else {
            [int]$Phase10Config.recovery.pilot_top_k
        }
    }
    $SelectArgs = @(
        ".\scripts\phase_10_student_architecture\select_recovery_candidates.py",
        "--reference-summary", (Join-Path $ReferenceDevEval "summary.json"),
        "--top-k", "$RecoveryTopK",
        "--out", $RecoverySelection
    )
    foreach ($Spec in $RawCandidateSpecs) { $SelectArgs += @("--candidate", $Spec) }
    Invoke-PythonStep "[5] Selecting a bounded recovery set..." $SelectArgs

    $Selection = Get-Content $RecoverySelection -Raw | ConvertFrom-Json
    if ($MaxSteps -le 0) {
        $MaxSteps = if ($Mode -eq "Full") {
            [int]$Phase10Config.recovery.max_steps_full
        } else {
            [int]$Phase10Config.recovery.max_steps_pilot
        }
    }
    foreach ($Name in $Selection.selected_candidates) {
        $RawPath = Join-Path $OutputRoot "candidates\$Name\raw"
        $RecoveryRoot = Join-Path $OutputRoot "candidates\$Name\recovery"
        if (-not ($ReuseExisting -and (Test-Path (Join-Path $RecoveryRoot "training_manifest.json")))) {
            $TrainArgs = @(
                ".\scripts\phase_9_student_distillation\train_student.py",
                "--student", $RawPath,
                "--teacher", $TeacherModel,
                "--objective", "kd",
                "--train-file", $TrainFile,
                "--dev-file", $DevFile,
                "--output-dir", $RecoveryRoot,
                "--max-steps", "$MaxSteps",
                "--seed", "$Seed"
            )
            if ($UseLora) { $TrainArgs += "--lora" }
            if ($Teacher4Bit) { $TrainArgs += "--teacher-4bit" }
            Invoke-PythonStep "[6] Recovery-distilling $Name under the original teacher..." $TrainArgs
        }
        $RecoveredModel = if ($UseLora) { $RawPath } else { Join-Path $RecoveryRoot "model" }
        $RecoveredAdapter = if ($UseLora) { Join-Path $RecoveryRoot "adapter" } else { "" }
        $RecoveredEval = Join-Path $OutputRoot "candidates\$Name\evaluation_recovered_dev"
        Invoke-RagEvaluation "[7] Evaluating recovered candidate $Name on DEV..." $RecoveredModel $RecoveredAdapter $DevFile $RecoveredEval
        $FinalCandidateSpecs.Add("recovered_${Name}=$(Join-Path $RecoveredEval 'summary.json')") | Out-Null
    }
}

if (-not [string]::IsNullOrWhiteSpace($ChallengerSource) -and $Mode -ne "Evaluate") {
    $ChallengerInit = Join-Path $OutputRoot "challengers\$ChallengerName\initialization"
    if (-not ($ReuseExisting -and (Test-Path (Join-Path $ChallengerInit "student_creation_manifest.json")))) {
        $InitArgs = @(
            ".\scripts\phase_9_student_distillation\create_student.py",
            "--source", $ChallengerSource,
            "--output", $ChallengerInit,
            "--mode", "pretrained_clone"
        )
        if (Test-Path $ChallengerInit) { $InitArgs += "--overwrite" }
        Invoke-PythonStep "[8] Cloning pretrained compact challenger..." $InitArgs
    }
    $ChallengerTrain = Join-Path $OutputRoot "challengers\$ChallengerName\sft"
    if (-not ($ReuseExisting -and (Test-Path (Join-Path $ChallengerTrain "training_manifest.json")))) {
        $ChallengerSteps = if ($MaxSteps -gt 0) { $MaxSteps } else { 200 }
        $ChallengerArgs = @(
            ".\scripts\phase_9_student_distillation\train_student.py",
            "--student", $ChallengerInit,
            "--objective", "sft",
            "--train-file", $TrainFile,
            "--dev-file", $DevFile,
            "--output-dir", $ChallengerTrain,
            "--max-steps", "$ChallengerSteps",
            "--seed", "$Seed"
        )
        if ($UseLora) { $ChallengerArgs += "--lora" }
        Invoke-PythonStep "[9] Sequence-level RAG SFT for compact challenger..." $ChallengerArgs
    }
    $ChallengerModel = if ($UseLora) { $ChallengerInit } else { Join-Path $ChallengerTrain "model" }
    $ChallengerAdapter = if ($UseLora) { Join-Path $ChallengerTrain "adapter" } else { "" }
    $ChallengerEval = Join-Path $OutputRoot "challengers\$ChallengerName\evaluation_dev"
    Invoke-RagEvaluation "[10] Evaluating compact challenger on DEV..." $ChallengerModel $ChallengerAdapter $DevFile $ChallengerEval
    $FinalCandidateSpecs.Add("$ChallengerName=$(Join-Path $ChallengerEval 'summary.json')") | Out-Null
}

# Freeze exactly one final candidate using DEV evidence before touching TEST.
$FinalSelectArgs = @(
    ".\scripts\phase_10_student_architecture\select_recovery_candidates.py",
    "--reference-summary", (Join-Path $ReferenceDevEval "summary.json"),
    "--purpose", "final",
    "--top-k", "1",
    "--out", $FinalDevSelection
)
foreach ($Spec in $FinalCandidateSpecs) { $FinalSelectArgs += @("--candidate", $Spec) }
Invoke-PythonStep "[11] Freezing the final Phase 10 candidate on DEV..." $FinalSelectArgs
$FinalSelection = Get-Content $FinalDevSelection -Raw | ConvertFrom-Json
$FinalName = [string]@($FinalSelection.selected_candidates)[0]
if ([string]::IsNullOrWhiteSpace($FinalName)) { throw "No Phase 10 candidate could be frozen on dev." }
$FinalPacket = @($FinalSelection.ranking | Where-Object { [string]$_.name -eq $FinalName })[0]
$FinalModel = [string]$FinalPacket.model
$FinalAdapter = [string]$FinalPacket.adapter

# SEALED TEST: reference + already-frozen candidate only. No candidate can be
# swapped after seeing these results.
$FinalTestEval = Join-Path $OutputRoot "evaluation_final_selected_test"
Invoke-RagEvaluation "[12] Sealed TEST reference evaluation..." $StudentModel "" $TestFile $ReferenceTestEval
Invoke-RagEvaluation "[13] Sealed TEST evaluation for $FinalName..." $FinalModel $FinalAdapter $TestFile $FinalTestEval

$GateArgs = @(
    ".\scripts\phase_10_student_architecture\check_phase_10_gate.py",
    "--reference-summary", (Join-Path $ReferenceTestEval "summary.json"),
    "--candidate", "$FinalName=$(Join-Path $FinalTestEval 'summary.json')",
    "--registry", $Registry,
    "--report", $CandidateReport,
    "--gate", $GateReport
)
Invoke-PythonStep "[Final] Building the Phase 10 registry and human stop gate..." $GateArgs

Write-Host ""
Write-Host "Phase 10 completed."
Write-Host "Registry: $Registry"
Write-Host "Report: $CandidateReport"
Write-Host "Gate: $GateReport"
Write-Host "After human PASS, run finalize_phase_10_selection.py with the exact selected candidate."

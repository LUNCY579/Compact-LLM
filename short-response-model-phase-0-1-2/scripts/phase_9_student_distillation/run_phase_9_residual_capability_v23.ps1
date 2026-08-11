param(
    [string]$V22RunRoot = ".\artifacts\phase_9_capability_composition_v22_lr1e4_s1200",
    [string]$GreedyAnchorModel = "",
    [string]$OutputRoot = ".\artifacts\phase_9_residual_capability_v23_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string]$OriginalCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$V211CorpusRoot = ".\data\phase_9_student_distillation\procedural_recovery_v211",
    [string]$V212CorpusRoot = ".\data\phase_9_student_distillation\general_reasoning_v212",
    [string]$V22CorpusRoot = ".\data\phase_9_student_distillation\multidomain_reasoning_v22",
    [string]$V23DirectRoot = ".\data\phase_9_student_distillation\residual_capability_v23\direct_reasoning",
    [string]$V23RagRoot = ".\data\phase_9_student_distillation\residual_capability_v23\procedural_rag",
    [int]$Seed = 23031,
    [int]$MaximumEvidenceTokens = 384,
    [ValidateSet("auto","cpu")][string]$AnchorDevice = "auto",
    [switch]$RegenerateDataset,
    [switch]$SkipTraining,
    [switch]$SkipDoseSweep,
    [switch]$SkipDiagnostics,
    [switch]$RunFreshSealedTest,
    [switch]$RequireCuda,
    [switch]$ReuseExisting,
    [switch]$KeepDoseModels,
    [switch]$AllowAggressiveTraining
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root
$ConfigPath = ".\configs\phase_9_residual_capability_v23.json"
$Config = Get-Content -Raw $ConfigPath | ConvertFrom-Json

function Invoke-PythonStep {
    param([string]$Name,[string[]]$Arguments)
    Write-Host ""; Write-Host $Name
    & python @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}
function Ensure-Path { param([string]$Path,[string]$Description) if (-not (Test-Path $Path)) { throw "Missing ${Description}: $Path" } }
function Should-Run { param([string]$Artifact) if (Test-Path $Artifact) { if ($ReuseExisting) { return $false } throw "Artifact already exists: $Artifact. Use -ReuseExisting deliberately." } return $true }
function Maybe-Cuda { param([string[]]$Args) if ($RequireCuda) { return @($Args + "--require-cuda") } return $Args }

Ensure-Path $V22RunRoot "v2.2 artifact root"
$OriginalAnchor = Join-Path $V22RunRoot "frozen_anchor"
$DerivedGreedy = Join-Path $V22RunRoot "track_d_greedy_soup\final_model"
if ([string]::IsNullOrWhiteSpace($GreedyAnchorModel)) { $GreedyAnchorModel = $DerivedGreedy }
$TrackAAdapter = Join-Path $V22RunRoot "track_a_broad_reasoning\adapter"
$TrackBAdapter = Join-Path $V22RunRoot "track_b_anchored_reasoning\adapter"
$ProcedureTeacherAdapter = Join-Path $V22RunRoot "track_c_procedure\adapter"
$AbstentionAdapter = Join-Path $V22RunRoot "track_c_abstention\adapter"
foreach ($p in @($OriginalAnchor,$GreedyAnchorModel,$TrackAAdapter,$TrackBAdapter,$ProcedureTeacherAdapter,$AbstentionAdapter)) { Ensure-Path $p "required v2.2 anchor/teacher artifact" }

$V22Selection = Join-Path $V22RunRoot "capability_composition_v22_selection.json"
if (Test-Path $V22Selection) {
    $sel = Get-Content -Raw $V22Selection | ConvertFrom-Json
    if ([string]$sel.selected_candidate -ne "greedy_soup") { throw "v2.3 requires the frozen v2.2 greedy_soup lineage; v2.2 selection reports $($sel.selected_candidate)." }
}

Ensure-Path $OriginalCorpusRoot "original Phase 9 corpus"; Ensure-Path $V211CorpusRoot "v2.1.1 corpus"; Ensure-Path $V212CorpusRoot "v2.1.2 corpus"; Ensure-Path $V22CorpusRoot "v2.2 corpus"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

# Source corpus paths. Old TEST splits are deliberately never assigned here:
# v2.3 must not tune against the already-consumed v2.2 sealed evaluation.
$OriginalTrain = Join-Path $OriginalCorpusRoot "reviewed_splits\train.reviewed.jsonl"
$OriginalDev = Join-Path $OriginalCorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$OriginalKnowledge = Join-Path $OriginalCorpusRoot "knowledge.jsonl"; $OriginalProcedures = Join-Path $OriginalCorpusRoot "procedures.jsonl"
$V211Dev = Join-Path $V211CorpusRoot "reviewed_splits\dev.reviewed.jsonl"; $V211Knowledge = Join-Path $V211CorpusRoot "knowledge.jsonl"; $V211Procedures = Join-Path $V211CorpusRoot "procedures.jsonl"
$V212Dev = Join-Path $V212CorpusRoot "reviewed_splits\dev.reviewed.jsonl"; $V22Dev = Join-Path $V22CorpusRoot "reviewed_splits\dev.reviewed.jsonl"

# ---------------------------------------------------------------------------
# A. Fresh structurally-held-out v2.3 corpora
# ---------------------------------------------------------------------------
if ($RegenerateDataset -or -not (Test-Path (Join-Path $V23DirectRoot "manifests\dataset_manifest.json"))) {
    Invoke-PythonStep "[v2.3 Data] Generate structural-OOD direct reasoning corpus" @(".\scripts\phase_9_student_distillation\generate_residual_discovery_v23.py","--output-root",$V23DirectRoot,"--seed","$Seed","--force")
}
if ($RegenerateDataset -or -not (Test-Path (Join-Path $V23RagRoot "manifests\dataset_manifest.json"))) {
    Invoke-PythonStep "[v2.3 Data] Generate fresh procedural Real-RAG corpus" @(".\scripts\phase_9_student_distillation\generate_residual_procedural_rag_v23.py","--output-root",$V23RagRoot,"--seed","$($Seed+10)","--force")
}
Invoke-PythonStep "[v2.3 Data] Validate both corpora" @(".\scripts\phase_9_student_distillation\validate_residual_capability_v23.py","--direct-root",$V23DirectRoot,"--rag-root",$V23RagRoot,"--require-direct-rows","24000","--require-rag-rows","7000")

$DirectDiscovery = Join-Path $V23DirectRoot "splits\discovery.reviewed.jsonl"; $DirectDev = Join-Path $V23DirectRoot "splits\dev.reviewed.jsonl"; $DirectOod = Join-Path $V23DirectRoot "splits\ood_dev.reviewed.jsonl"; $DirectSealed = Join-Path $V23DirectRoot "splits\fresh_sealed.reviewed.jsonl"
$DirectDevProbe = Join-Path $V23DirectRoot "probes\dev_probe.reviewed.jsonl"; $DirectOodProbe = Join-Path $V23DirectRoot "probes\ood_probe.reviewed.jsonl"
$V23RagDiscovery = Join-Path $V23RagRoot "splits\discovery.reviewed.jsonl"; $V23RagDev = Join-Path $V23RagRoot "splits\dev.reviewed.jsonl"; $V23RagOod = Join-Path $V23RagRoot "splits\ood_dev.reviewed.jsonl"; $V23RagSealed = Join-Path $V23RagRoot "splits\fresh_sealed.reviewed.jsonl"
$V23RagCalibrationBenchmark = Join-Path $V23RagRoot "calibration\discovery_calibration.reviewed.jsonl"

# ---------------------------------------------------------------------------
# B. Retrieval infrastructure. New v2.3 RAG calibration is frozen from the
# discovery split, not DEV/OOD-DEV, so DEV does not influence residual mining.
# ---------------------------------------------------------------------------
$OriginalDbCandidate = Join-Path $V22RunRoot "original_phase9.sqlite3"; $V211DbCandidate = Join-Path $V22RunRoot "procedural_v211.sqlite3"
$OriginalCalCandidate = Join-Path $V22RunRoot "original_calibration_query_aware_v21.json"; $V211CalCandidate = Join-Path $V22RunRoot "v211_calibration_query_aware_v21.json"
$OriginalDb = if (Test-Path $OriginalDbCandidate) { $OriginalDbCandidate } else { Join-Path $OutputRoot "original_phase9.sqlite3" }
$V211Db = if (Test-Path $V211DbCandidate) { $V211DbCandidate } else { Join-Path $OutputRoot "procedural_v211.sqlite3" }
$OriginalCalibration = if (Test-Path $OriginalCalCandidate) { $OriginalCalCandidate } else { Join-Path $OutputRoot "original_calibration_query_aware_v21.json" }
$V211Calibration = if (Test-Path $V211CalCandidate) { $V211CalCandidate } else { Join-Path $OutputRoot "v211_calibration_query_aware_v21.json" }
if (-not (Test-Path $OriginalDb)) { Invoke-PythonStep "[Retrieval] Build original Phase 9 index" @(".\scripts\phase_9_student_distillation\build_knowledge_index.py","--knowledge",$OriginalKnowledge,"--procedures",$OriginalProcedures,"--output",$OriginalDb) }
if (-not (Test-Path $V211Db)) { Invoke-PythonStep "[Retrieval] Build v2.1.1 index" @(".\scripts\phase_9_student_distillation\build_knowledge_index.py","--knowledge",$V211Knowledge,"--procedures",$V211Procedures,"--output",$V211Db) }
if (-not (Test-Path $OriginalCalibration)) { Invoke-PythonStep "[Retrieval] Freeze original DEV calibration" @(".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py","--benchmark",$OriginalDev,"--knowledge-db",$OriginalDb,"--output",$OriginalCalibration,"--retrieval-selection-policy","query_aware_v21","--require-research-ready") }
if (-not (Test-Path $V211Calibration)) { Invoke-PythonStep "[Retrieval] Freeze historical procedure DEV calibration" @(".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py","--benchmark",$V211Dev,"--knowledge-db",$V211Db,"--output",$V211Calibration,"--retrieval-selection-policy","query_aware_v21","--require-research-ready") }
$V23RagDb = Join-Path $OutputRoot "v23_procedural.sqlite3"; $V23RagCalibration = Join-Path $OutputRoot "v23_procedural_discovery_calibration.json"
if (-not (Test-Path $V23RagDb)) { Invoke-PythonStep "[Retrieval] Build fresh v2.3 procedural index" @(".\scripts\phase_9_student_distillation\build_knowledge_index.py","--knowledge",(Join-Path $V23RagRoot "knowledge.jsonl"),"--procedures",(Join-Path $V23RagRoot "procedures.jsonl"),"--output",$V23RagDb) }
if (-not (Test-Path $V23RagCalibration)) { Invoke-PythonStep "[Retrieval] Freeze v2.3 discovery calibration" @(".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py","--benchmark",$V23RagCalibrationBenchmark,"--knowledge-db",$V23RagDb,"--output",$V23RagCalibration,"--retrieval-selection-policy","query_aware_v23","--require-research-ready") }

# ---------------------------------------------------------------------------
# C. Oracle capability audit / residual mining.
# ---------------------------------------------------------------------------
$MiningRoot = Join-Path $OutputRoot "residual_mining"; New-Item -ItemType Directory -Force -Path $MiningRoot | Out-Null
$PreservationBenchmark = Join-Path $MiningRoot "preservation_benchmark.train_only.jsonl"
if (-not (Test-Path $PreservationBenchmark)) { Invoke-PythonStep "[Mining] Build TRAIN-only preservation benchmark" @(".\scripts\phase_9_student_distillation\build_preservation_mining_subset_v23.py","--train-file",$OriginalTrain,"--output",$PreservationBenchmark,"--limit","$($Config.data.preservation_mining_rows)","--seed","$($Seed+20)") }

function Eval-DirectDiscovery {
    param([string]$Label,[string]$Model,[string]$Adapter,[string]$Dest)
    $summary = Join-Path $Dest "summary.json"; if (-not (Should-Run $summary)) { return }
    $args=@(".\scripts\phase_9_student_distillation\evaluate_residual_reasoning_v23.py","--model",$Model,"--benchmark",$DirectDiscovery,"--output-dir",$Dest,"--minimum-cases","500","--resume")
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $args += @("--adapter",$Adapter) }; if ($RequireCuda) { $args += "--require-cuda" }; Invoke-PythonStep $Label $args
}
function Eval-V23RagDiscovery {
    param([string]$Label,[string]$Model,[string]$Adapter,[string]$Dest)
    $summary=Join-Path $Dest "summary.json"; if (-not (Should-Run $summary)) { return }
    $args=@(".\scripts\phase_9_student_distillation\evaluate_rag.py","--model",$Model,"--benchmark",$V23RagDiscovery,"--output-dir",$Dest,"--knowledge-db",$V23RagDb,"--gate-calibration",$V23RagCalibration,"--conditions","real_rag","--retrieval-selection-policy","query_aware_v23","--maximum-evidence-tokens","$MaximumEvidenceTokens","--max-new-tokens","32","--minimum-cases-per-condition","1","--minimum-supported-cases","1","--minimum-unsupported-cases","1","--smoke-test","--resume")
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $args += @("--adapter",$Adapter) }; if ($RequireCuda) { $args += "--require-cuda" }; Invoke-PythonStep $Label $args
}
function Eval-PreservationMining {
    param([string]$Dest)
    $summary=Join-Path $Dest "summary.json"; if (-not (Should-Run $summary)) { return }
    $args=@(".\scripts\phase_9_student_distillation\evaluate_rag.py","--model",$GreedyAnchorModel,"--benchmark",$PreservationBenchmark,"--output-dir",$Dest,"--knowledge-db",$OriginalDb,"--gate-calibration",$OriginalCalibration,"--conditions","real_rag","--retrieval-selection-policy","query_aware_v21","--maximum-evidence-tokens","$MaximumEvidenceTokens","--max-new-tokens","32","--minimum-cases-per-condition","1","--minimum-supported-cases","1","--minimum-unsupported-cases","1","--smoke-test","--resume")
    if ($RequireCuda) { $args += "--require-cuda" }; Invoke-PythonStep "[Mining] Verify greedy preservation cases" $args
}

$DGreedy=Join-Path $MiningRoot "direct_greedy"; $DA=Join-Path $MiningRoot "direct_track_a"; $DB=Join-Path $MiningRoot "direct_track_b"; $DP=Join-Path $MiningRoot "direct_procedure_teacher"
Eval-DirectDiscovery "[Mining] Direct discovery - greedy anchor" $GreedyAnchorModel "" $DGreedy
Eval-DirectDiscovery "[Mining] Direct discovery - Track A teacher" $OriginalAnchor $TrackAAdapter $DA
Eval-DirectDiscovery "[Mining] Direct discovery - Track B teacher" $OriginalAnchor $TrackBAdapter $DB
Eval-DirectDiscovery "[Mining] Direct discovery - procedure teacher" $OriginalAnchor $ProcedureTeacherAdapter $DP
$RGreedy=Join-Path $MiningRoot "rag_greedy"; $RB=Join-Path $MiningRoot "rag_track_b"; $RP=Join-Path $MiningRoot "rag_procedure_teacher"
Eval-V23RagDiscovery "[Mining] Procedural discovery - greedy anchor" $GreedyAnchorModel "" $RGreedy
Eval-V23RagDiscovery "[Mining] Procedural discovery - Track B teacher" $OriginalAnchor $TrackBAdapter $RB
Eval-V23RagDiscovery "[Mining] Procedural discovery - procedure teacher" $OriginalAnchor $ProcedureTeacherAdapter $RP
$PresEval=Join-Path $MiningRoot "preservation_greedy"; Eval-PreservationMining $PresEval

$Mined = Join-Path $MiningRoot "mined"
if (Should-Run (Join-Path $Mined "RESIDUAL_MINING_AUDIT.json")) {
    Invoke-PythonStep "[Mining] Gold-verified residual / oracle-union audit" @(
        ".\scripts\phase_9_student_distillation\mine_residual_capabilities_v23.py",
        "--direct-benchmark",$DirectDiscovery,"--direct-anchor-results",(Join-Path $DGreedy "results.jsonl"),"--direct-teacher-a-results",(Join-Path $DA "results.jsonl"),"--direct-teacher-b-results",(Join-Path $DB "results.jsonl"),"--direct-teacher-procedure-results",(Join-Path $DP "results.jsonl"),
        "--procedure-benchmark",$V23RagDiscovery,"--procedure-anchor-results",(Join-Path $RGreedy "results.jsonl"),"--procedure-teacher-b-results",(Join-Path $RB "results.jsonl"),"--procedure-teacher-procedure-results",(Join-Path $RP "results.jsonl"),
        "--preservation-benchmark",$PreservationBenchmark,"--preservation-anchor-results",(Join-Path $PresEval "results.jsonl"),"--output-root",$Mined,"--preservation-limit","2400","--seed","$($Seed+30)"
    )
}
$TrainingViews=Join-Path $MiningRoot "training_views"
if (Should-Run (Join-Path $TrainingViews "training_mix_manifest.json")) {
    Invoke-PythonStep "[Mining] Build preservation-heavy residual training views" @(
        ".\scripts\phase_9_student_distillation\build_residual_training_mix_v23.py","--reasoning-residual",(Join-Path $Mined "reasoning_residual.reviewed.jsonl"),"--procedure-residual",(Join-Path $Mined "procedure_residual.reviewed.jsonl"),"--preservation",(Join-Path $Mined "preservation_replay.reviewed.jsonl"),"--output-root",$TrainingViews,
        "--reasoning-total","$($Config.data.reasoning_training_rows)","--reasoning-residual-fraction","$($Config.data.reasoning_residual_fraction)","--procedure-total","$($Config.data.procedure_training_rows)","--procedure-residual-fraction","$($Config.data.procedure_residual_fraction)","--minimum-reasoning-residual","$($Config.data.minimum_reasoning_residual_rows)","--minimum-procedure-residual","$($Config.data.minimum_procedure_residual_rows)","--maximum-repeat-factor","$($Config.data.maximum_repeat_factor)","--seed","$($Seed+40)"
    )
}
$MixManifest=Get-Content -Raw (Join-Path $TrainingViews "training_mix_manifest.json") | ConvertFrom-Json
$ReasoningReady=[bool]$MixManifest.readiness.reasoning_ready; $ProcedureReady=[bool]$MixManifest.readiness.procedure_ready

# ---------------------------------------------------------------------------
# D/E. Independent preservation-constrained residual LoRAs.
# ---------------------------------------------------------------------------
function Train-Residual {
    param([string]$Label,[string]$TrainFile,[string]$Dest,[double]$LR,[int]$Steps,[int]$Rank,[double]$AnchorWeight,[double]$AnchorTemperature)
    $manifest=Join-Path $Dest "training_manifest.json"; if (-not (Should-Run $manifest)) { return }
    if (-not $AllowAggressiveTraining) {
        if ($LR -gt [double]$Config.safety.maximum_default_learning_rate) { throw "v2.3 guard: LR $LR exceeds configured maximum." }
        if ($Steps -gt [int]$Config.safety.maximum_steps_per_adapter) { throw "v2.3 guard: steps $Steps exceed configured maximum." }
    }
    $args=@(".\scripts\phase_9_student_distillation\train_student.py","--student",$GreedyAnchorModel,"--objective","sft","--train-file",$TrainFile,"--output-dir",$Dest,"--max-steps","$Steps","--learning-rate","$LR","--gradient-accumulation-steps","32","--max-sequence-length","768","--maximum-evidence-tokens","$MaximumEvidenceTokens","--max-completion-tokens","32","--first-answer-token-weight","1.5","--evidence-anchor-token-weight","1.5","--abstention-token-weight","1.5","--eos-loss-weight","1.3","--seed","$Seed","--pipeline-revision","phase9_residual_capability_v2_3","--lora","--lora-rank","$Rank","--anchor-model",$GreedyAnchorModel,"--anchor-kl-weight","$AnchorWeight","--anchor-temperature","$AnchorTemperature","--anchor-device",$AnchorDevice)
    if ($RequireCuda) { $args += "--require-cuda" }; Invoke-PythonStep $Label $args
}
$ReasoningTrain=Join-Path $TrainingViews "reasoning_residual_mixed.reviewed.jsonl"; $ProcedureTrain=Join-Path $TrainingViews "procedure_residual_mixed.reviewed.jsonl"
$ReasoningDir=Join-Path $OutputRoot "residual_reasoning_training"; $ProcedureDir=Join-Path $OutputRoot "residual_procedure_training"
if (-not $SkipTraining -and $ReasoningReady) { Train-Residual "[Train R] Anchored answerable-reasoning residual" $ReasoningTrain $ReasoningDir ([double]$Config.training.reasoning_residual.learning_rate) ([int]$Config.training.reasoning_residual.max_steps) ([int]$Config.training.reasoning_residual.lora_rank) ([double]$Config.training.reasoning_residual.anchor_kl_weight) ([double]$Config.training.reasoning_residual.anchor_temperature) }
if (-not $SkipTraining -and $ProcedureReady) { Train-Residual "[Train P] Anchored procedural residual" $ProcedureTrain $ProcedureDir ([double]$Config.training.procedure_residual.learning_rate) ([int]$Config.training.procedure_residual.max_steps) ([int]$Config.training.procedure_residual.lora_rank) ([double]$Config.training.procedure_residual.anchor_kl_weight) ([double]$Config.training.procedure_residual.anchor_temperature) }
$ReasoningAdapter=Join-Path $ReasoningDir "adapter"; $ProcedureAdapter=Join-Path $ProcedureDir "adapter"
if ($ReasoningReady -and -not $SkipTraining) { Ensure-Path $ReasoningAdapter "reasoning residual adapter" }
if ($ProcedureReady -and -not $SkipTraining) { Ensure-Path $ProcedureAdapter "procedure residual adapter" }

# ---------------------------------------------------------------------------
# F. Coarse + local-refined capability-dose calibration on DEV probes.
# ---------------------------------------------------------------------------
function Run-Dose {
    param([string]$Label,[string]$Adapter,[string]$Dest)
    $result=Join-Path $Dest "DOSE_SWEEP_RESULT.json"; if (-not (Should-Run $result)) { return }
    $args=@(".\scripts\phase_9_student_distillation\run_residual_dose_sweep_v23.py","--config",$ConfigPath,"--label",$Label,"--base-model",$GreedyAnchorModel,"--adapter",$Adapter,"--output-root",$Dest,"--original-benchmark",$OriginalDev,"--original-db",$OriginalDb,"--original-calibration",$OriginalCalibration,"--historical-procedural-benchmark",$V211Dev,"--historical-procedural-db",$V211Db,"--historical-procedural-calibration",$V211Calibration,"--v23-rag-benchmark",$V23RagDev,"--v23-rag-ood-benchmark",$V23RagOod,"--v23-rag-db",$V23RagDb,"--v23-rag-calibration",$V23RagCalibration,"--v23-direct-benchmark",$DirectDevProbe,"--v23-direct-ood-benchmark",$DirectOodProbe,"--maximum-evidence-tokens","$MaximumEvidenceTokens")
    if ($RequireCuda) { $args += "--require-cuda" }; if ($ReuseExisting) { $args += "--reuse-existing" }; if ($KeepDoseModels) { $args += "--retain-models" }; Invoke-PythonStep "[Dose] $Label residual" $args
}
$ReasoningDoseRoot=Join-Path $OutputRoot "dose_reasoning"; $ProcedureDoseRoot=Join-Path $OutputRoot "dose_procedure"
if (-not $SkipDoseSweep -and $ReasoningReady -and (Test-Path $ReasoningAdapter)) { Run-Dose "reasoning" $ReasoningAdapter $ReasoningDoseRoot }
if (-not $SkipDoseSweep -and $ProcedureReady -and (Test-Path $ProcedureAdapter)) { Run-Dose "procedure" $ProcedureAdapter $ProcedureDoseRoot }
$ReasoningWeight=0.0; $ProcedureWeight=0.0; $ReasoningDoseModel=$GreedyAnchorModel; $ProcedureDoseModel=$GreedyAnchorModel
if (Test-Path (Join-Path $ReasoningDoseRoot "DOSE_SWEEP_RESULT.json")) { $r=Get-Content -Raw (Join-Path $ReasoningDoseRoot "DOSE_SWEEP_RESULT.json") | ConvertFrom-Json; $ReasoningWeight=[double]$r.selected_weight; $ReasoningDoseModel=[string]$r.selected_model }
if (Test-Path (Join-Path $ProcedureDoseRoot "DOSE_SWEEP_RESULT.json")) { $r=Get-Content -Raw (Join-Path $ProcedureDoseRoot "DOSE_SWEEP_RESULT.json") | ConvertFrom-Json; $ProcedureWeight=[double]$r.selected_weight; $ProcedureDoseModel=[string]$r.selected_model }

# ---------------------------------------------------------------------------
# G/I. Full DEV/OOD evaluation of anchor, safe single residuals, and a small
# preregistered joint composition family. No old TEST data is referenced.
# ---------------------------------------------------------------------------
function Eval-FullCandidate {
    param([string]$Label,[string]$Model,[string]$Dest)
    $manifest=Join-Path $Dest "candidate_evaluation_manifest.json"; if (-not (Should-Run $manifest)) { return }
    $args=@(".\scripts\phase_9_student_distillation\evaluate_candidate_v23.py","--model",$Model,"--output-root",$Dest,"--original-benchmark",$OriginalDev,"--original-db",$OriginalDb,"--original-calibration",$OriginalCalibration,"--historical-procedural-benchmark",$V211Dev,"--historical-procedural-db",$V211Db,"--historical-procedural-calibration",$V211Calibration,"--v23-rag-benchmark",$V23RagDev,"--v23-rag-ood-benchmark",$V23RagOod,"--v23-rag-db",$V23RagDb,"--v23-rag-calibration",$V23RagCalibration,"--v23-direct-benchmark",$DirectDev,"--v23-direct-ood-benchmark",$DirectOod,"--v212-benchmark",$V212Dev,"--v22-benchmark",$V22Dev,"--maximum-evidence-tokens","$MaximumEvidenceTokens","--research-ready-rag","--resume")
    if ($RequireCuda) { $args += "--require-cuda" }; Invoke-PythonStep $Label $args
}
function Compose-Joint {
    param([string]$Name,[double]$RW,[double]$PW)
    if ($RW -le 0 -and $PW -le 0) { return $null }
    $dir=Join-Path $OutputRoot ("joint_"+$Name); New-Item -ItemType Directory -Force -Path $dir | Out-Null; $ingredients=@()
    if ($RW -gt 0) { $ingredients += [ordered]@{name="reasoning_residual";adapter=$ReasoningAdapter;weight=$RW} }
    if ($PW -gt 0) { $ingredients += [ordered]@{name="procedure_residual";adapter=$ProcedureAdapter;weight=$PW} }
    $spec=Join-Path $dir "composition.json"; @{schema_version="2.3";ingredients=$ingredients} | ConvertTo-Json -Depth 8 | Set-Content $spec; $model=Join-Path $dir "model"
    if (Should-Run (Join-Path $model "v23_composition_manifest.json")) { Invoke-PythonStep "[Compose] $Name" @(".\scripts\phase_9_student_distillation\compose_residual_adapters_v23.py","--base-model",$GreedyAnchorModel,"--spec",$spec,"--output-dir",$model,"--force") }
    return $model
}

$Candidates=@()
function Add-Candidate {
    param([string]$Name,[string]$Track,[string]$Model,[string]$EvalRoot)
    $script:Candidates += [ordered]@{name=$Name;track=$Track;model=$Model;adapter=$null;evaluation_manifest=(Join-Path $EvalRoot "candidate_evaluation_manifest.json")}
}
$GreedyEval=Join-Path $OutputRoot "eval_greedy_anchor"; Eval-FullCandidate "[DEV/OOD] Greedy v2.2 anchor" $GreedyAnchorModel $GreedyEval; Add-Candidate "greedy_anchor" "baseline_v22_greedy" $GreedyAnchorModel $GreedyEval
# Historical original anchor is evaluated for regression visibility but is not part of v2.3 selection.
$HistoricalEval=Join-Path $OutputRoot "eval_historical_original_anchor"; Eval-FullCandidate "[DEV/OOD diagnostic] Historical original anchor" $OriginalAnchor $HistoricalEval
if ($ReasoningWeight -gt 0 -and (Test-Path $ReasoningDoseModel)) { $e=Join-Path $OutputRoot "eval_best_reasoning_dose"; Eval-FullCandidate "[DEV/OOD] Best reasoning dose" $ReasoningDoseModel $e; Add-Candidate "best_reasoning_dose" "residual_reasoning" $ReasoningDoseModel $e }
if ($ProcedureWeight -gt 0 -and (Test-Path $ProcedureDoseModel)) { $e=Join-Path $OutputRoot "eval_best_procedure_dose"; Eval-FullCandidate "[DEV/OOD] Best procedure dose" $ProcedureDoseModel $e; Add-Candidate "best_procedure_dose" "residual_procedure" $ProcedureDoseModel $e }
$JointSpecs=@()
if ($ReasoningWeight -gt 0 -or $ProcedureWeight -gt 0) {
    $JointSpecs += @([ordered]@{name="best_joint";rw=$ReasoningWeight;pw=$ProcedureWeight},[ordered]@{name="half_joint";rw=0.5*$ReasoningWeight;pw=0.5*$ProcedureWeight},[ordered]@{name="reasoning_full_procedure_half";rw=$ReasoningWeight;pw=0.5*$ProcedureWeight},[ordered]@{name="reasoning_half_procedure_full";rw=0.5*$ReasoningWeight;pw=$ProcedureWeight})
}
foreach ($j in $JointSpecs) {
    if (($j.rw -le 0 -and $j.pw -le 0) -or ($j.rw -eq $ReasoningWeight -and $j.pw -eq 0) -or ($j.pw -eq $ProcedureWeight -and $j.rw -eq 0)) { continue }
    $m=Compose-Joint $j.name ([double]$j.rw) ([double]$j.pw); if ($null -ne $m) { $e=Join-Path $OutputRoot ("eval_"+$j.name); Eval-FullCandidate ("[DEV/OOD] Joint "+$j.name) $m $e; Add-Candidate $j.name "residual_joint_composition" $m $e }
}

$CandidatesManifest=Join-Path $OutputRoot "candidates_dev_v23.json"; @{schema_version="2.3";candidates=$Candidates} | ConvertTo-Json -Depth 12 | Set-Content $CandidatesManifest
$SelectionJson=Join-Path $OutputRoot "residual_capability_v23_selection.json"; $SelectionMd=Join-Path $OutputRoot "RESIDUAL_CAPABILITY_V23_SELECTION.md"
if (Should-Run $SelectionJson) { Invoke-PythonStep "[Select] v2.3 residual capability matrix / Pareto frontier" @(".\scripts\phase_9_student_distillation\select_residual_capability_v23.py","--config",$ConfigPath,"--candidates-manifest",$CandidatesManifest,"--output-json",$SelectionJson,"--output-md",$SelectionMd) }
$Selection=Get-Content -Raw $SelectionJson | ConvertFrom-Json
Write-Host ""; Write-Host "DEV/OOD-selected v2.3 candidate: $($Selection.selected_candidate)"

# ---------------------------------------------------------------------------
# H. Mechanistic and system-level upper-bound diagnostics.
# ---------------------------------------------------------------------------
if (-not $SkipDiagnostics) {
    $diag=Join-Path $OutputRoot "diagnostics"; New-Item -ItemType Directory -Force -Path $diag | Out-Null
    if ($ReasoningReady -and $ProcedureReady -and (Test-Path $ReasoningAdapter) -and (Test-Path $ProcedureAdapter)) {
        $args=@(".\scripts\phase_9_student_distillation\analyze_adapter_interference_v23.py","--adapter",("residual_reasoning="+$ReasoningAdapter),"--adapter",("residual_procedure="+$ProcedureAdapter),"--adapter",("v22_abstention="+$AbstentionAdapter),"--adapter",("v22_track_b="+$TrackBAdapter),"--adapter",("v22_procedure="+$ProcedureTeacherAdapter),"--output-json",(Join-Path $diag "adapter_interference.json"),"--output-md",(Join-Path $diag "ADAPTER_INTERFERENCE.md"),"--top-fraction","$($Config.diagnostics.adapter_interference.top_fraction)")
        Invoke-PythonStep "[Diagnostics] Adapter interference map" $args
    }
    $oracleArgs=@(".\scripts\phase_9_student_distillation\oracle_routing_upper_bound_v23.py","--direct",("greedy="+(Join-Path $GreedyEval "v23_direct\results.jsonl")),"--rag",("greedy="+(Join-Path $GreedyEval "v23_procedural_rag\results.jsonl")),"--output-json",(Join-Path $diag "oracle_routing_upper_bound.json"),"--output-md",(Join-Path $diag "ORACLE_ROUTING_UPPER_BOUND.md"))
    if ($ReasoningWeight -gt 0) { $oracleArgs += @("--direct",("reasoning="+(Join-Path $OutputRoot "eval_best_reasoning_dose\v23_direct\results.jsonl")),"--rag",("reasoning="+(Join-Path $OutputRoot "eval_best_reasoning_dose\v23_procedural_rag\results.jsonl"))) }
    if ($ProcedureWeight -gt 0) { $oracleArgs += @("--direct",("procedure="+(Join-Path $OutputRoot "eval_best_procedure_dose\v23_direct\results.jsonl")),"--rag",("procedure="+(Join-Path $OutputRoot "eval_best_procedure_dose\v23_procedural_rag\results.jsonl"))) }
    Invoke-PythonStep "[Diagnostics] Oracle routing upper bound" $oracleArgs
}

# ---------------------------------------------------------------------------
# J. Fresh v2.3 sealed evaluation. It touches ONLY new v2.3 sealed splits.
# ---------------------------------------------------------------------------
if ($RunFreshSealedTest) {
    if ($Selection.policy_sensitivity_disagreement) { throw "Policy sensitivity disagrees; resolve on DEV/OOD before fresh sealed TEST." }
    $SelectedModel=[string]$Selection.selected_model; Ensure-Path $SelectedModel "selected v2.3 model"
    $SealRoot=Join-Path $OutputRoot "fresh_sealed_selected"; New-Item -ItemType Directory -Force -Path $SealRoot | Out-Null
    Set-Content -Path (Join-Path $OutputRoot "FRESH_SEALED_TEST_POLICY.txt") -Value @"
v2.3 DEV/OOD-selected candidate: $($Selection.selected_candidate)
The v2.2 sealed TEST is consumed and is not used here.
Only fresh v2.3 direct and fresh v2.3 procedural Real-RAG sealed splits are evaluated.
The v2.3 RAG gate uses the frozen DISCOVERY calibration; no sealed recalibration, retraining, merging, or candidate switching is allowed.
"@
    $dArgs=@(".\scripts\phase_9_student_distillation\evaluate_residual_reasoning_v23.py","--model",$SelectedModel,"--benchmark",$DirectSealed,"--output-dir",(Join-Path $SealRoot "direct"),"--minimum-cases","500","--resume"); if ($RequireCuda) { $dArgs += "--require-cuda" }; Invoke-PythonStep "[FRESH SEALED] Direct reasoning" $dArgs
    $rArgs=@(".\scripts\phase_9_student_distillation\evaluate_rag.py","--model",$SelectedModel,"--benchmark",$V23RagSealed,"--output-dir",(Join-Path $SealRoot "procedural_rag"),"--knowledge-db",$V23RagDb,"--gate-calibration",$V23RagCalibration,"--conditions","oracle_rag,real_rag","--retrieval-selection-policy","query_aware_v23","--maximum-evidence-tokens","$MaximumEvidenceTokens","--max-new-tokens","32","--minimum-cases-per-condition","1","--minimum-supported-cases","1","--minimum-unsupported-cases","1","--require-research-ready","--resume"); if ($RequireCuda) { $rArgs += "--require-cuda" }; Invoke-PythonStep "[FRESH SEALED] Procedural Real-RAG" $rArgs
}

Write-Host ""; Write-Host "v2.3 preservation-constrained residual capability workflow complete."; Write-Host "Output: $OutputRoot"; Write-Host "Review RESIDUAL_MINING_AUDIT.md, DOSE_SWEEP_RESULT.md files, RESIDUAL_CAPABILITY_V23_SELECTION.md, and diagnostics before any architecture compression work."

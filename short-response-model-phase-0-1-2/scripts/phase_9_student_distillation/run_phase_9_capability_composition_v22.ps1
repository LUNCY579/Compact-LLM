param(
    [Parameter(Mandatory=$true)][string]$StudentModel,
    [string]$StudentAdapter = "",
    [string]$OutputRoot = ".\artifacts\phase_9_capability_composition_v22_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string]$OriginalCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$V211CorpusRoot = ".\data\phase_9_student_distillation\procedural_recovery_v211",
    [string]$V212CorpusRoot = ".\data\phase_9_student_distillation\general_reasoning_v212",
    [string]$V22CorpusRoot = ".\data\phase_9_student_distillation\multidomain_reasoning_v22",
    [int]$Seed = 22042,
    [int]$MaximumEvidenceTokens = 384,
    [ValidateSet("auto","cpu")][string]$AnchorDevice = "auto",
    [switch]$RegenerateDataset,
    [switch]$SkipTraining,
    [switch]$SkipWeightedComposition,
    [switch]$SkipLayerSeparated,
    [switch]$SkipGreedy,
    [switch]$RunSealedTest,
    [switch]$RequireCuda,
    [switch]$ReuseExisting,
    [switch]$AllowAggressiveTraining
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root
$ConfigPath = ".\configs\phase_9_capability_composition_v22.json"
$Config = Get-Content -Raw $ConfigPath | ConvertFrom-Json

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""; Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}
function Ensure-Path { param([string]$Path,[string]$Description) if (-not (Test-Path $Path)) { throw "Missing ${Description}: $Path" } }
function Should-Run { param([string]$Artifact) if (Test-Path $Artifact) { if ($ReuseExisting) { return $false } throw "Artifact already exists: $Artifact. Use -ReuseExisting deliberately." } return $true }

Ensure-Path $StudentModel "input student/base checkpoint"
if (-not [string]::IsNullOrWhiteSpace($StudentAdapter)) { Ensure-Path $StudentAdapter "input student adapter" }
Ensure-Path $OriginalCorpusRoot "original Phase 9 corpus"
Ensure-Path $V211CorpusRoot "v2.1.1 corpus"
Ensure-Path $V212CorpusRoot "v2.1.2 corpus"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$OriginalTrain = Join-Path $OriginalCorpusRoot "reviewed_splits\train.reviewed.jsonl"
$OriginalDev = Join-Path $OriginalCorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$OriginalTest = Join-Path $OriginalCorpusRoot "reviewed_splits\test.reviewed.jsonl"
$OriginalKnowledge = Join-Path $OriginalCorpusRoot "knowledge.jsonl"
$OriginalProcedures = Join-Path $OriginalCorpusRoot "procedures.jsonl"
$V211Train = Join-Path $V211CorpusRoot "reviewed_splits\train.reviewed.jsonl"
$V211Dev = Join-Path $V211CorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$V211Test = Join-Path $V211CorpusRoot "reviewed_splits\test.reviewed.jsonl"
$V211Knowledge = Join-Path $V211CorpusRoot "knowledge.jsonl"
$V211Procedures = Join-Path $V211CorpusRoot "procedures.jsonl"
$V212Dev = Join-Path $V212CorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$V212Test = Join-Path $V212CorpusRoot "reviewed_splits\test.reviewed.jsonl"

if ($RegenerateDataset -or -not (Test-Path (Join-Path $V22CorpusRoot "manifests\dataset_manifest.json"))) {
    Invoke-PythonStep "[v2.2 Data] Generate 40.5k multi-domain reasoning corpus" @(
        ".\scripts\phase_9_student_distillation\generate_multidomain_reasoning_v22.py",
        "--output-root", $V22CorpusRoot, "--original-train", $OriginalTrain, "--v211-train", $V211Train,
        "--seed", "$Seed", "--force"
    )
}
Invoke-PythonStep "[v2.2 Data] Validate multi-domain corpus" @(
    ".\scripts\phase_9_student_distillation\validate_multidomain_reasoning_v22.py",
    "--corpus-root", $V22CorpusRoot, "--require-rows", "40000"
)
$V22TrainBroad = Join-Path $V22CorpusRoot "curriculum_views\track_a_broad_general.reviewed.jsonl"
$V22TrainAnchored = Join-Path $V22CorpusRoot "curriculum_views\track_b_anchored_mixed.reviewed.jsonl"
$V22Dev = Join-Path $V22CorpusRoot "reviewed_splits\dev.reviewed.jsonl"
$V22Test = Join-Path $V22CorpusRoot "reviewed_splits\test.reviewed.jsonl"

# Materialize exactly one immutable anchor. Every independent LoRA in Tracks A-C
# and the layer-separated experiment starts from this same checkpoint.
$FrozenAnchor = Join-Path $OutputRoot "frozen_anchor"
if (Should-Run (Join-Path $FrozenAnchor "v22_anchor_manifest.json")) {
    $args = @(".\scripts\phase_9_student_distillation\materialize_student_anchor_v22.py", "--student", $StudentModel, "--output-dir", $FrozenAnchor, "--force")
    if (-not [string]::IsNullOrWhiteSpace($StudentAdapter)) { $args += @("--adapter", $StudentAdapter) }
    Invoke-PythonStep "[Anchor] Materialize frozen original student" $args
}
Ensure-Path $FrozenAnchor "frozen anchor checkpoint"

# Frozen retrieval infrastructure: DEV calibrations are reused by every candidate and TEST.
$OriginalDb = Join-Path $OutputRoot "original_phase9.sqlite3"
$V211Db = Join-Path $OutputRoot "procedural_v211.sqlite3"
if (-not (Test-Path $OriginalDb)) { Invoke-PythonStep "[Retrieval] Build original index" @(".\scripts\phase_9_student_distillation\build_knowledge_index.py","--knowledge",$OriginalKnowledge,"--procedures",$OriginalProcedures,"--output",$OriginalDb) }
if (-not (Test-Path $V211Db)) { Invoke-PythonStep "[Retrieval] Build v2.1.1 index" @(".\scripts\phase_9_student_distillation\build_knowledge_index.py","--knowledge",$V211Knowledge,"--procedures",$V211Procedures,"--output",$V211Db) }
$OriginalCalibration = Join-Path $OutputRoot "original_calibration_query_aware_v21.json"
$V211Calibration = Join-Path $OutputRoot "v211_calibration_query_aware_v21.json"
if (-not (Test-Path $OriginalCalibration)) { Invoke-PythonStep "[Retrieval] Freeze original DEV calibration" @(".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py","--benchmark",$OriginalDev,"--knowledge-db",$OriginalDb,"--output",$OriginalCalibration,"--retrieval-selection-policy","query_aware_v21","--require-research-ready") }
if (-not (Test-Path $V211Calibration)) { Invoke-PythonStep "[Retrieval] Freeze v2.1.1 DEV calibration" @(".\scripts\phase_9_student_distillation\calibrate_evidence_gate.py","--benchmark",$V211Dev,"--knowledge-db",$V211Db,"--output",$V211Calibration,"--retrieval-selection-policy","query_aware_v21","--require-research-ready") }

function Eval-Candidate {
    param([string]$Label,[string]$Model,[string]$Adapter,[string]$Destination,[switch]$Sealed)
    $manifest = Join-Path $Destination "candidate_evaluation_manifest.json"
    if (-not (Should-Run $manifest)) { return }
    $orig = if ($Sealed) { $OriginalTest } else { $OriginalDev }
    $proc = if ($Sealed) { $V211Test } else { $V211Dev }
    $multi = if ($Sealed) { $V22Test } else { $V22Dev }
    $r212 = if ($Sealed) { $V212Test } else { $V212Dev }
    $args = @(
        ".\scripts\phase_9_student_distillation\evaluate_candidate_v22.py",
        "--model", $Model, "--output-root", $Destination,
        "--original-benchmark", $orig, "--original-db", $OriginalDb, "--original-calibration", $OriginalCalibration,
        "--procedural-benchmark", $proc, "--procedural-db", $V211Db, "--procedural-calibration", $V211Calibration,
        "--multidomain-benchmark", $multi, "--v212-benchmark", $r212,
        "--maximum-evidence-tokens", "$MaximumEvidenceTokens", "--resume"
    )
    if (-not [string]::IsNullOrWhiteSpace($Adapter)) { $args += @("--adapter", $Adapter) }
    if ($RequireCuda) { $args += "--require-cuda" }
    # Full DEV and sealed TEST candidates must both be research-ready; greedy
    # probe search uses its own bounded smoke contract.
    $args += "--research-ready-rag"
    Invoke-PythonStep $Label $args
}

function Train-FreshAdapter {
    param([string]$Label,[string]$TrainFile,[string]$Destination,[double]$LearningRate,[int]$Steps,[int]$Rank,[double]$AnchorWeight=0.0,[double]$AnchorTemperature=1.0,[string]$LayerIndices="")
    $manifest = Join-Path $Destination "training_manifest.json"
    if (-not (Should-Run $manifest)) { return }
    if (-not $AllowAggressiveTraining) {
        if ($LearningRate -gt [double]$Config.safety.maximum_default_learning_rate) { throw "v2.2 guard: LR $LearningRate exceeds configured maximum." }
        if ($Steps -gt [int]$Config.safety.maximum_steps_per_adapter) { throw "v2.2 guard: steps $Steps exceed configured maximum." }
    }
    $args = @(
        ".\scripts\phase_9_student_distillation\train_student.py", "--student", $FrozenAnchor, "--objective", "sft",
        "--train-file", $TrainFile, "--dev-file", $V22Dev, "--output-dir", $Destination,
        "--max-steps", "$Steps", "--learning-rate", "$LearningRate", "--gradient-accumulation-steps", "32",
        "--max-sequence-length", "512", "--maximum-evidence-tokens", "$MaximumEvidenceTokens", "--max-completion-tokens", "32",
        "--first-answer-token-weight", "1.5", "--evidence-anchor-token-weight", "1.5", "--abstention-token-weight", "1.75", "--eos-loss-weight", "1.3",
        "--eval-steps", "80", "--seed", "$Seed", "--pipeline-revision", "phase9_capability_composition_v2_2",
        "--lora", "--lora-rank", "$Rank"
    )
    if ($AnchorWeight -gt 0) { $args += @("--anchor-model",$FrozenAnchor,"--anchor-kl-weight","$AnchorWeight","--anchor-temperature","$AnchorTemperature","--anchor-device",$AnchorDevice) }
    if (-not [string]::IsNullOrWhiteSpace($LayerIndices)) { $args += @("--lora-layer-indices",$LayerIndices) }
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep $Label $args
}

$Candidates = New-Object System.Collections.ArrayList
function Add-CandidateEntry {
    param([string]$Name,[string]$Track,[string]$Model,[string]$Adapter,[string]$EvalRoot)
    $AdapterValue = if ([string]::IsNullOrWhiteSpace($Adapter)) { $null } else { $Adapter }
    [void]$Candidates.Add([ordered]@{
        name=$Name; track=$Track; model=$Model; adapter=$AdapterValue;
        multidomain_summary=(Join-Path $EvalRoot "multidomain\summary.json");
        v212_summary=(Join-Path $EvalRoot "v212\summary.json");
        original_rag_summary=(Join-Path $EvalRoot "original_rag\summary.json");
        procedural_summary=(Join-Path $EvalRoot "procedural_rag\summary.json")
    })
}

$BaseEval = Join-Path $OutputRoot "eval_frozen_anchor"
Eval-Candidate "[DEV] Frozen anchor" $FrozenAnchor "" $BaseEval
Add-CandidateEntry "frozen_anchor" "baseline" $FrozenAnchor "" $BaseEval
if ($SkipTraining) {
    $BaseManifest = Join-Path $OutputRoot "candidates_dev.json"; @{candidates=$Candidates} | ConvertTo-Json -Depth 8 | Set-Content $BaseManifest
    Write-Host "v2.2 baseline-only workflow complete: $OutputRoot"; exit 0
}

# Track A: broad multi-domain SFT from the frozen anchor.
$TrackA = Join-Path $OutputRoot "track_a_broad_reasoning"
Train-FreshAdapter "[Track A] Broad multi-domain reasoning" $V22TrainBroad $TrackA ([double]$Config.tracks.A_broad_reasoning.learning_rate) ([int]$Config.tracks.A_broad_reasoning.max_steps) ([int]$Config.tracks.A_broad_reasoning.lora_rank)
$TrackAAdapter = Join-Path $TrackA "adapter"; Ensure-Path $TrackAAdapter "Track A adapter"
$TrackAEval = Join-Path $OutputRoot "eval_track_a"; Eval-Candidate "[DEV] Track A" $FrozenAnchor $TrackAAdapter $TrackAEval; Add-CandidateEntry "track_a_broad" "A_broad_reasoning" $FrozenAnchor $TrackAAdapter $TrackAEval

# Track B: same broad curriculum, but explicitly anchored to the original logits.
$TrackB = Join-Path $OutputRoot "track_b_anchored_reasoning"
Train-FreshAdapter "[Track B] KL-anchored broad reasoning" $V22TrainAnchored $TrackB ([double]$Config.tracks.B_anchored_reasoning.learning_rate) ([int]$Config.tracks.B_anchored_reasoning.max_steps) ([int]$Config.tracks.B_anchored_reasoning.lora_rank) ([double]$Config.tracks.B_anchored_reasoning.anchor_kl_weight) ([double]$Config.tracks.B_anchored_reasoning.anchor_temperature)
$TrackBAdapter = Join-Path $TrackB "adapter"; Ensure-Path $TrackBAdapter "Track B adapter"
$TrackBEval = Join-Path $OutputRoot "eval_track_b"; Eval-Candidate "[DEV] Track B" $FrozenAnchor $TrackBAdapter $TrackBEval; Add-CandidateEntry "track_b_anchored" "B_anchored_reasoning" $FrozenAnchor $TrackBAdapter $TrackBEval

# Track C: independent capability LoRAs. None continues another specialist.
$SpecialistAdapters = [ordered]@{}
$SpecialistViews = [ordered]@{
    general = "specialist_general.reviewed.jsonl"; grounding = "specialist_grounding.reviewed.jsonl"; procedure = "specialist_procedure.reviewed.jsonl";
    abstention = "specialist_abstention.reviewed.jsonl"; social = "specialist_social.reviewed.jsonl"
}
foreach ($cap in $SpecialistViews.Keys) {
    $dest = Join-Path $OutputRoot "track_c_${cap}"
    $view = Join-Path $V22CorpusRoot ("curriculum_views\" + $SpecialistViews[$cap])
    Train-FreshAdapter "[Track C] Specialist $cap" $view $dest ([double]$Config.tracks.C_parallel_specialists.learning_rate) ([int]$Config.tracks.C_parallel_specialists.max_steps) ([int]$Config.tracks.C_parallel_specialists.lora_rank)
    $adapter = Join-Path $dest "adapter"; Ensure-Path $adapter "specialist $cap adapter"; $SpecialistAdapters[$cap] = $adapter
    $eval = Join-Path $OutputRoot "eval_specialist_${cap}"; Eval-Candidate "[DEV] Specialist $cap" $FrozenAnchor $adapter $eval; Add-CandidateEntry "specialist_${cap}" "C_parallel_specialists" $FrozenAnchor $adapter $eval
}

# Track D1: exact weighted static LoRA-delta presets. These materialize one dense
# checkpoint with no extra inference adapter memory.
if (-not $SkipWeightedComposition) {
    foreach ($presetProp in $Config.tracks.D_composition.weighted_presets.PSObject.Properties) {
        $preset = $presetProp.Name; $weights = $presetProp.Value; $dir = Join-Path $OutputRoot "track_d_weighted_${preset}"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $ingredients = @(); foreach ($cap in $SpecialistAdapters.Keys) { $ingredients += [ordered]@{name=$cap;adapter=$SpecialistAdapters[$cap];weight=[double]$weights.$cap} }
        $spec = Join-Path $dir "composition.json"; @{schema_version="2.2";ingredients=$ingredients} | ConvertTo-Json -Depth 8 | Set-Content $spec
        $model = Join-Path $dir "model"
        if (Should-Run (Join-Path $model "v22_composition_manifest.json")) { Invoke-PythonStep "[Track D] Weighted composition $preset" @(".\scripts\phase_9_student_distillation\compose_lora_adapters_v22.py","--base-model",$FrozenAnchor,"--spec",$spec,"--output-dir",$model,"--force") }
        $eval=Join-Path $OutputRoot "eval_weighted_${preset}"; Eval-Candidate "[DEV] Weighted $preset" $model "" $eval; Add-CandidateEntry "weighted_${preset}" "D_weighted" $model "" $eval
    }
}

# Track D2: layer-separated specialists. A central band is split into disjoint
# transformer-layer groups. The mapping is a preregistered interference test,
# not a claim that capability localization is already known.
if (-not $SkipLayerSeparated) {
    $LayerPlan = Join-Path $OutputRoot "layer_plan_v22.json"
    if (-not (Test-Path $LayerPlan)) { Invoke-PythonStep "[Track D] Derive non-overlapping layer plan" @(".\scripts\phase_9_student_distillation\derive_layer_plan_v22.py","--model",$FrozenAnchor,"--output",$LayerPlan,"--central-fraction","$($Config.tracks.D_composition.layer_separated.central_fraction)","--capability-order",(($Config.tracks.D_composition.layer_separated.capability_order) -join ',')) }
    $Plan = Get-Content -Raw $LayerPlan | ConvertFrom-Json; $LayerAdapters=[ordered]@{}
    foreach ($cap in $SpecialistViews.Keys) {
        $indices = (($Plan.layers.$cap | ForEach-Object { [string]$_ }) -join ',')
        $dest = Join-Path $OutputRoot "track_d_layer_${cap}"; $view=Join-Path $V22CorpusRoot ("curriculum_views\" + $SpecialistViews[$cap])
        Train-FreshAdapter "[Track D] Layer-scoped $cap ($indices)" $view $dest ([double]$Config.tracks.C_parallel_specialists.learning_rate) ([int]$Config.tracks.C_parallel_specialists.max_steps) ([int]$Config.tracks.C_parallel_specialists.lora_rank) 0.0 1.0 $indices
        $LayerAdapters[$cap]=Join-Path $dest "adapter"; Ensure-Path $LayerAdapters[$cap] "layer specialist $cap adapter"
    }
    $LayerDir=Join-Path $OutputRoot "track_d_layer_separated_merge"; New-Item -ItemType Directory -Force -Path $LayerDir | Out-Null
    $layerIngredients=@(); foreach($cap in $LayerAdapters.Keys){$layerIngredients += [ordered]@{name=$cap;adapter=$LayerAdapters[$cap];weight=1.0}}
    $layerSpec=Join-Path $LayerDir "composition.json"; @{schema_version="2.2";ingredients=$layerIngredients} | ConvertTo-Json -Depth 8 | Set-Content $layerSpec
    $layerModel=Join-Path $LayerDir "model"; if(Should-Run (Join-Path $layerModel "v22_composition_manifest.json")){Invoke-PythonStep "[Track D] Merge disjoint-layer adapters" @(".\scripts\phase_9_student_distillation\compose_lora_adapters_v22.py","--base-model",$FrozenAnchor,"--spec",$layerSpec,"--output-dir",$layerModel,"--force")}
    $layerEval=Join-Path $OutputRoot "eval_layer_separated"; Eval-Candidate "[DEV] Layer-separated composition" $layerModel "" $layerEval; Add-CandidateEntry "layer_separated" "D_layer_separated" $layerModel "" $layerEval
}

# Track D3: true greedy DEV-probe composition. Each addition is accepted only
# after evaluating the actual materialized mixture. TEST is never used here.
if (-not $SkipGreedy) {
    $GreedyRoot=Join-Path $OutputRoot "track_d_greedy_soup"
    $gargs=@(
        ".\scripts\phase_9_student_distillation\run_greedy_adapter_soup_v22.py","--config",$ConfigPath,"--base-model",$FrozenAnchor,"--output-root",$GreedyRoot,
        "--original-benchmark",$OriginalDev,"--original-db",$OriginalDb,"--original-calibration",$OriginalCalibration,
        "--procedural-benchmark",$V211Dev,"--procedural-db",$V211Db,"--procedural-calibration",$V211Calibration,
        "--multidomain-benchmark",$V22Dev,"--v212-benchmark",$V212Dev,"--maximum-evidence-tokens","$MaximumEvidenceTokens",
        "--adapter",("anchored_general="+$TrackBAdapter)
    )
    foreach($cap in $SpecialistAdapters.Keys){$gargs += @("--adapter",($cap+"="+$SpecialistAdapters[$cap]))}
    if($RequireCuda){$gargs += "--require-cuda"}
    if(Should-Run (Join-Path $GreedyRoot "GREEDY_SOUP_RESULT.json")){Invoke-PythonStep "[Track D] Greedy adapter soup" $gargs}
    $Greedy = Get-Content -Raw (Join-Path $GreedyRoot "GREEDY_SOUP_RESULT.json") | ConvertFrom-Json
    $greedyModel=[string]$Greedy.final_model; $greedyEval=Join-Path $GreedyRoot "full_dev_eval"
    Add-CandidateEntry "greedy_soup" "D_greedy" $greedyModel "" $greedyEval
}

# Unified capability matrix and Pareto selection.
$CandidatesManifest=Join-Path $OutputRoot "candidates_dev.json"
if (-not (Test-Path $CandidatesManifest) -or -not $ReuseExisting) { @{schema_version="2.2";candidates=$Candidates} | ConvertTo-Json -Depth 10 | Set-Content $CandidatesManifest }
$SelectionJson=Join-Path $OutputRoot "capability_composition_v22_selection.json"; $SelectionMd=Join-Path $OutputRoot "CAPABILITY_COMPOSITION_V22_SELECTION.md"
if (Should-Run $SelectionJson) {
    Invoke-PythonStep "[Select] Build capability matrix / Pareto frontier" @(".\scripts\phase_9_student_distillation\select_capability_composition_v22.py","--config",$ConfigPath,"--candidates-manifest",$CandidatesManifest,"--output-json",$SelectionJson,"--output-md",$SelectionMd)
}
$Selection=Get-Content -Raw $SelectionJson | ConvertFrom-Json
Write-Host ""; Write-Host "DEV-selected v2.2 candidate: $($Selection.selected_candidate)"

if($RunSealedTest){
    if($Selection.policy_sensitivity_disagreement){throw "Policy sensitivity disagrees; resolve on DEV before sealed TEST."}
    $SelectedModel=[string]$Selection.selected_model; $SelectedAdapter=if($null -eq $Selection.selected_adapter){""}else{[string]$Selection.selected_adapter}
    Set-Content -Path (Join-Path $OutputRoot "SEALED_TEST_POLICY.txt") -Value @"
DEV-selected candidate: $($Selection.selected_candidate)
Original and procedural TEST use frozen DEV calibrations.
No TEST-based training, recalibration, merging, or candidate switching is allowed.
"@
    Eval-Candidate "[SEALED] Selected v2.2 candidate" $SelectedModel $SelectedAdapter (Join-Path $OutputRoot "sealed_selected_test") -Sealed
}

Write-Host ""; Write-Host "v2.2 capability-composition workflow complete."; Write-Host "Output: $OutputRoot"; Write-Host "Review CAPABILITY_COMPOSITION_V22_SELECTION.md and per-track reports before Student Optimization v2."

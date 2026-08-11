param(
    [string]$V23RunRoot = ".\artifacts\phase_9_residual_capability_v23",
    [string]$V22RunRoot = ".\artifacts\phase_9_capability_composition_v22_lr1e4_s1200",
    [string]$OutputRoot = ".\artifacts\phase_9_grokking_dynamics_v231_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string]$ConfigPath = ".\configs\phase_9_grokking_dynamics_v231.json",
    [string]$V23DirectRoot = ".\data\phase_9_student_distillation\residual_capability_v23\direct_reasoning",
    [string]$V23RagRoot = ".\data\phase_9_student_distillation\residual_capability_v23\procedural_rag",
    [ValidateSet("pilot","standard","extended")][string]$Profile = "standard",
    [string]$AnchorDevice = "auto",
    [switch]$RequireCuda,
    [switch]$ReuseExisting,
    [switch]$SkipCoreTraining,
    [switch]$SkipDoseSurface,
    [switch]$SkipRepresentation,
    [switch]$SkipFactorSweep,
    [switch]$SkipProcedureControl,
    [switch]$SkipAnchorRelativeNorm,
    [int]$FactorMaxRuns = 0
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

$Python = "python"
$LocalVenv = Join-Path $Root ".venv\Scripts\python.exe"
$ParentVenv = Join-Path (Split-Path $Root -Parent) ".venv\Scripts\python.exe"
if (Test-Path $LocalVenv) { $Python = $LocalVenv }
elseif (Test-Path $ParentVenv) { $Python = $ParentVenv }

function Invoke-PythonStep([string]$Label, [object[]]$Args) {
    Write-Host ""; Write-Host "=== $Label ===" -ForegroundColor Cyan
    & $Python @Args | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}
function Ensure-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path $Path)) { throw "Missing ${Label}: $Path" }
}
function Latest-Checkpoint([string]$RootPath) {
    $root = Join-Path $RootPath "checkpoints"
    if (-not (Test-Path $root)) { return $null }
    $items = Get-ChildItem -Directory $root -Filter "checkpoint-*" | ForEach-Object {
        $m = [regex]::Match($_.Name, 'checkpoint-(\d+)$')
        if ($m.Success) { [PSCustomObject]@{ Step=[int]$m.Groups[1].Value; Path=$_.FullName } }
    } | Sort-Object Step -Descending
    if ($items.Count -gt 0) { return $items[0].Path }
    return $null
}

Ensure-Path $ConfigPath "v2.3.1 config"
Ensure-Path $V23RunRoot "completed v2.3 run root"
Ensure-Path $V22RunRoot "completed v2.2 run root"
Ensure-Path $V23DirectRoot "v2.3 direct corpus"
Ensure-Path $V23RagRoot "v2.3 procedural corpus"

$Config = Get-Content -Raw $ConfigPath | ConvertFrom-Json
$ProfileCfg = $Config.profiles.$Profile
$Core = $Config.core_training
$GreedyAnchor = Join-Path $V22RunRoot "track_d_greedy_soup\final_model"
$ReasoningResidual = Join-Path $V23RunRoot "residual_mining\mined\reasoning_residual.reviewed.jsonl"
$ProcedureResidual = Join-Path $V23RunRoot "residual_mining\mined\procedure_residual.reviewed.jsonl"
$Preservation = Join-Path $V23RunRoot "residual_mining\mined\preservation_replay.reviewed.jsonl"
$ReasoningV23Mix = Join-Path $V23RunRoot "residual_mining\training_views\reasoning_residual_mixed.reviewed.jsonl"
$ProcedureV23Mix = Join-Path $V23RunRoot "residual_mining\training_views\procedure_residual_mixed.reviewed.jsonl"
$MiningAudit = Join-Path $V23RunRoot "residual_mining\mined\RESIDUAL_MINING_AUDIT.json"
$V23RagDb = Join-Path $V23RunRoot "v23_procedural.sqlite3"
$V23RagCalibration = Join-Path $V23RunRoot "v23_procedural_discovery_calibration.json"
foreach ($pair in @(
    @($GreedyAnchor,"v2.2 greedy anchor"),
    @($ReasoningResidual,"v2.3 reasoning residual"),
    @($ProcedureResidual,"v2.3 procedure residual"),
    @($Preservation,"v2.3 preservation replay"),
    @($ReasoningV23Mix,"v2.3 reasoning mix"),
    @($MiningAudit,"v2.3 mining audit")
)) { Ensure-Path $pair[0] $pair[1] }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
Invoke-PythonStep "Validate v2.3.1 prerequisites" @(
    ".\scripts\phase_9_student_distillation\validate_grokking_dynamics_v231.py",
    "--config",$ConfigPath,"--v23-root",$V23RunRoot,"--v22-root",$V22RunRoot,"--direct-root",$V23DirectRoot,
    "--output",(Join-Path $OutputRoot "PREREQUISITE_AUDIT.json")
)

$DataRoot = Join-Path $OutputRoot "data_views"
$DataManifest = Join-Path $DataRoot "grokking_data_manifest.json"
if (-not ($ReuseExisting -and (Test-Path $DataManifest))) {
    Invoke-PythonStep "Build non-sealed training variants and probes" @(
        ".\scripts\phase_9_student_distillation\build_grokking_variants_v231.py",
        "--config",$ConfigPath,
        "--reasoning-residual",$ReasoningResidual,"--preservation",$Preservation,"--original-v23-mix",$ReasoningV23Mix,
        "--dev",(Join-Path $V23DirectRoot "splits\dev.reviewed.jsonl"),"--ood-dev",(Join-Path $V23DirectRoot "splits\ood_dev.reviewed.jsonl"),
        "--output-root",$DataRoot,"--seed","$($Config.probes.seed)"
    )
}
$TrainProbe = Join-Path $DataRoot "probes\train_answerable.reviewed.jsonl"
$DevProbe = Join-Path $DataRoot "probes\dev.reviewed.jsonl"
$OodProbe = Join-Path $DataRoot "probes\ood.reviewed.jsonl"
$RepTrainProbe = Join-Path $DataRoot "probes\representation_train.reviewed.jsonl"
$RepOodProbe = Join-Path $DataRoot "probes\representation_ood.reviewed.jsonl"
$BaselineMix = Join-Path $DataRoot "training_variants\baseline_original_v23.reviewed.jsonl"

# -------------------------------------------------------------------------
# A. Core long-horizon reasoning run: exact v2.3 mix, but much longer and
#    instrumented. Trainer keeps only two resumable optimizer checkpoints;
#    lightweight adapter snapshots are kept at every save event.
# -------------------------------------------------------------------------
$CoreRoot = Join-Path $OutputRoot "core_reasoning_long_horizon"
$CoreManifest = Join-Path $CoreRoot "training_manifest.json"
if (-not $SkipCoreTraining -and -not ($ReuseExisting -and (Test-Path $CoreManifest))) {
    $trainArgs = @(
        ".\scripts\phase_9_student_distillation\train_student.py",
        "--student",$GreedyAnchor,"--objective","sft","--train-file",$BaselineMix,"--output-dir",$CoreRoot,
        "--max-steps","$($ProfileCfg.core_max_steps)","--learning-rate","$($Core.learning_rate)","--weight-decay","$($Core.weight_decay)",
        "--gradient-accumulation-steps","$($Core.gradient_accumulation_steps)","--warmup-ratio","$($Core.warmup_ratio)",
        "--max-sequence-length","$($Core.max_sequence_length)","--maximum-evidence-tokens","$($Core.maximum_evidence_tokens)","--max-completion-tokens","$($Core.max_completion_tokens)",
        "--first-answer-token-weight","1.5","--evidence-anchor-token-weight","1.5","--abstention-token-weight","1.5","--eos-loss-weight","1.3",
        "--seed","$($Core.seed)","--pipeline-revision","phase9_grokking_dynamics_v2_3_1_core",
        "--lora","--lora-rank","$($Core.lora_rank)","--target-module-group","$($Core.target_module_group)",
        "--anchor-model",$GreedyAnchor,"--anchor-kl-weight","$($Core.anchor_kl_weight)","--anchor-temperature","$($Core.anchor_temperature)","--anchor-device",$AnchorDevice,
        "--save-steps","$($ProfileCfg.core_save_steps)","--save-total-limit","2","--keep-checkpoints","--no-save-only-model-checkpoints",
        "--dynamics-log-file",(Join-Path $CoreRoot "optimizer_dynamics.jsonl"),"--dynamics-snapshot-dir",(Join-Path $CoreRoot "dynamics_snapshots"),
        "--dynamics-gradient-interval","$($Core.dynamics_gradient_interval)","--dynamics-ema-beta","$($Core.dynamics_ema_beta)"
    )
    if ($ReuseExisting) {
        $latest = Latest-Checkpoint $CoreRoot
        if ($null -ne $latest) { $trainArgs += @("--resume-from-checkpoint",$latest) }
    }
    if ($RequireCuda) { $trainArgs += "--require-cuda" }
    Invoke-PythonStep "Core long-horizon anchored reasoning training" $trainArgs
}
Ensure-Path (Join-Path $CoreRoot "adapter") "core final adapter"

$CurveRoot = Join-Path $OutputRoot "core_time_curve"
$TimeCurve = Join-Path $CurveRoot "training_time_curve.json"
if (-not ($ReuseExisting -and (Test-Path $TimeCurve))) {
    $args = @(
        ".\scripts\phase_9_student_distillation\evaluate_grokking_time_curve_v231.py",
        "--training-root",$CoreRoot,"--base-model",$GreedyAnchor,"--train-probe",$TrainProbe,"--dev-probe",$DevProbe,"--ood-probe",$OodProbe,
        "--output-root",$CurveRoot,"--dose","1.0","--logit-limit","$($Config.probes.logit_cases_per_split)","--reuse-existing"
    )
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep "Evaluate every core checkpoint" $args
}

$AdapterDynamics = Join-Path $OutputRoot "adapter_dynamics.json"
if (-not ($ReuseExisting -and (Test-Path $AdapterDynamics))) {
    $args = @(
        ".\scripts\phase_9_student_distillation\analyze_lora_dynamics_v231.py","--training-root",$CoreRoot,"--base-model",$GreedyAnchor,
        "--output-json",$AdapterDynamics,"--output-csv",(Join-Path $OutputRoot "adapter_dynamics.csv")
    )
    if (-not $SkipAnchorRelativeNorm) { $args += "--include-anchor-relative-norm" }
    Invoke-PythonStep "Analyze LoRA norm/rank dynamics" $args
}

$DoseRoot = Join-Path $OutputRoot "dose_time_surface"
$DoseSurface = Join-Path $DoseRoot "dose_time_surface.json"
if ($SkipDoseSurface) {
    if (-not (Test-Path $DoseSurface)) { Set-Content -Path $DoseSurface -Value '{"schema_version":"2.3.1","rows":[],"steps":[],"doses":[],"sealed_data_used":false}' }
} elseif (-not ($ReuseExisting -and (Test-Path $DoseSurface))) {
    $args = @(
        ".\scripts\phase_9_student_distillation\evaluate_grokking_dose_surface_v231.py","--config",$ConfigPath,"--training-root",$CoreRoot,"--base-model",$GreedyAnchor,
        "--time-curve",$TimeCurve,"--train-probe",$TrainProbe,"--dev-probe",$DevProbe,"--ood-probe",$OodProbe,"--output-root",$DoseRoot,"--reuse-existing"
    )
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep "Evaluate adaptive step x dose surface" $args
}

$Representation = Join-Path $OutputRoot "representation_dynamics.json"
if (-not $SkipRepresentation -and -not ($ReuseExisting -and (Test-Path $Representation))) {
    $args = @(
        ".\scripts\phase_9_student_distillation\analyze_representation_dynamics_v231.py","--config",$ConfigPath,"--training-root",$CoreRoot,"--base-model",$GreedyAnchor,
        "--train-probe",$RepTrainProbe,"--ood-probe",$RepOodProbe,"--time-curve",$TimeCurve,"--output-json",$Representation,"--profile",$Profile
    )
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep "Probe cross-format representation dynamics" $args
}

# -------------------------------------------------------------------------
# B. One-factor phase diagram. This is intentionally NOT a Cartesian grid.
# -------------------------------------------------------------------------
$FactorRoot = Join-Path $OutputRoot "factor_sweep"
$FactorManifest = Join-Path $FactorRoot "factor_sweep_manifest.json"
if (-not $SkipFactorSweep -and -not ($ReuseExisting -and (Test-Path $FactorManifest))) {
    $args = @(
        ".\scripts\phase_9_student_distillation\run_grokking_factor_sweep_v231.py","--config",$ConfigPath,"--variants-manifest",$DataManifest,"--base-model",$GreedyAnchor,
        "--train-probe",$TrainProbe,"--dev-probe",$DevProbe,"--ood-probe",$OodProbe,"--output-root",$FactorRoot,"--profile",$Profile,"--reuse-existing"
    )
    if ($FactorMaxRuns -gt 0) { $args += @("--max-runs","$FactorMaxRuns") }
    if ($RequireCuda) { $args += "--require-cuda" }
    Invoke-PythonStep "Run one-factor grokking phase screen" $args
}
$FactorPhase = Join-Path $OutputRoot "factor_phase_diagram.json"
if (Test-Path $FactorManifest) {
    Invoke-PythonStep "Summarize factor phase diagram" @(
        ".\scripts\phase_9_student_distillation\summarize_grokking_factors_v231.py","--config",$ConfigPath,"--factor-manifest",$FactorManifest,
        "--output-json",$FactorPhase,"--output-csv",(Join-Path $OutputRoot "factor_phase_diagram.csv"),"--output-md",(Join-Path $OutputRoot "FACTOR_PHASE_DIAGRAM.md")
    )
}

# -------------------------------------------------------------------------
# C. Procedure residual is a negative/contrast control. It is not allowed to
#    influence the reasoning grokking label or touch sealed data.
# -------------------------------------------------------------------------
$ProcedureControlCurve = Join-Path $OutputRoot "procedure_control\procedure_time_curve.json"
if (-not $SkipProcedureControl -and [bool]$Config.controls.procedure_long_horizon_control -and (Test-Path $ProcedureV23Mix) -and (Test-Path $V23RagDb) -and (Test-Path $V23RagCalibration)) {
    $PTrain = Join-Path $OutputRoot "procedure_control\training"
    $PManifest = Join-Path $PTrain "training_manifest.json"
    $PSteps = [math]::Max(100,[int]([double]$ProfileCfg.core_max_steps * [double]$Config.controls.procedure_control_steps_fraction_of_core))
    if (-not ($ReuseExisting -and (Test-Path $PManifest))) {
        $args = @(
            ".\scripts\phase_9_student_distillation\train_student.py","--student",$GreedyAnchor,"--objective","sft","--train-file",$ProcedureV23Mix,"--output-dir",$PTrain,
            "--max-steps","$PSteps","--learning-rate","5e-6","--weight-decay","0.0","--gradient-accumulation-steps","32","--warmup-ratio","0.05",
            "--max-sequence-length","768","--maximum-evidence-tokens","384","--max-completion-tokens","32","--first-answer-token-weight","1.5","--evidence-anchor-token-weight","1.5","--abstention-token-weight","1.5","--eos-loss-weight","1.3",
            "--seed","$($Core.seed)","--pipeline-revision","phase9_grokking_dynamics_v2_3_1_procedure_control","--lora","--lora-rank","16",
            "--anchor-model",$GreedyAnchor,"--anchor-kl-weight","0.35","--anchor-temperature","1.5","--anchor-device",$AnchorDevice,
            "--save-steps","$($ProfileCfg.core_save_steps)","--save-total-limit","2","--keep-checkpoints","--no-save-only-model-checkpoints",
            "--dynamics-log-file",(Join-Path $PTrain "optimizer_dynamics.jsonl"),"--dynamics-snapshot-dir",(Join-Path $PTrain "dynamics_snapshots")
        )
        if ($RequireCuda) { $args += "--require-cuda" }
        Invoke-PythonStep "Procedure long-horizon contrast control" $args
    }
    if (-not ($ReuseExisting -and (Test-Path $ProcedureControlCurve))) {
        $args = @(
            ".\scripts\phase_9_student_distillation\evaluate_grokking_procedure_curve_v231.py","--training-root",$PTrain,"--base-model",$GreedyAnchor,
            "--dev-benchmark",(Join-Path $V23RagRoot "splits\dev.reviewed.jsonl"),"--ood-benchmark",(Join-Path $V23RagRoot "splits\ood_dev.reviewed.jsonl"),
            "--knowledge-db",$V23RagDb,"--calibration",$V23RagCalibration,"--output-root",(Join-Path $OutputRoot "procedure_control"),"--reuse-existing"
        )
        if ($RequireCuda) { $args += "--require-cuda" }
        Invoke-PythonStep "Evaluate procedure-control dynamics" $args
    }
}

# -------------------------------------------------------------------------
# D. Synthesis: strict diagnostic label, sensitivity, and full observatory report.
# -------------------------------------------------------------------------
$Transition = Join-Path $OutputRoot "transition_detection.json"
$detectArgs = @(
    ".\scripts\phase_9_student_distillation\detect_grokking_transitions_v231.py","--config",$ConfigPath,"--time-curve",$TimeCurve,"--dose-surface",$DoseSurface,
    "--adapter-dynamics",$AdapterDynamics,"--optimizer-dynamics",(Join-Path $CoreRoot "optimizer_dynamics.jsonl"),
    "--output-json",$Transition,"--output-md",(Join-Path $OutputRoot "TRANSITION_DETECTION.md")
)
if (Test-Path $FactorManifest) { $detectArgs += @("--factor-manifest",$FactorManifest) }
if (Test-Path $Representation) { $detectArgs += @("--representation",$Representation) }
Invoke-PythonStep "Detect delayed-generalization / readout-threshold signatures" $detectArgs

$compileArgs = @(
    ".\scripts\phase_9_student_distillation\compile_grokking_report_v231.py","--config",$ConfigPath,"--data-manifest",$DataManifest,"--time-curve",$TimeCurve,"--dose-surface",$DoseSurface,
    "--transition",$Transition,"--adapter-dynamics",$AdapterDynamics,"--v23-mining-audit",$MiningAudit,
    "--output-json",(Join-Path $OutputRoot "grokking_dynamics_report.json"),"--output-md",(Join-Path $OutputRoot "GROKKING_DYNAMICS_REPORT.md")
)
if (Test-Path $Representation) { $compileArgs += @("--representation",$Representation) }
if (Test-Path $FactorPhase) { $compileArgs += @("--factor-phase",$FactorPhase) }
if (Test-Path $ProcedureControlCurve) { $compileArgs += @("--procedure-control-curve",$ProcedureControlCurve) }
Invoke-PythonStep "Compile v2.3.1 Grokking Dynamics Observatory" $compileArgs

Write-Host ""; Write-Host "v2.3.1 Grokking Dynamics Observatory complete." -ForegroundColor Green
Write-Host "Output: $OutputRoot"
Write-Host "Primary report: $(Join-Path $OutputRoot 'GROKKING_DYNAMICS_REPORT.md')"
Write-Host "No sealed evaluation is part of this workflow."

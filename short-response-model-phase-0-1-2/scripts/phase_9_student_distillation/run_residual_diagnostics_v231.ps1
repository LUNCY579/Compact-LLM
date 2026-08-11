param(
    [string]$V23Root = ".\artifacts\phase_9_residual_capability_v23",
    [string]$V22RunRoot = ".\artifacts\phase_9_capability_composition_v22_lr1e4_s1200",
    [string]$GreedyAnchorModel = "",
    [string]$OriginalCorpusRoot = ".\data\phase_9_student_distillation\rag_reasoning_v3_large",
    [string]$V211CorpusRoot = ".\data\phase_9_student_distillation\procedural_recovery_v211",
    [string]$V212CorpusRoot = ".\data\phase_9_student_distillation\general_reasoning_v212",
    [string]$V22CorpusRoot = ".\data\phase_9_student_distillation\multidomain_reasoning_v22",
    [string]$V23DirectRoot = ".\data\phase_9_student_distillation\residual_capability_v23\direct_reasoning",
    [string]$V23RagRoot = ".\data\phase_9_student_distillation\residual_capability_v23\procedural_rag",
    [switch]$RequireCuda,
    [switch]$ReuseExisting,
    [switch]$ForceCompose,
    [switch]$SkipGpu
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

# Prefer the repo venv (torch/peft) over PATH python (e.g. hermes) when present.
$Python = "python"
$RepoVenv = Join-Path (Resolve-Path (Join-Path $Root "..")).Path ".venv\Scripts\python.exe"
$LocalVenv = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $LocalVenv) { $Python = $LocalVenv }
elseif (Test-Path $RepoVenv) { $Python = $RepoVenv }

if ([string]::IsNullOrWhiteSpace($GreedyAnchorModel)) {
    $GreedyAnchorModel = Join-Path $V22RunRoot "track_d_greedy_soup\final_model"
}

$args = @(
    ".\scripts\phase_9_student_distillation\run_residual_diagnostics_v231.py",
    "--v23-root", $V23Root,
    "--config", ".\configs\phase_9_residual_capability_v23.json",
    "--greedy-model", $GreedyAnchorModel,
    "--v22-run-root", $V22RunRoot,
    "--original-corpus-root", $OriginalCorpusRoot,
    "--v211-corpus-root", $V211CorpusRoot,
    "--v212-corpus-root", $V212CorpusRoot,
    "--v22-corpus-root", $V22CorpusRoot,
    "--v23-direct-root", $V23DirectRoot,
    "--v23-rag-root", $V23RagRoot
)
if ($RequireCuda) { $args += "--require-cuda" }
if ($ReuseExisting) { $args += "--reuse-existing" }
if ($ForceCompose) { $args += "--force-compose" }
if ($SkipGpu) { $args += "--skip-gpu" }

Write-Host "v2.3.1 diagnostic-completion hotfix using $Python"
& $Python @args | Out-Host
if ($LASTEXITCODE -ne 0) { throw "v2.3.1 diagnostics failed with exit code $LASTEXITCODE." }
Write-Host "Done. Review $V23Root\diagnostics_v231\ and RESIDUAL_CAPABILITY_V23_RESEARCH_MATRIX.md"

<#
.SYNOPSIS
Evaluate retrieval/context budgeting and benchmark exact prefix-KV reuse.

.DESCRIPTION
The runner resolves the Phase 4B-selected checkpoint, executes model-aware
compact-versus-full context evaluation, benchmarks stable-prefix cache reuse,
and generates both the evidence report and pending human stop gate.
#>
param(
    [string]$Model = "",
    [string]$Cases = ".\data\context\context_eval.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_5_context",
    [int]$TopK = 6,
    [int]$TotalBudget = 512,
    [int]$MemoryBudget = 160,
    [int]$RecentBudget = 256,
    [int]$MaxNewTokens = 32,
    [int]$PrefixRepeats = 3
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

Invoke-PythonStep "[1/4] Checking Phase 4B human approval..." @(
    ".\scripts\common\check_human_gate.py", "--phase", "4B"
)
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = python .\scripts\common\resolve_reference_model.py
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve the Phase 4B-selected model." }
}
if (-not (Test-Path $Model)) { throw "Phase 5 requires a local selected checkpoint: $Model" }
if (-not (Test-Path $Cases)) { throw "Missing Phase 5 context cases: $Cases" }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Invoke-PythonStep "[2/4] Evaluating retrieval, budgets, and compact-context generation..." @(
    ".\scripts\phase_5_context_system\evaluate_context_system.py",
    "--cases", $Cases,
    "--model", $Model,
    "--out-dir", $OutputRoot,
    "--top-k", "$TopK",
    "--total-budget", "$TotalBudget",
    "--memory-budget", "$MemoryBudget",
    "--recent-budget", "$RecentBudget",
    "--max-new-tokens", "$MaxNewTokens"
)

$PrefixResult = Join-Path $OutputRoot "prefix_cache.json"
Invoke-PythonStep "[3/4] Benchmarking stable-prefix KV reuse..." @(
    ".\scripts\phase_5_context_system\benchmark_prefix_cache.py",
    "--model", $Model,
    "--cases", $Cases,
    "--out", $PrefixResult,
    "--repeats", "$PrefixRepeats"
)

Invoke-PythonStep "[4/4] Creating Phase 5 report and stop gate..." @(
    ".\scripts\phase_5_context_system\check_phase_5_gate.py",
    "--evaluation", (Join-Path $OutputRoot "evaluation.json"),
    "--prefix-cache", $PrefixResult
)

Write-Host ""
Write-Host "Phase 5 execution completed."
Write-Host "Evidence: $OutputRoot\evaluation.json and retrieval_traces.jsonl"
Write-Host "Report: reports\CONTEXT_SYSTEM_REPORT.md"
Write-Host "Stop gate: reports\PHASE_5_STOP_GATE.md"
Write-Host "Review the traces and raw replies before changing Human decision to PASS."

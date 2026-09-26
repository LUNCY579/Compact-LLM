#requires -Version 7.0
# Fixed experiment sequence. Restart this controller after an app interruption:
# complete arms are validated/skipped; incomplete arms recover committed state.
# No quality-selected checkpoint, additional optimizer budget, Phase 2, or composition.
[CmdletBinding()]
param([switch]$StopAfterPrimarySmoke)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$pilotLab = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pilotRepo = (Resolve-Path -LiteralPath (Join-Path $pilotLab '../..')).Path
$pilotPython = (Resolve-Path -LiteralPath (Join-Path $pilotRepo '../Chat-Bubble/.venv/Scripts/python.exe')).Path
$pilotRunner = Join-Path $pilotLab 'scripts/phase1_run.py'
$pilotAuditor = Join-Path $pilotLab 'scripts/audit_phase1_results.py'
$pilotExecution = [Guid]::NewGuid().ToString('N')
$pilotAuditDir = Join-Path $pilotLab "results/audit_20260905/resumable_$pilotExecution"
$pilotStatusFile = Join-Path $PSScriptRoot 'resumable_phase1_status.json'
$pilotCancelFile = Join-Path $PSScriptRoot 'resumable_phase1.cancel'
$pilotStart = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
$pilotLock = $null
$pilotOrdinal = 0
$pilotPolicy = 'backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8'

function Write-PilotStatus([string]$State, [hashtable]$Details = @{}) {
    $payload = [ordered]@{
        state = $State; updated_utc = [DateTime]::UtcNow.ToString('o')
        controller_pid = $PID; controller_start_time_utc = $pilotStart
        execution_id = $pilotExecution; private_audit_directory = $pilotAuditDir
        script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        detail = $Details; phase2_execution_allowed = $false; composition_execution_allowed = $false
        final_report_pending = $true
    }
    $temporary = "$pilotStatusFile.$pilotExecution.tmp"
    [IO.File]::WriteAllText($temporary, ($payload | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temporary, $pilotStatusFile, $true)
}

function Assert-NotCancelled {
    if (Test-Path -LiteralPath $pilotCancelFile) { throw 'Cancellation marker present; no further command starts' }
}

function Wait-PilotGpu([string]$Label) {
    while ($true) {
        Assert-NotCancelled
        $jobs = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'phase1_run\.py' })
        $lines = @(& nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
        if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 1) { throw 'GPU preflight query failed' }
        $free = [int]$lines[0]
        if ($jobs.Count -eq 0 -and $free -ge 5500) { return $free }
        Write-PilotStatus 'WAITING_FOR_GPU_OR_EXISTING_JOB' @{ next = $Label; free_mib = $free; live_model_pids = @($jobs | ForEach-Object { $_.ProcessId }) }
        Start-Sleep -Seconds 60
    }
}

function Invoke-PilotPython([string]$Label, [string[]]$Arguments, [switch]$CpuOnly) {
    Assert-NotCancelled
    $script:pilotOrdinal += 1
    $log = Join-Path $PSScriptRoot ('resumable_{0}_{1:D2}_{2}.log' -f $pilotExecution, $script:pilotOrdinal, $Label)
    $priorVisibility = $env:CUDA_VISIBLE_DEVICES
    $exit = $null
    if ($CpuOnly) { $env:CUDA_VISIBLE_DEVICES = '-1' }
    Write-PilotStatus 'RUNNING_COMMAND' @{ label = $Label; arguments = $Arguments; log = $log; cpu_only = [bool]$CpuOnly }
    try {
        & $pilotPython -u @Arguments *> $log
        $exit = $LASTEXITCODE
    }
    finally { $env:CUDA_VISIBLE_DEVICES = $priorVisibility }
    if ($exit -ne 0) { throw "$Label failed with exit $exit; inspect $log. Preserve occupied arms and recover only from validated state." }
    Assert-NotCancelled
}

function Assert-TrueChecks($Checks, [string]$Label) {
    $fields = @($Checks.PSObject.Properties)
    if ($fields.Count -eq 0) { throw "Missing checks: $Label" }
    foreach ($field in $fields) { if ($field.Value -isnot [bool] -or $field.Value -ne $true) { throw "Failed check: $Label/$($field.Name)" } }
}

function Assert-Empty($Object, [string]$Field, [string]$Label) {
    $property = $Object.PSObject.Properties[$Field]
    if ($null -eq $property -or $null -eq $property.Value -or @($property.Value).Count -ne 0) { throw "Missing/nonempty discrepancies: $Label/$Field" }
}

function Audit-PilotStages([string[]]$Stages, [string]$Label) {
    $output = Join-Path $pilotAuditDir "$Label.json"
    Invoke-PilotPython $Label (@($pilotAuditor, '--stages') + $Stages + @('--output', $output)) -CpuOnly
    $audit = Get-Content -Raw -LiteralPath $output | ConvertFrom-Json -Depth 100
    foreach ($stage in $Stages) {
        $entry = $audit.stages.PSObject.Properties[$stage].Value
        $expected = if ($stage -match 'smoke') { 'SMOKE_ONLY' } else { 'VALID' }
        if ($entry.decision.status -ne $expected) { throw "Invalid/incomplete raw audit: $stage" }
        Assert-TrueChecks $entry.pair_checks "$stage/pairs"
        Assert-Empty $entry 'stored_summary_decision_discrepancies' $stage
        Assert-Empty $entry 'combined_resource_discrepancies' $stage
        foreach ($arm in @('A','B','C')) {
            $record = $entry.arms.PSObject.Properties[$arm].Value
            if ($record.status -ne 'VALID') { throw "Invalid raw arm: $stage/$arm" }
            Assert-TrueChecks $record.receipt_checks "$stage/$arm"
            Assert-Empty $record 'discrepancies' "$stage/$arm"
        }
        $manifest = Get-Content -Raw -LiteralPath (Join-Path $pilotLab "results/$stage/run_manifest.json") | ConvertFrom-Json -Depth 100
        if ($manifest.status -ne 'COMPLETE' -or $manifest.canonical_checkpoint_unchanged -ne $true) { throw "Root stage is not complete/unchanged: $stage" }
    }
}

try {
    Set-Location -LiteralPath $pilotRepo
    $pilotLock = [IO.File]::Open((Join-Path $PSScriptRoot 'resumable_phase1_controller.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    Assert-NotCancelled
    New-Item -ItemType Directory -Path $pilotAuditDir | Out-Null
    $env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'
    $env:CUDA_VISIBLE_DEVICES = '0'; $env:CUDA_LAUNCH_BLOCKING = '1'
    $env:PYTORCH_ALLOC_CONF = $pilotPolicy; $env:PYTORCH_CUDA_ALLOC_CONF = $pilotPolicy
    try { [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) } catch { }
    [ordered]@{ CUDA_VISIBLE_DEVICES = '0'; CUDA_LAUNCH_BLOCKING = '1'; PYTORCH_ALLOC_CONF = $pilotPolicy; PYTORCH_CUDA_ALLOC_CONF = $pilotPolicy; python = $pilotPython; cwd = $pilotRepo } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $pilotAuditDir 'execution_environment.json') -Encoding utf8
    $stages = [System.Collections.Generic.List[string]]::new()
    foreach ($core in @('a11','a1')) {
        foreach ($smoke in @($true,$false)) {
            $stage = if ($smoke) { 'phase1_smoke' } else { 'phase1' }
            if ($core -eq 'a1') { $stage += '_a1' }
            $common = @($pilotRunner, '--core', $core, '--max-steps', '300')
            if ($smoke) { $common += '--smoke' }
            foreach ($arm in @('A','B','C')) {
                $armPath = Join-Path $pilotLab "results/$stage/arm_$arm"
                $manifestPath = Join-Path $armPath 'run_manifest.json'
                $complete = $false
                if (Test-Path -LiteralPath $manifestPath) {
                    $armManifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json -Depth 100
                    $complete = $armManifest.status -eq 'COMPLETE'
                }
                if ($complete) { Write-PilotStatus 'COMPLETED_ARM_PENDING_STAGE_REVALIDATION' @{ stage = $stage; arm = $arm }; continue }
                $free = Wait-PilotGpu "$stage/$arm"
                $arguments = $common + @('--arms', $arm)
                if ((Test-Path -LiteralPath $armPath) -and @(Get-ChildItem -LiteralPath $armPath -Force).Count -gt 0) { $arguments += '--resume' }
                Invoke-PilotPython "${stage}_$arm" $arguments
            }
            $null = Wait-PilotGpu "$stage/combine"
            Invoke-PilotPython "${stage}_combine" ($common + '--combine')
            Audit-PilotStages @($stage) "${stage}_audit"
            $stages.Add($stage)
            if ($StopAfterPrimarySmoke -and $core -eq 'a11' -and $smoke) {
                Write-PilotStatus 'SMOKE_COMPLETE_RESTORE_CHECK_PENDING' @{ stage = $stage; audit = (Join-Path $pilotAuditDir "${stage}_audit.json") }
                return
            }
        }
    }
    Audit-PilotStages @($stages) 'all_phase1_final_audit'
    $saved = Join-Path $pilotAuditDir 'saved_adapter_values.json'
    Invoke-PilotPython 'saved_adapter_values' (@((Join-Path $pilotLab 'results/audit_20260905/verify_saved_adapter_values.py'), '--require-complete', '--stages') + @($stages) + @('--output', $saved)) -CpuOnly
    $savedReport = Get-Content -Raw -LiteralPath $saved | ConvertFrom-Json -Depth 100
    if ($savedReport.status -ne 'PASS' -or $savedReport.canonical_checkpoint_unchanged -ne $true) { throw 'Saved-adapter verification failed' }
    Write-PilotStatus 'COMPLETE_DOCS_PENDING' @{ stages = @($stages); final_audit = (Join-Path $pilotAuditDir 'all_phase1_final_audit.json'); saved_adapter_audit = $saved }
}
catch {
    if ($null -ne $pilotLock) { Write-PilotStatus 'FAILED_REVIEW_REQUIRED' @{ error = $_.Exception.Message; action = 'Inspect real process liveness and committed checkpoints; never restart a live arm or infer a scientific null.' } }
    throw
}
finally { if ($null -ne $pilotLock) { $pilotLock.Dispose() } }

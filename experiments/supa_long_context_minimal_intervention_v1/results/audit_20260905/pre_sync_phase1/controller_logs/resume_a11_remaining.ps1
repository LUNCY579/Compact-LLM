#requires -Version 7.0
# Recovery after verified termination of the original controllers and model.
# Arm A is retained. Interrupted B was archived; B restarts its full fixed budget.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$pilotLab = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pilotRepo = (Resolve-Path -LiteralPath (Join-Path $pilotLab '../..')).Path
$pilotPython = (Resolve-Path -LiteralPath (Join-Path $pilotRepo '../Chat-Bubble/.venv/Scripts/python.exe')).Path
$pilotRunner = Join-Path $pilotLab 'scripts/phase1_run.py'
$pilotStatus = Join-Path $PSScriptRoot 'resume_a11_remaining_status.json'
$pilotCancel = Join-Path $PSScriptRoot 'resume_a11_remaining.cancel'
$pilotLock = $null
$pilotStart = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
$pilotPolicy = 'backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8'

function Write-PilotState([string]$State, [hashtable]$Detail = @{}) {
    [ordered]@{
        state = $State; updated_utc = [DateTime]::UtcNow.ToString('o')
        controller_pid = $PID; controller_start_time_utc = $pilotStart
        script_path = $PSCommandPath
        script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        detail = $Detail
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $pilotStatus -Encoding utf8
}

function Assert-PilotNotCancelled {
    if (Test-Path -LiteralPath $pilotCancel) { throw 'Recovery cancellation marker present; no further arm will start' }
}

try {
    Set-Location -LiteralPath $pilotRepo
    $pilotLock = [IO.File]::Open((Join-Path $PSScriptRoot 'resume_a11_remaining.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    Assert-PilotNotCancelled
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:PYTORCH_ALLOC_CONF = $pilotPolicy
    $env:PYTORCH_CUDA_ALLOC_CONF = $pilotPolicy
    $pilotEnvironmentReceipt = Join-Path $pilotLab 'results/phase1/allocator_environment_retry1.json'
    if (Test-Path -LiteralPath $pilotEnvironmentReceipt) { throw 'Existing recovery environment receipt; manual review required before retry' }
    [ordered]@{
        policy = $pilotPolicy; variables = @('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')
        applied_to = @('B','C'); CUDA_VISIBLE_DEVICES = '0'
        recovery = 'Retain completed A; repeat interrupted B from initialization, then execute C'
        fixed_budget = '300 optimizer attempts per trained arm; no model/data/hyperparameter changes'
        recorded_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $pilotEnvironmentReceipt -Encoding utf8
    foreach ($pilotArm in @('B','C')) {
        $pilotArmPath = Join-Path $pilotLab "results/phase1/arm_$pilotArm"
        if ((Test-Path -LiteralPath $pilotArmPath) -and @(Get-ChildItem -LiteralPath $pilotArmPath -Force).Count -gt 0) {
            throw "Occupied arm directory; refusing duplicate/restart: $pilotArmPath"
        }
        $pilotLog = Join-Path $PSScriptRoot "audit_a11_bounded_${pilotArm}_r1.log"
        if (Test-Path -LiteralPath $pilotLog) { throw "Existing recovery log; manual review required: $pilotLog" }
        while ($true) {
            Assert-PilotNotCancelled
            $pilotOtherJobs = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'phase1_run\.py' })
            $pilotGpu = @(& nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
            if ($LASTEXITCODE -ne 0 -or $pilotGpu.Count -ne 1) { throw 'GPU preflight failed' }
            $pilotFree = [int]$pilotGpu[0]
            if ($pilotOtherJobs.Count -eq 0 -and $pilotFree -ge 5500) { break }
            Write-PilotState 'WAITING_FOR_GPU_OR_EXISTING_MODEL' @{ next_arm = $pilotArm; free_mib = $pilotFree; existing_model_pids = @($pilotOtherJobs | ForEach-Object { $_.ProcessId }) }
            Start-Sleep -Seconds 60
        }
        Write-PilotState 'RUNNING_ARM' @{ arm = $pilotArm; free_mib_at_start = $pilotFree; log = $pilotLog }
        & $pilotPython -u $pilotRunner --core a11 --arms $pilotArm --max-steps 300 *> $pilotLog
        if ($LASTEXITCODE -ne 0) { throw "A1.1 recovery arm $pilotArm failed; inspect $pilotLog" }
    }
    Assert-PilotNotCancelled
    Write-PilotState 'COMBINING'
    & $pilotPython -u $pilotRunner --core a11 --max-steps 300 --combine
    if ($LASTEXITCODE -ne 0) { throw 'A1.1 recovery combine failed' }
    $pilotManifest = Get-Content -Raw -LiteralPath (Join-Path $pilotLab 'results/phase1/run_manifest.json') | ConvertFrom-Json -Depth 100
    if ($pilotManifest.status -ne 'COMPLETE' -or $pilotManifest.canonical_checkpoint_unchanged -ne $true) { throw 'A1.1 stage is not complete with an unchanged checkpoint' }
    Write-PilotState 'COMPLETE_A11_AUDIT_PENDING'
}
catch {
    if ($null -ne $pilotLock) { Write-PilotState 'FAILED_REVIEW_REQUIRED' @{ error = $_.Exception.Message } }
    throw
}
finally {
    if ($null -ne $pilotLock) { $pilotLock.Dispose() }
}

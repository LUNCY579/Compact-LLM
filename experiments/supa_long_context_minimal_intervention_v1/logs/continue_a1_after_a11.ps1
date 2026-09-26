#requires -Version 7.0
<#
Review-only preparation: this driver is not self-starting.
Launch only after review, supplying the observed predecessor's exact PID and
UTC process start time while it is still alive:
  pwsh -NoProfile -File <this-file> -A11ControllerPid <observed-pid> `
    -ExpectedStartTime '<observed-start-time-utc>'
The dedicated resume_a11_remaining.ps1 predecessor also requires its matching
resume_a11_remaining.launch.json identity and source-hash receipt.

The driver waits for a successful predecessor exit, validates raw A1.1 results,
then runs A1 smoke and bounded A/B/C. It never runs Phase 2 or composition,
changes source/configuration, deletes artifacts, or updates RESULT/HANDOFF.
Monitor logs/continuation_a1_driver_status.json. Its private audit output paths
avoid collisions with the root agent's default audit JSON files.
Create logs/continuation_a1.cancel to prevent any further execution. If a child
is already running, it finishes normally before the driver stops; no live job
is killed. Verify status PID/start time against the live process, not the file alone.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, [int]::MaxValue)]
    [int]$A11ControllerPid,

    [Parameter(Mandatory = $true)]
    [DateTimeOffset]$ExpectedStartTime
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$LabRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $LabRoot '../..')).Path
$PythonPath = (Resolve-Path -LiteralPath (Join-Path $RepositoryRoot '../Chat-Bubble/.venv/Scripts/python.exe')).Path
$RunnerPath = Join-Path $LabRoot 'scripts/phase1_run.py'
$RecoveryControllerPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'resume_a11_remaining.ps1'))
$RecoveryControllerLaunchPath = Join-Path $PSScriptRoot 'resume_a11_remaining.launch.json'
$AuditorPath = Join-Path $LabRoot 'scripts/audit_phase1_results.py'
$SavedAdapterVerifier = Join-Path $LabRoot 'results/audit_20260905/verify_saved_adapter_values.py'
$ContinuationPath = Join-Path $LabRoot 'results/audit_20260905/CONTINUATION.md'
$AllocatorPolicy = 'backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8'
$ExecutionId = [Guid]::NewGuid().ToString('N')
$PrivateAuditRoot = Join-Path $LabRoot "results/audit_20260905/continuation_$ExecutionId"
$MarkerPath = Join-Path $PSScriptRoot 'continuation_a1_driver_status.json'
$CancellationPath = Join-Path $PSScriptRoot 'continuation_a1.cancel'
$LockPath = Join-Path $PSScriptRoot 'continuation_a1_driver.lock'
$DriverStartTimeUtc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
$DriverLock = $null
$Controller = $null
$script:AuditOutputs = [System.Collections.Generic.List[string]]::new()
$script:CommandCounter = 0
$script:CurrentChild = $null
$script:DriverActive = $true

function Write-DriverStatus {
    param([string]$State, [hashtable]$Detail = @{})
    $payload = [ordered]@{
        schema_version = 1
        state = $State
        updated_utc = [DateTime]::UtcNow.ToString('o')
        driver_pid = $PID
        driver_start_time_utc = $DriverStartTimeUtc
        execution_id = $ExecutionId
        driver_active = $script:DriverActive
        audit_outputs_are_private = $true
        root_documents_pending = $true
        root_must_not_launch_duplicate_a1_work = $script:DriverActive
        predecessor_pid = $A11ControllerPid
        predecessor_start_time_utc = $ExpectedStartTime.UtcDateTime.ToString('o')
        repository_root = $RepositoryRoot
        python = $PythonPath
        allocator_policy = $AllocatorPolicy
        private_audit_directory = $PrivateAuditRoot
        cancellation_marker = $CancellationPath
        audit_outputs = @($script:AuditOutputs)
        phase2_execution_allowed = $false
        composition_execution_allowed = $false
        detail = $Detail
    }
    $temporaryMarker = "$MarkerPath.$ExecutionId.tmp"
    [IO.File]::WriteAllText($temporaryMarker, ($payload | ConvertTo-Json -Depth 15), [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temporaryMarker, $MarkerPath, $true)
}

function Assert-NotCancelled {
    if (Test-Path -LiteralPath $CancellationPath) {
        throw [OperationCanceledException]::new("Cancellation marker present: $CancellationPath; no further execution will start")
    }
}

function Read-RequiredJson {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required receipt missing: $Path" }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -Depth 100
}

function Assert-AllTrue {
    param($Checks, [string]$Label)
    $properties = @($Checks.PSObject.Properties)
    if ($properties.Count -eq 0) { throw "No checks present: $Label" }
    foreach ($property in $properties) {
        if ($property.Value -isnot [bool] -or $property.Value -ne $true) {
            throw "Check failed: $Label/$($property.Name)"
        }
    }
}

function Assert-EmptyDiscrepancies {
    param($Object, [string]$Field, [string]$Label)
    $property = $Object.PSObject.Properties[$Field]
    if ($null -eq $property -or $null -eq $property.Value -or @($property.Value).Count -ne 0) {
        throw "Missing or nonempty discrepancy array: $Label/$Field"
    }
}

function Get-HashMapSignature {
    param($Object)
    return (@($Object.PSObject.Properties | Sort-Object Name | ForEach-Object {
        $value = if ($_.Value -is [PSCustomObject]) { Get-HashMapSignature $_.Value } else { [string]$_.Value }
        "$($_.Name)=$value"
    }) -join "`n")
}

function Assert-StageReceipt {
    param([string]$Stage, [string]$Core)
    $stageDirectory = Join-Path $LabRoot "results/$Stage"
    $manifest = Read-RequiredJson (Join-Path $stageDirectory 'run_manifest.json')
    $summary = Read-RequiredJson (Join-Path $stageDirectory 'summary.json')
    if ($manifest.status -ne 'COMPLETE' -or $manifest.canonical_checkpoint_unchanged -ne $true) {
        throw "Stage is not COMPLETE with an unchanged canonical checkpoint: $Stage"
    }
    if ($summary.status -ne 'COMPLETE' -or (@($summary.completed_arms | Sort-Object) -join ',') -ne 'A,B,C' -or
        @($summary.missing_arms).Count -ne 0) { throw "Root summary is not fully COMPLETE: $Stage" }
    if ($manifest.config.core -ne $Core) { throw "Wrong core in $Stage manifest" }
    $coreFolder = if ($Core -eq 'a11') { 'Rineya-Supa-A1.1-1.1.0-devfreeze' } else { 'Rineya-Supa-A1-1.0.0-devfreeze' }
    $expectedCorePath = (Resolve-Path -LiteralPath (Join-Path $RepositoryRoot "../Chat-Bubble/Rineya_Published_Cores/$coreFolder")).Path
    if ([IO.Path]::GetFullPath($manifest.canonical_core) -ne $expectedCorePath) { throw "Canonical path mismatch: $Stage" }
    if (@($manifest.checkpoint_sha256_before.PSObject.Properties).Count -ne 10 -or
        (Get-HashMapSignature $manifest.checkpoint_sha256_before) -ne (Get-HashMapSignature $manifest.checkpoint_sha256_after)) {
        throw "Incomplete or unequal canonical hash maps: $Stage"
    }
    foreach ($property in $manifest.checkpoint_sha256_after.PSObject.Properties) {
        if ([IO.Path]::GetFileName($property.Name) -ne $property.Name) { throw 'Unexpected checkpoint filename in receipt' }
        $current = (Get-FileHash -LiteralPath (Join-Path $expectedCorePath $property.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($current -ne $property.Value) { throw "Current canonical hash mismatch: $Stage/$($property.Name)" }
    }
    foreach ($property in $manifest.source_sha256.PSObject.Properties) {
        $current = (Get-FileHash -LiteralPath (Join-Path $RepositoryRoot $property.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($current -ne $property.Value) { throw "Frozen source/config hash mismatch: $($property.Name)" }
    }
}

function Get-LivePhase1Jobs {
    return @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -and $_.CommandLine -match 'phase1_run\.py'
    } | Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine)
}

function Wait-ForIdleGpu {
    param([string]$Label)
    while ($true) {
        Assert-NotCancelled
        $liveJobs = @(Get-LivePhase1Jobs)
        $gpuLines = @(& nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>&1)
        if ($LASTEXITCODE -ne 0 -or $gpuLines.Count -ne 1) { throw "VRAM query failed before $Label : $gpuLines" }
        $freeMiB = 0
        if (-not [int]::TryParse(([string]$gpuLines[0]).Trim(), [ref]$freeMiB)) { throw 'Invalid nvidia-smi free-memory result' }
        if ($liveJobs.Count -eq 0 -and $freeMiB -ge 5500) { return $freeMiB }
        Write-DriverStatus 'WAITING_FOR_GPU_OR_EXISTING_JOB' @{
            next_execution = $Label; minimum_free_mib = 5500; free_mib = $freeMiB
            live_phase1_jobs = $liveJobs; next_check_seconds = 60
            action = 'Wait only; never restart or terminate an existing process'
        }
        Start-Sleep -Seconds 60
    }
}

function Invoke-RecordedPython {
    param([string]$Label, [string[]]$PythonArguments, [switch]$CpuOnly, [Nullable[int]]$FreeMiB = $null)
    $script:CommandCounter += 1
    Assert-NotCancelled
    $prefix = Join-Path $PSScriptRoot ('continuation_{0}_{1:D2}_{2}' -f $ExecutionId, $script:CommandCounter, $Label)
    $stdout = "$prefix.stdout.log"
    $stderr = "$prefix.stderr.log"
    $receiptPath = "$prefix.execution.json"
    $priorCudaVisibility = $env:CUDA_VISIBLE_DEVICES
    if ($CpuOnly) { $env:CUDA_VISIBLE_DEVICES = '-1' }
    $receipt = [ordered]@{
        status = 'STARTING'; label = $Label; start_utc = [DateTime]::UtcNow.ToString('o')
        executable = $PythonPath; arguments = @($PythonArguments); working_directory = $RepositoryRoot
        cpu_only = [bool]$CpuOnly; free_vram_mib_at_start = $FreeMiB
        environment = [ordered]@{
            PYTHONUTF8 = $env:PYTHONUTF8; PYTHONUNBUFFERED = $env:PYTHONUNBUFFERED
            PYTORCH_ALLOC_CONF = $env:PYTORCH_ALLOC_CONF; PYTORCH_CUDA_ALLOC_CONF = $env:PYTORCH_CUDA_ALLOC_CONF
            CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES; CUDA_LAUNCH_BLOCKING = $env:CUDA_LAUNCH_BLOCKING
        }
        stdout_log = $stdout; stderr_log = $stderr
        driver_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    try {
        $quoted = @($PythonArguments | ForEach-Object {
            # All values come from fixed driver paths/flags; embedded quotes or a
            # trailing slash are refused rather than interpreted by another shell.
            if ($_ -match '"' -or $_.EndsWith('\')) { throw 'Unsafe native argument shape' }
            '"' + $_ + '"'
        })
        $receipt | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $receiptPath -Encoding utf8
        $script:CurrentChild = Start-Process -FilePath $PythonPath -ArgumentList $quoted -WorkingDirectory $RepositoryRoot `
            -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    }
    finally { $env:CUDA_VISIBLE_DEVICES = $priorCudaVisibility }
    $receipt.status = 'RUNNING'
    $receipt['process_id'] = $script:CurrentChild.Id
    $receipt['process_start_time_utc'] = $script:CurrentChild.StartTime.ToUniversalTime().ToString('o')
    $receipt | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $receiptPath -Encoding utf8
    do {
        $executionState = if (Test-Path -LiteralPath $CancellationPath) { 'CANCEL_REQUESTED_WAITING_FOR_CHILD' } else { 'RUNNING_EXECUTION' }
        Write-DriverStatus $executionState @{
            label = $Label; process_id = $script:CurrentChild.Id
            process_start_time_utc = $receipt.process_start_time_utc
            stdout_log = $stdout; stderr_log = $stderr; execution_receipt = $receiptPath
            audit_writer_active = [bool]$CpuOnly; next_check_seconds = 60
        }
        $finished = $script:CurrentChild.WaitForExit(60000)
    } until ($finished)
    $script:CurrentChild.Refresh()
    $exitCode = $script:CurrentChild.ExitCode
    $receipt.status = if ($exitCode -eq 0) { 'COMPLETE' } else { 'FAILED' }
    $receipt['exit_code'] = $exitCode
    $receipt['end_utc'] = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $receiptPath -Encoding utf8
    $script:CurrentChild.Dispose()
    $script:CurrentChild = $null
    if ($exitCode -ne 0) { throw "$Label failed with exit $exitCode; inspect $stderr and $stdout" }
    Assert-NotCancelled
}

function Invoke-AndValidateRawAudit {
    param([string[]]$Stages, [string]$Label)
    $outputPath = Join-Path $PrivateAuditRoot "$Label.raw_numbers.json"
    Invoke-RecordedPython -Label $Label -CpuOnly -PythonArguments (@('-u', $AuditorPath, '--stages') + $Stages + @('--output', $outputPath))
    $script:AuditOutputs.Add($outputPath)
    $report = Read-RequiredJson $outputPath
    foreach ($stage in $Stages) {
        $entry = $report.stages.PSObject.Properties[$stage].Value
        $expectedStatus = if ($stage -match 'smoke') { 'SMOKE_ONLY' } else { 'VALID' }
        if ($entry.decision.status -ne $expectedStatus) { throw "Raw audit invalid/incomplete: $stage" }
        Assert-AllTrue $entry.pair_checks "$stage/pair_checks"
        Assert-EmptyDiscrepancies $entry 'stored_summary_decision_discrepancies' $stage
        Assert-EmptyDiscrepancies $entry 'combined_resource_discrepancies' $stage
        foreach ($arm in @('A', 'B', 'C')) {
            $armEntry = $entry.arms.PSObject.Properties[$arm].Value
            if ($armEntry.status -ne 'VALID') { throw "Raw arm audit invalid: $stage/$arm" }
            Assert-AllTrue $armEntry.receipt_checks "$stage/$arm/receipt_checks"
            Assert-EmptyDiscrepancies $armEntry 'discrepancies' "$stage/$arm"
        }
        $core = if ($stage.EndsWith('_a1')) { 'a1' } else { 'a11' }
        Assert-StageReceipt $stage $core
    }
}

function Invoke-A1Stage {
    param([string]$Stage, [switch]$Smoke)
    $stageDirectory = Join-Path $LabRoot "results/$Stage"
    foreach ($arm in @('A', 'B', 'C')) {
        $armDirectory = Join-Path $stageDirectory "arm_$arm"
        if ((Test-Path -LiteralPath $armDirectory) -and @(Get-ChildItem -LiteralPath $armDirectory -Force).Count -gt 0) {
            throw "Occupied A1 arm directory; refuse duplicate/restart: $armDirectory"
        }
    }
    New-Item -ItemType Directory -Path $stageDirectory -Force | Out-Null
    $environmentPath = Join-Path $stageDirectory 'allocator_environment.json'
    if (Test-Path -LiteralPath $environmentPath) { throw "Existing allocator receipt; stage requires manual review: $environmentPath" }
    [ordered]@{ policy = $AllocatorPolicy; variables = @('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')
        applied_to = @('A','B','C'); CUDA_LAUNCH_BLOCKING = '1'; execution_id = $ExecutionId; driver = $PSCommandPath
        scope = 'Fixed native allocator policy; no model/data/hyperparameter change'
        recorded_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $environmentPath -Encoding utf8
    foreach ($arm in @('A', 'B', 'C')) {
        $freeMiB = Wait-ForIdleGpu "$Stage/arm_$arm"
        $arguments = @('-u', $RunnerPath, '--core', 'a1', '--arms', $arm, '--max-steps', '300')
        if ($Smoke) { $arguments += '--smoke' }
        Invoke-RecordedPython -Label "${Stage}_$arm" -PythonArguments $arguments -FreeMiB $freeMiB
    }
    $freeMiB = Wait-ForIdleGpu "$Stage/combine"
    $arguments = @('-u', $RunnerPath, '--core', 'a1', '--max-steps', '300', '--combine')
    if ($Smoke) { $arguments += '--smoke' }
    Invoke-RecordedPython -Label "${Stage}_combine" -PythonArguments $arguments -FreeMiB $freeMiB
    Assert-StageReceipt $Stage 'a1'
    Invoke-AndValidateRawAudit -Stages @($Stage) -Label "${Stage}_audit"
}

try {
    Set-Location -LiteralPath $RepositoryRoot
    if ((Get-Content -LiteralPath $ContinuationPath -Raw) -notmatch [Regex]::Escape($AllocatorPolicy)) {
        throw 'CONTINUATION.md does not contain the expected fixed allocator policy'
    }
    if ($env:CUDA_VISIBLE_DEVICES -and $env:CUDA_VISIBLE_DEVICES -ne '0') {
        throw 'Driver requires default GPU visibility or CUDA_VISIBLE_DEVICES=0 to match GPU-0 preflight'
    }
    $DriverLock = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    Assert-NotCancelled
    New-Item -ItemType Directory -Path $PrivateAuditRoot | Out-Null
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:CUDA_LAUNCH_BLOCKING = '1'
    $env:PYTORCH_ALLOC_CONF = $AllocatorPolicy
    $env:PYTORCH_CUDA_ALLOC_CONF = $AllocatorPolicy
    $Controller = [Diagnostics.Process]::GetProcessById($A11ControllerPid)
    $null = $Controller.Handle  # Keep the observed process object, not a later PID reuse.
    if ($Controller.StartTime.ToUniversalTime().Ticks -ne $ExpectedStartTime.UtcDateTime.Ticks) {
        throw 'Predecessor PID start time mismatch; refuse to wait on a different process'
    }
    $controllerCim = Get-CimInstance Win32_Process -Filter "ProcessId = $A11ControllerPid"
    $controllerCommandLine = if ($null -eq $controllerCim) { '' } else { [string]$controllerCim.CommandLine }
    $isInlineController = $controllerCommandLine -match 'phase1_run\.py' -and $controllerCommandLine -match '--core a11'
    $escapedRecoveryPath = [Regex]::Escape($RecoveryControllerPath)
    $recoveryFilePattern = '(?:^|\s)-File\s+(?:"' + $escapedRecoveryPath + '"|' + $escapedRecoveryPath + ')(?=\s|$)'
    $isRecoveryController = $controllerCommandLine -match $recoveryFilePattern
    if ($Controller.ProcessName -notin @('pwsh','powershell') -or (-not $isInlineController -and -not $isRecoveryController)) {
        throw 'Predecessor is not the observed A1.1 PowerShell controller'
    }
    if ($isRecoveryController) {
        $recoveryLaunch = Read-RequiredJson $RecoveryControllerLaunchPath
        $recordedRecoveryPath = [string]$recoveryLaunch.script_path
        if (-not [IO.Path]::IsPathFullyQualified($recordedRecoveryPath) -or
            [IO.Path]::GetFullPath($recordedRecoveryPath) -ne $RecoveryControllerPath) {
            throw 'Recovery controller launch receipt has a different script path'
        }
        if ([long]$recoveryLaunch.controller_pid -ne $A11ControllerPid -or
            ([DateTimeOffset]$recoveryLaunch.controller_start_time_utc).UtcDateTime.Ticks -ne $ExpectedStartTime.UtcDateTime.Ticks) {
            throw 'Recovery controller launch receipt does not match the observed process identity'
        }
        $recordedRecoveryHash = [string]$recoveryLaunch.script_sha256
        $currentRecoveryHash = (Get-FileHash -LiteralPath $RecoveryControllerPath -Algorithm SHA256).Hash
        if ($recordedRecoveryHash -notmatch '\A[0-9a-fA-F]{64}\z' -or $currentRecoveryHash -ne $recordedRecoveryHash) {
            throw 'Recovery controller source does not match its launch receipt SHA256'
        }
    }
    while (-not $Controller.HasExited) {
        Assert-NotCancelled
        $identity = Get-Process -Id $A11ControllerPid -ErrorAction SilentlyContinue
        if ($null -ne $identity -and $identity.StartTime.ToUniversalTime().Ticks -ne $ExpectedStartTime.UtcDateTime.Ticks) {
            throw 'Predecessor PID was reused; refuse continuation'
        }
        Write-DriverStatus 'WAITING_FOR_A11_CONTROLLER' @{ next_check_seconds = 60; no_a1_process_started = $true }
        $null = $Controller.WaitForExit(60000)
    }
    $Controller.Refresh()
    if ($Controller.ExitCode -ne 0) { throw "A1.1 controller exited unsuccessfully: $($Controller.ExitCode)" }
    if (@(Get-LivePhase1Jobs).Count -gt 0) { throw 'A Phase 1 child remains live after predecessor exit; refuse duplicate work' }
    Assert-StageReceipt 'phase1' 'a11'
    Invoke-AndValidateRawAudit -Stages @('phase1_smoke','phase1') -Label 'a11_prerequisite_audit'
    foreach ($stage in @('phase1_smoke_a1','phase1_a1')) {
        foreach ($arm in @('A','B','C')) {
            $existingArm = Join-Path $LabRoot "results/$stage/arm_$arm"
            if ((Test-Path -LiteralPath $existingArm) -and @(Get-ChildItem -LiteralPath $existingArm -Force).Count -gt 0) {
                throw "Existing A1 work requires review before continuation: $existingArm"
            }
        }
    }
    Invoke-A1Stage -Stage 'phase1_smoke_a1' -Smoke
    Invoke-A1Stage -Stage 'phase1_a1'
    $allStages = @('phase1_smoke','phase1','phase1_smoke_a1','phase1_a1')
    Invoke-AndValidateRawAudit -Stages $allStages -Label 'all_phase1_final_audit'
    $savedOutput = Join-Path $PrivateAuditRoot 'saved_adapter_values.final.json'
    Invoke-RecordedPython -Label 'saved_adapter_values_final' -CpuOnly -PythonArguments (@('-u', $SavedAdapterVerifier, '--require-complete', '--stages') + $allStages + @('--output', $savedOutput))
    $script:AuditOutputs.Add($savedOutput)
    $savedReport = Read-RequiredJson $savedOutput
    if ($savedReport.status -ne 'PASS' -or $savedReport.canonical_checkpoint_unchanged -ne $true -or
        $savedReport.model_loads -ne 0 -or $savedReport.gpu_used -ne $false -or @($savedReport.arms).Count -ne 8 -or
        (Get-HashMapSignature $savedReport.checkpoint_sha256_before) -ne (Get-HashMapSignature $savedReport.checkpoint_sha256_after)) {
        throw 'Saved-adapter final verification failed or was incomplete'
    }
    foreach ($record in $savedReport.arms) {
        if ($record.status -ne 'PASS') { throw "Saved adapter failed: $($record.stage)/$($record.arm)" }
        Assert-AllTrue $record.checks "$($record.stage)/$($record.arm)/saved_adapter_checks"
    }
    $expectedSavedPairs = @($allStages | ForEach-Object { "$_/B"; "$_/C" } | Sort-Object)
    $actualSavedPairs = @($savedReport.arms | ForEach-Object { "$($_.stage)/$($_.arm)" } | Sort-Object)
    if (($expectedSavedPairs -join ',') -ne ($actualSavedPairs -join ',')) { throw 'Saved-adapter stage/arm coverage mismatch' }
    $script:DriverActive = $false
    Write-DriverStatus 'COMPLETE_DOCS_PENDING' @{
        completed_stages = $allStages; all_raw_audits_passed = $true; saved_adapters_passed = $true
        root_action = 'Review private audit files, update RESULT/HANDOFF, and interpret the preregistered gates; this driver never starts Phase 2 or composition'
    }
}
catch {
    $childStillRunning = $null -ne $script:CurrentChild -and -not $script:CurrentChild.HasExited
    $script:DriverActive = $childStillRunning
    if ($null -ne $DriverLock) {
        $failureState = if ($_.Exception -is [OperationCanceledException]) { 'CANCELLED_NO_FURTHER_EXECUTION' } else { 'FAILED_REVIEW_REQUIRED' }
        Write-DriverStatus $failureState @{
            error = $_.Exception.Message; child_may_still_be_running = $childStillRunning
            child_process_id = if ($null -ne $script:CurrentChild) { $script:CurrentChild.Id } else { $null }
            action = 'Do not retry or delete occupied directories; inspect existing process identity, receipts and logs'
        }
    }
    throw
}
finally {
    if ($null -ne $Controller) { $Controller.Dispose() }
    if ($null -ne $DriverLock) { $DriverLock.Dispose() }
}

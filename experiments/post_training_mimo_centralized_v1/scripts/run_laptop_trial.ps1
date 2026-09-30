[CmdletBinding()]
param(
    [ValidateSet('load_forward','full','backward_resume','unpadded_ce')]
    [string]$Stage = 'unpadded_ce',
    [ValidateRange(0,600)]
    [int]$WaitSeconds = 600,
    [ValidatePattern('^[A-Za-z0-9_-]{0,48}$')]
    [string]$RunName = '',
    [switch]$AuditOnly
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$taskPython = Join-Path (Split-Path $taskRoot -Parent) '.venv-mq001\Scripts\python.exe'
$taskNames = @{
    load_forward = 'E0_LAPTOP_STREAM_LOAD_FORWARD.json'
    full = 'E0_LAPTOP_STREAM_FULL.json'
    backward_resume = 'E0_LAPTOP_STREAM_BACKWARD_RESUME.json'
    unpadded_ce = 'E0_LAPTOP_STREAM_UNPADDED_CE.json'
}
$taskConfig = Join-Path $taskRoot ('experiments\post_training_mimo_centralized_v1\configs\' + $taskNames[$Stage])
$taskArtifactRoot = Join-Path $taskRoot 'artifacts\mimo_central'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw 'Pinned Python is missing.' }
if (-not $RunName) { $RunName = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8) }
$taskOutput = Join-Path $taskArtifactRoot ('headless_' + $Stage + '_' + $RunName)
$taskDispatch = $taskOutput + '_dispatch'
if ((Test-Path -LiteralPath $taskOutput) -or (Test-Path -LiteralPath $taskDispatch)) { throw 'Run directories must be new.' }
function Write-TaskReceipt([string]$Path, $Value) {
    $taskStream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    $taskWriter = [IO.StreamWriter]::new($taskStream, [Text.UTF8Encoding]::new($false))
    try { $taskWriter.WriteLine(($Value | ConvertTo-Json -Depth 12)) }
    finally { $taskWriter.Dispose() }
}
function Get-TaskHash([string]$Path) {
    $taskHashStream = [IO.File]::OpenRead($Path)
    $taskAlgorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($taskAlgorithm.ComputeHash($taskHashStream)).Replace('-','').ToLowerInvariant() }
    finally { $taskHashStream.Dispose(); $taskAlgorithm.Dispose() }
}
Push-Location -LiteralPath $taskRoot
$taskNativeInvoked = $false
$taskStarted = [DateTime]::UtcNow
$taskSamples = [Collections.Generic.List[object]]::new()
try {
    $taskHead = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Git source identity unavailable.' }
    $taskBranch = (& git branch --show-current).Trim()
    if ($taskBranch -ne 'research/post-training-mimo-centralized-v1') { throw 'Wrong experimental branch.' }
    $taskDirty = @(& git status --porcelain --untracked-files=all)
    if (($taskDirty.Count -gt 0) -and (-not $AuditOnly)) { throw 'Commit source before a native trial.' }
    $taskValidation = 'import json,sys; from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import validate_config; validate_config(json.load(open(sys.argv[1])))'
    & $taskPython -B -c $taskValidation $taskConfig
    if ($LASTEXITCODE -ne 0) { throw 'Registered config validation failed.' }
    $taskCfg = Get-Content -LiteralPath $taskConfig -Raw | ConvertFrom-Json
    $taskConfigHash = Get-TaskHash $taskConfig
    # Budget the entire worker plus system reserve and 64 MiB for its supervisor.
    $taskRequired = $taskCfg.worker_free_gib + $taskCfg.worker_rss_limit_gib + 0.0625
    New-Item -ItemType Directory -Path $taskDispatch | Out-Null
    $taskArgs = @('-B','-m','experiments.post_training_mimo_centralized_v1.scripts.supervise_native',
                  '--config',$taskConfig,'--output',$taskOutput)
    $taskPlan = @{
        status = 'HEADLESS_CONTROLLER_CONFIGURED'; started_at_utc = $taskStarted.ToString('o')
        source_sha = $taskHead; source_dirty = ($taskDirty.Count -gt 0)
        stage = $Stage; config_path = $taskConfig; config_sha256 = $taskConfigHash
        python = $taskPython; arguments = $taskArgs; output = $taskOutput
        resource_profile = $taskCfg.resource_profile; required_before_launch_GiB = $taskRequired
        wait_seconds_ceiling = $WaitSeconds; worker_seconds_ceiling = $taskCfg.wall_seconds
        system_reserve_GiB = $taskCfg.worker_free_gib; worker_RSS_ceiling_GiB = $taskCfg.worker_rss_limit_gib
        e1_authorized = $false; audit_only = [bool]$AuditOnly; controller_pid = $PID
    }
    Write-TaskReceipt (Join-Path $taskDispatch 'PLAN.json') $taskPlan
    if ($AuditOnly) {
        Write-Output ('Audit only: ' + $taskDispatch + '; no native worker launched.')
        return
    }
    Write-Output ('Waiting for at least ' + $taskRequired + ' GiB available RAM. Output: ' + $taskOutput)
    Write-Output 'This is a bounded mechanics trial. Run this in a regular PowerShell window outside the desktop app.'
    $taskStable = 0
    while ($true) {
        $taskNow = [DateTime]::UtcNow
        $taskFree = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1048576
        $taskSamples.Add(@{ at_utc = $taskNow.ToString('o'); available_RAM_GiB = $taskFree })
        if ($taskFree -ge $taskRequired) { $taskStable++ } else { $taskStable = 0 }
        if ($taskStable -ge 2) { break }
        if (($taskNow - $taskStarted).TotalSeconds -ge $WaitSeconds) { throw 'RAM admission window expired; no native worker launched.' }
        Start-Sleep -Seconds 2
    }
    $taskCurrentHead = (& git rev-parse HEAD).Trim()
    $taskCurrentDirty = @(& git status --porcelain --untracked-files=all)
    if (($taskCurrentHead -ne $taskHead) -or ($taskCurrentDirty.Count -gt 0)) { throw 'Source changed while waiting; no native worker launched.' }
    if ((Get-TaskHash $taskConfig) -ne $taskConfigHash) { throw 'Config changed while waiting.' }
    Write-TaskReceipt (Join-Path $taskDispatch 'ADMISSION.json') @{
        status = 'CONTROLLER_RESOURCE_ADMITTED_ONLY'; observations = $taskSamples.ToArray()
        elapsed_wait_seconds = ([DateTime]::UtcNow - $taskStarted).TotalSeconds; e1_authorized = $false
    }
    $taskNativeInvoked = $true
    & $taskPython @taskArgs
    $taskExit = $LASTEXITCODE
    Write-TaskReceipt (Join-Path $taskDispatch 'COMPLETION.json') @{
        status = $(if ($taskExit -eq 0) { 'WORKER_RETURNED_SUCCESS_STATUS_REVIEW_REQUIRED' } else { 'WORKER_RETURNED_FAILURE' })
        native_supervisor_invoked = $true; supervisor_exit_code = $taskExit
        elapsed_controller_seconds = ([DateTime]::UtcNow - $taskStarted).TotalSeconds
        output = $taskOutput; e1_authorized = $false
    }
    if ($taskExit -ne 0) { Write-Output ('Trial stopped; inspect receipts under ' + $taskOutput); exit $taskExit }
    Write-Output ('Trial finished; scope and qualification still require receipt review: ' + $taskOutput)
}
catch {
    if (Test-Path -LiteralPath $taskDispatch) {
        Write-TaskReceipt (Join-Path $taskDispatch 'FAILURE_RECEIPT.json') @{
            status = 'HEADLESS_CONTROLLER_STOPPED'; error = $_.Exception.Message
            native_supervisor_invoked = $taskNativeInvoked; observations = $taskSamples.ToArray()
            elapsed_controller_seconds = ([DateTime]::UtcNow - $taskStarted).TotalSeconds; e1_authorized = $false
        }
    }
    throw
}
finally { Pop-Location }

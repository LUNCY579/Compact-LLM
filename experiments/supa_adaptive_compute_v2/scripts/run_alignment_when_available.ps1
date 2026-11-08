# Resource-only scheduling for the unchanged original CPU alignment caller.
[CmdletBinding()]
param(
    [ValidateRange(1,86400)][int]$MaxQueueSeconds = 3600,
    [ValidateRange(1,16)][int]$MaxPairs = 1
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$programmeRoot = Split-Path -Parent $PSScriptRoot
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $programmeRoot)
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$runId = 'alignment-cpu-v2-002'
$configPath = Join-Path $programmeRoot 'cross_scale/configs/alignment_v2.json'
$caller = Join-Path $programmeRoot 'cross_scale/scripts/run_alignment.py'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$queueId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss_fffffffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$queueDirectory = Join-Path $programmeRoot "cross_scale/evidence/00_execution_queue/$queueId"
New-Item -ItemType Directory -Path $queueDirectory -ErrorAction Stop | Out-Null
$started = [DateTime]::UtcNow
$receipt = [ordered]@{
    status='STARTED'; pid=$PID; start_utc=$started.ToString('o'); run_id=$runId
    script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    config_sha256=(Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
    policy='Original CPU matrix in fixed pair order; only pre-load RAM admission failures retry into new attempts; other failures stop for review'
    max_queue_seconds=$MaxQueueSeconds; max_pairs=$MaxPairs; completed_pairs=0
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $queueDirectory 'started.json') -Encoding utf8
Write-Output "QUEUE_DIR=$queueDirectory"
$queueExit = 2
try {
    Set-Location -LiteralPath $repositoryRoot
    while (([DateTime]::UtcNow-$started).TotalSeconds -lt $MaxQueueSeconds) {
        $pending = $null
        :choosePair foreach ($supaLayer in $config.supa_layers) {
            foreach ($donorLayer in $config.donor_layers) {
                $pairDirectory = Join-Path $programmeRoot "cross_scale/evidence/02_bridge_objectives/$runId/S${supaLayer}_D${donorLayer}"
                $attemptNumber = 0
                if (Test-Path -LiteralPath $pairDirectory) {
                    $attempts = @(Get-ChildItem -LiteralPath $pairDirectory -Directory | Sort-Object Name)
                    $completeCount = 0
                    foreach ($attempt in $attempts) {
                        if ($attempt.Name -notmatch '^attempt_(\d{3,})$') { throw 'Unrecognized pair attempt directory' }
                        $attemptNumber = [Math]::Max($attemptNumber,1+[int]$Matches[1])
                        $terminal = Join-Path $attempt.FullName 'run_manifest.json'
                        if (!(Test-Path -LiteralPath $terminal)) {
                            $terminal = Join-Path $attempt.FullName 'abandonment.json'
                            if (!(Test-Path -LiteralPath $terminal)) { throw "Unfinished attempt requires explicit review: $($attempt.FullName)" }
                        }
                        $prior = Get-Content -LiteralPath $terminal -Raw | ConvertFrom-Json
                        if ($prior.status -eq 'COMPLETE') { $completeCount += 1; continue }
                        if ($prior.status -eq 'ABANDONED' -and $prior.original_process_absent -eq $true) {
                            # Only the registered explicit recovery command may
                            # establish this terminal state; never create it here.
                            continue
                        }
                        $failurePath = Join-Path $attempt.FullName 'failure.json'
                        if (!(Test-Path -LiteralPath $failurePath)) { throw 'Failed attempt lacks failure receipt' }
                        $failure = Get-Content -LiteralPath $failurePath -Raw | ConvertFrom-Json
                        if (!($prior.status -eq 'FAILED' -and $prior.arm -eq 'pair' -and $prior.device -eq 'cpu' -and
                              $failure.type -eq 'MemoryError' -and $failure.error -match '^insufficient available physical RAM:' -and
                              !(Test-Path -LiteralPath (Join-Path $attempt.FullName 'metrics.json')))) {
                            throw "Pair failed outside pre-load admission: $($attempt.FullName)"
                        }
                    }
                    if ($completeCount -gt 1) { throw 'Multiple successful pair attempts require review' }
                    if ($completeCount -eq 1) { continue }
                }
                $pending = [pscustomobject]@{supa=$supaLayer; donor=$donorLayer; attempt=$attemptNumber; directory=$pairDirectory}
                break choosePair
            }
        }
        if ($null -eq $pending) { $receipt.status='MATRIX_PAIRS_COMPLETE'; $queueExit=0; break }
        $hostGiB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
        $leasePresent = Test-Path -LiteralPath (Join-Path $programmeRoot '.runtime.lock')
        [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); host_available_gib=$hostGiB; lease_present=$leasePresent; pending=$pending} |
            ConvertTo-Json -Depth 5 -Compress | Add-Content -LiteralPath (Join-Path $queueDirectory 'events.jsonl') -Encoding utf8
        if ($leasePresent -or $hostGiB -lt 4.0) {
            Write-Output ('WAIT host={0:N2}GiB lease={1}' -f $hostGiB,$leasePresent)
            Start-Sleep -Seconds 30
            continue
        }
        $attemptPath = Join-Path $pending.directory ('attempt_{0:D3}' -f $pending.attempt)
        $logPath = Join-Path $queueDirectory ('S{0}_D{1}_attempt_{2:D3}.log' -f $pending.supa,$pending.donor,$pending.attempt)
        Write-Output "RUN_PAIR=S$($pending.supa)_D$($pending.donor) ATTEMPT=$($pending.attempt)"
        & $researchPython $caller --config $configPath pair --run-id $runId --supa-layer $pending.supa --donor-layer $pending.donor --attempt $pending.attempt 2>&1 | Tee-Object -FilePath $logPath
        $childExit = $LASTEXITCODE
        if ($childExit -eq 0) {
            $receipt.completed_pairs += 1
            if ($receipt.completed_pairs -ge $MaxPairs) { $receipt.status='REQUESTED_PAIRS_COMPLETE'; $queueExit=0; break }
            continue
        }
        $finalPath = Join-Path $attemptPath 'run_manifest.json'
        $failurePath = Join-Path $attemptPath 'failure.json'
        if (!(Test-Path -LiteralPath $finalPath) -or !(Test-Path -LiteralPath $failurePath)) { throw 'Pair exited without complete terminal evidence' }
        $failedRun = Get-Content -LiteralPath $finalPath -Raw | ConvertFrom-Json
        $errorReceipt = Get-Content -LiteralPath $failurePath -Raw | ConvertFrom-Json
        if (!($failedRun.status -eq 'FAILED' -and $errorReceipt.type -eq 'MemoryError' -and
              $errorReceipt.error -match '^insufficient available physical RAM:' -and
              !(Test-Path -LiteralPath (Join-Path $attemptPath 'metrics.json')))) { throw "Pair failed outside admission: $attemptPath" }
        Start-Sleep -Seconds 30
    }
    if ($receipt.status -eq 'STARTED') { $receipt.status='QUEUE_BUDGET_EXHAUSTED' }
}
catch {
    $queueExit=1; $receipt.status='STOPPED_FOR_REVIEW'; $receipt.failure=$_.Exception.Message
    Write-Output "QUEUE_STOPPED: $($_.Exception.Message)"
}
finally {
    $receipt.end_utc=[DateTime]::UtcNow.ToString('o')
    $receipt.wall_seconds=([DateTime]::UtcNow-$started).TotalSeconds
    $receipt.exit_code=$queueExit
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $queueDirectory 'queue_manifest.json') -Encoding utf8
}
exit $queueExit

# Operational scheduling only; all science remains in committed Python callers.
# Execute exactly one ready stage, then return its result for review.
[CmdletBinding()]
param(
    [ValidateRange(1,86400)][int]$MaxQueueSeconds = 3600,
    [ValidateSet('protocol_gpu_v2_3.json','protocol_gpu_v2_4.json')]
    [string]$NativeConfigName = 'protocol_gpu_v2_3.json',
    [switch]$RetryNativeAdmission
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$programmeRoot = Split-Path -Parent $PSScriptRoot
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $programmeRoot)
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$nativeConfigPath = Join-Path $programmeRoot "native_armr/configs/$NativeConfigName"
$nativeConfig = Get-Content -LiteralPath $nativeConfigPath -Raw | ConvertFrom-Json
$queueId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss_fffffffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$queueDirectory = Join-Path $programmeRoot "00_audit/execution_queues/$queueId"
New-Item -ItemType Directory -Path $queueDirectory -ErrorAction Stop | Out-Null
$started = [DateTime]::UtcNow
$receipt = [ordered]@{
    status = 'STARTED'; pid = $PID; start_utc = $started.ToString('o')
    script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    policy = 'One stage only: prefer specified pending native GPU qualification; otherwise next original CPU bridge pair. No retries of failed science.'
    native_config_path = $nativeConfigPath
    native_config_sha256 = (Get-FileHash -LiteralPath $nativeConfigPath -Algorithm SHA256).Hash.ToLowerInvariant()
    max_queue_seconds = $MaxQueueSeconds
    retry_native_zero_artifact_admission = [bool]$RetryNativeAdmission
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $queueDirectory 'started.json') -Encoding utf8
Write-Output "QUEUE_DIR=$queueDirectory"
$stageExit = 2
try {
    Set-Location -LiteralPath $repositoryRoot
    while (([DateTime]::UtcNow - $started).TotalSeconds -lt $MaxQueueSeconds) {
        $systemMemory = Get-CimInstance Win32_OperatingSystem
        $freeGpu = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0
        if ($LASTEXITCODE -ne 0) { throw 'NVIDIA resource query failed' }
        $gpuMiB = [double]::Parse(($freeGpu | Select-Object -First 1).Trim(), [Globalization.CultureInfo]::InvariantCulture)
        $hostGiB = $systemMemory.FreePhysicalMemory / 1MB
        $leasePresent = Test-Path -LiteralPath (Join-Path $programmeRoot '.runtime.lock')
        [ordered]@{ at_utc=[DateTime]::UtcNow.ToString('o'); host_available_gib=$hostGiB; gpu_available_mib=$gpuMiB; lease_present=$leasePresent } |
            ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $queueDirectory 'resources.jsonl') -Encoding utf8
        $arguments = $null
        if (!$leasePresent -and $hostGiB -ge 4.0) {
            $qualificationReceipts = @(Get-ChildItem -Path (Join-Path $programmeRoot 'native_armr/evidence/00_runtime_qualification/*/run_manifest.json') |
                ForEach-Object {
                    $qualification = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                    if ($qualification.runtime_qualification -eq $nativeConfig.runtime_qualification) {
                        [pscustomobject]@{ receipt=$qualification; directory=$_.DirectoryName }
                    }
                })
            $qualificationPending = !$qualificationReceipts.Count
            if (!$qualificationPending -and $RetryNativeAdmission) {
                $admissionFailures = @($qualificationReceipts | Where-Object {
                    $_.receipt.status -eq 'FAILED' -and
                    $_.receipt.failure_receipt.type -eq 'MemoryError' -and
                    $_.receipt.failure_receipt.message -match '^insufficient (available physical RAM|free VRAM):' -and
                    !(Test-Path -LiteralPath (Join-Path $_.directory 'report.json')) -and
                    @(Get-ChildItem -LiteralPath $_.directory -Recurse -File -Filter '*.pt').Count -eq 0
                })
                $qualificationPending = $admissionFailures.Count -eq $qualificationReceipts.Count
            }
            # The qualifier initializes CUDA before its CPU reference. Reserve
            # staging headroom for that context; the scientific floor stays 3GiB.
            if ($qualificationPending -and $gpuMiB -ge 3584 -and $hostGiB -ge 4.25) {
                $receipt.stage = $nativeConfig.runtime_qualification
                $arguments = @((Join-Path $PSScriptRoot 'qualify_native_gpu.py'),'--config',$nativeConfigPath,'--manifest',(Join-Path $programmeRoot 'data/generated/v2/native_armr/manifest.json'))
            }
            elseif ($qualificationPending) {
                # The original 16-pair order is fixed; stop if an earlier attempt
                # exists without success rather than silently retrying it.
                :findPair foreach ($supaLayer in @(8,12,16,19)) {
                    foreach ($donorLayer in @(12,16,20,24)) {
                        $pairDirectory = Join-Path $programmeRoot "cross_scale/evidence/02_bridge_objectives/alignment-cpu-v2-002/S${supaLayer}_D${donorLayer}"
                        $nextAttempt = 0
                        if (Test-Path -LiteralPath $pairDirectory) {
                            $attempts = @(Get-ChildItem -LiteralPath $pairDirectory -Directory)
                            $completeCount = 0
                            foreach ($priorAttempt in $attempts) {
                                if ($priorAttempt.Name -notmatch '^attempt_(\d{3,})$') { throw 'Unknown pair attempt directory' }
                                $nextAttempt = [Math]::Max($nextAttempt,1+[int]$Matches[1])
                                $final = Join-Path $priorAttempt.FullName 'run_manifest.json'
                                if (!(Test-Path -LiteralPath $final)) { $final = Join-Path $priorAttempt.FullName 'abandonment.json' }
                                if (!(Test-Path -LiteralPath $final)) { throw 'Unfinished pair requires explicit review' }
                                $prior = Get-Content -LiteralPath $final -Raw | ConvertFrom-Json
                                if ($prior.status -eq 'COMPLETE') { $completeCount += 1; continue }
                                if ($prior.status -eq 'ABANDONED' -and $prior.original_process_absent -eq $true) { continue }
                                $priorFailurePath = Join-Path $priorAttempt.FullName 'failure.json'
                                if (!(Test-Path -LiteralPath $priorFailurePath)) { throw 'Pair lacks failure receipt' }
                                $priorFailure = Get-Content -LiteralPath $priorFailurePath -Raw | ConvertFrom-Json
                                if (!($prior.status -eq 'FAILED' -and $priorFailure.type -eq 'MemoryError' -and
                                      $priorFailure.error -match '^insufficient available physical RAM:' -and
                                      !(Test-Path -LiteralPath (Join-Path $priorAttempt.FullName 'metrics.json')))) {
                                    throw 'Pair failed outside pre-load admission; review required'
                                }
                            }
                            if ($completeCount -gt 1) { throw 'Duplicate complete pair attempts require review' }
                            if ($completeCount -eq 1) { continue }
                        }
                        $receipt.stage = "cross_scale_cpu_S${supaLayer}_D${donorLayer}"
                        $arguments = @((Join-Path $programmeRoot 'cross_scale/scripts/run_alignment.py'),'--config',(Join-Path $programmeRoot 'cross_scale/configs/alignment_v2.json'),'pair','--run-id','alignment-cpu-v2-002','--supa-layer',"$supaLayer",'--donor-layer',"$donorLayer",'--attempt',"$nextAttempt")
                        break findPair
                    }
                }
            }
            else { throw 'Native qualification has terminal evidence; review it before scheduling another stage' }
        }
        if ($null -ne $arguments) {
            $receipt.arguments = $arguments
            $receipt.launched_utc = [DateTime]::UtcNow.ToString('o')
            $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $queueDirectory 'launch.json') -Encoding utf8
            Write-Output "LAUNCH_STAGE=$($receipt.stage)"
            & $researchPython @arguments 2>&1 | Tee-Object -FilePath (Join-Path $queueDirectory 'stage.log')
            $stageExit = $LASTEXITCODE
            $receipt.status = if ($stageExit -eq 0) { 'STAGE_COMPLETE' } else { 'STOPPED_FOR_REVIEW' }
            break
        }
        Write-Output ('WAIT host={0:N2}GiB gpu={1:N0}MiB lease={2}' -f $hostGiB,$gpuMiB,$leasePresent)
        Start-Sleep -Seconds 30
    }
    if ($receipt.status -eq 'STARTED') { $receipt.status = 'QUEUE_BUDGET_EXHAUSTED' }
}
catch {
    $stageExit = 1
    $receipt.status = 'STOPPED_FOR_REVIEW'
    $receipt.failure = $_.Exception.Message
    Write-Output "QUEUE_STOPPED: $($_.Exception.Message)"
}
finally {
    $receipt.end_utc = [DateTime]::UtcNow.ToString('o')
    $receipt.exit_code = $stageExit
    $receipt.wall_seconds = ([DateTime]::UtcNow - $started).TotalSeconds
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $queueDirectory 'queue_manifest.json') -Encoding utf8
}
exit $stageExit

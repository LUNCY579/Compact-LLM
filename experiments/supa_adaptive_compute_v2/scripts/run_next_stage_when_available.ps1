# Operational scheduling only; all science remains in committed Python callers.
# Execute exactly one ready stage, then return its result for review.
[CmdletBinding()]
param(
    [ValidateRange(1,86400)][int]$MaxQueueSeconds = 3600,
    [switch]$RetryNativeAdmission
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$programmeRoot = Split-Path -Parent $PSScriptRoot
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $programmeRoot)
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$queueId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss_fffffffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$queueDirectory = Join-Path $programmeRoot "00_audit/execution_queues/$queueId"
New-Item -ItemType Directory -Path $queueDirectory -ErrorAction Stop | Out-Null
$started = [DateTime]::UtcNow
$receipt = [ordered]@{
    status = 'STARTED'; pid = $PID; start_utc = $started.ToString('o')
    script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    policy = 'One stage only: prefer pending native v2.3 GPU qualification; otherwise next original CPU bridge pair. No retries of failed science.'
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
                    if ($qualification.runtime_qualification -eq 'native_gpu_fp32_v2_3') {
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
                $receipt.stage = 'native_v2_3_runtime_qualification'
                $arguments = @((Join-Path $PSScriptRoot 'qualify_native_gpu.py'),'--config',(Join-Path $programmeRoot 'native_armr/configs/protocol_gpu_v2_3.json'),'--manifest',(Join-Path $programmeRoot 'data/generated/v2/native_armr/manifest.json'))
            }
            elseif ($qualificationPending) {
                # The original 16-pair order is fixed; stop if an earlier attempt
                # exists without success rather than silently retrying it.
                :findPair foreach ($supaLayer in @(8,12,16,19)) {
                    foreach ($donorLayer in @(12,16,20,24)) {
                        $pairDirectory = Join-Path $programmeRoot "cross_scale/evidence/02_bridge_objectives/alignment-cpu-v2-002/S${supaLayer}_D${donorLayer}"
                        if (Test-Path -LiteralPath $pairDirectory) {
                            $attempts = @(Get-ChildItem -LiteralPath $pairDirectory -Directory)
                            $complete = @($attempts | ForEach-Object {
                                $final = Join-Path $_.FullName 'run_manifest.json'
                                if (!(Test-Path -LiteralPath $final)) { throw 'Unfinished pair requires explicit review' }
                                Get-Content -LiteralPath $final -Raw | ConvertFrom-Json
                            } | Where-Object status -eq 'COMPLETE')
                            if ($complete.Count -ne 1 -or $attempts.Count -ne 1) { throw 'Existing pair attempts require explicit review' }
                            continue
                        }
                        $receipt.stage = "cross_scale_cpu_S${supaLayer}_D${donorLayer}"
                        $arguments = @((Join-Path $programmeRoot 'cross_scale/scripts/run_alignment.py'),'--config',(Join-Path $programmeRoot 'cross_scale/configs/alignment_v2.json'),'pair','--run-id','alignment-cpu-v2-002','--supa-layer',"$supaLayer",'--donor-layer',"$donorLayer",'--attempt','0')
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

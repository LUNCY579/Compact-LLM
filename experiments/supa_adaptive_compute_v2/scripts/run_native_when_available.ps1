# Resource-only scheduling around the unchanged registered scientific caller.
# This launcher never changes a candidate, model setting, dataset or scorer.
[CmdletBinding()]
param(
    [ValidateSet('architecture_dev','confirmatory_dev','confirmatory_diagnostics')]
    [string]$Role = 'architecture_dev',
    [ValidateSet('protocol_gpu_v2_2.json','protocol_gpu_v2_3.json')]
    [string]$ConfigName = 'protocol_gpu_v2_2.json',
    [ValidateSet('registered','control-first')]
    [string]$ArmOrder = 'registered',
    [ValidateRange(1,86400)][int]$MaxQueueSeconds = 7200,
    [switch]$InspectOnly
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$programmeRoot = Split-Path -Parent $PSScriptRoot
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $programmeRoot)
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$configPath = Join-Path $programmeRoot "native_armr/configs/$ConfigName"
$manifestPath = Join-Path $programmeRoot 'data/generated/v2/native_armr/manifest.json'
$caller = Join-Path $PSScriptRoot 'run_native_campaign.py'
$roleStage = if ($Role -eq 'architecture_dev') { '01_architecture_dev' } else { '02_confirmatory_run' }
$scientificEvidence = Join-Path $programmeRoot "native_armr/evidence/$roleStage"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$campaignArms = @('A0','A1','A3','A16','AM')
if ($ArmOrder -eq 'control-first') {
    if ($Role -ne 'architecture_dev' -or $ConfigName -ne 'protocol_gpu_v2_3.json') {
        throw 'Control-first scheduling is registered only for the v2.3 architecture stage'
    }
    $campaignArms = @('A0','AM','A1','A3','A16')
}

function Read-Resources {
    $systemMemory = Get-CimInstance Win32_OperatingSystem
    $freeGpu = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA resource query failed' }
    $gpuMiB = [double]::Parse(($freeGpu | Select-Object -First 1).Trim(), [Globalization.CultureInfo]::InvariantCulture)
    $hostGiB = $systemMemory.FreePhysicalMemory / 1MB
    $leasePresent = Test-Path -LiteralPath (Join-Path $programmeRoot '.runtime.lock')
    [pscustomobject]@{
        at_utc = [DateTime]::UtcNow.ToString('o')
        host_available_gib = $hostGiB
        gpu_available_mib = $gpuMiB
        programme_lease_present = $leasePresent
        # Reserve room for both the light scheduler and model-process imports.
        ready = (!$leasePresent -and $hostGiB -ge 2.5 -and $gpuMiB -ge 3584)
    }
}

if ($InspectOnly) {
    Read-Resources | ConvertTo-Json
    exit 0
}

$queueId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss_fffffffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$queueDirectory = Join-Path $programmeRoot "native_armr/evidence/00_execution_queue/$queueId"
New-Item -ItemType Directory -Path $queueDirectory -ErrorAction Stop | Out-Null
$events = Join-Path $queueDirectory 'events.jsonl'
$queueStarted = [DateTime]::UtcNow
$queueManifest = [ordered]@{
    status = 'STARTED'; pid = $PID; role = $Role
    config_path = $configPath
    arms = $campaignArms
    start_utc = $queueStarted.ToString('o')
    script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    config_sha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
    dataset_manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    policy = 'Resume exact registered campaign; only zero-example RAM/VRAM admission failures are retried'
    max_queue_seconds = $MaxQueueSeconds
}
$queueManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $queueDirectory 'started.json') -Encoding utf8
Write-Output "QUEUE_DIR=$queueDirectory"
$attempt = 0
$queueExit = 1
try {
    Set-Location -LiteralPath $repositoryRoot
    while (([DateTime]::UtcNow - $queueStarted).TotalSeconds -lt $MaxQueueSeconds) {
        $resources = Read-Resources
        $resources | ConvertTo-Json -Compress | Add-Content -LiteralPath $events -Encoding utf8
        if (!$resources.ready) {
            Write-Output ('WAIT resources: host={0:N2}GiB gpu={1:N0}MiB lease={2}' -f $resources.host_available_gib,$resources.gpu_available_mib,$resources.programme_lease_present)
            Start-Sleep -Seconds 30
            continue
        }
        $attempt += 1
        $launched = [DateTime]::UtcNow
        $log = Join-Path $queueDirectory ('campaign_{0:D3}.log' -f $attempt)
        Write-Output "RESUME role=$Role attempt=$attempt"
        & $researchPython $caller --config $configPath --manifest $manifestPath --role $Role --arms $campaignArms 2>&1 | Tee-Object -FilePath $log
        $childExit = $LASTEXITCODE
        if ($childExit -eq 0) {
            $queueManifest.status = 'CAMPAIGN_COMPLETE'
            $queueExit = 0
            break
        }
        $recent = @(Get-ChildItem -LiteralPath $scientificEvidence -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Get-Item -LiteralPath (Join-Path $_.FullName 'run_manifest.json') -ErrorAction SilentlyContinue } |
            Where-Object LastWriteTimeUtc -ge $launched | Sort-Object LastWriteTimeUtc -Descending)
        if (!$recent.Count) { throw "Campaign exited $childExit without terminal scientific evidence; manual review required" }
        $failure = Get-Content -LiteralPath $recent[0].FullName -Raw | ConvertFrom-Json
        $admissionOnly = ($failure.status -eq 'FAILED' -and $failure.device -eq $config.device -and
            $failure.role -eq $Role -and $failure.completed_examples -eq 0 -and
            [IO.Path]::GetFullPath($failure.config_path) -eq [IO.Path]::GetFullPath($configPath) -and
            $failure.failure_receipt.type -eq 'MemoryError' -and
            $failure.failure_receipt.message -match '^insufficient (available physical RAM|free VRAM):')
        if (!$admissionOnly) { throw "Campaign failed outside resource admission: $($recent[0].FullName)" }
        [ordered]@{ at_utc=[DateTime]::UtcNow.ToString('o'); event='ADMISSION_RETRY'; receipt=$recent[0].FullName; exit_code=$childExit } |
            ConvertTo-Json -Compress | Add-Content -LiteralPath $events -Encoding utf8
        Start-Sleep -Seconds 30
    }
    if ($queueManifest.status -eq 'STARTED') {
        $queueManifest.status = 'QUEUE_BUDGET_EXHAUSTED'
        $queueExit = 2
    }
}
catch {
    $queueManifest.status = 'STOPPED_FOR_REVIEW'
    $queueManifest.failure = $_.Exception.Message
    Write-Output "QUEUE_STOPPED: $($_.Exception.Message)"
}
finally {
    $queueManifest.end_utc = [DateTime]::UtcNow.ToString('o')
    $queueManifest.campaign_attempts = $attempt
    $queueManifest.wall_seconds = ([DateTime]::UtcNow - $queueStarted).TotalSeconds
    $queueManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $queueDirectory 'queue_manifest.json') -Encoding utf8
}
exit $queueExit

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$programmeRoot = 'C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2'
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$configPath = Join-Path $programmeRoot 'cross_scale/generation_stage/v1/configs/generation_screen_gpu_v1.json'
$caller = Join-Path $programmeRoot 'cross_scale/generation_stage/v1/scripts/qualify_runtime.py'
$started = [DateTime]::UtcNow
$receipt = [ordered]@{
    status = 'WAITING'; pid = $PID; start_utc = $started.ToString('o')
    purpose = 'One X0 CPU qualification phase after the current Native controller exits'
    native_controller_pid = 177132; native_controller_start_utc = '2026-09-14T04:48:53.3580016Z'
    config_path = $configPath; config_sha256 = (Get-FileHash -LiteralPath $configPath).Hash.ToLowerInvariant()
    registration_sha256 = 'dcfc67e0828a3b47da41980a88ce47813b4f8b63c184502f6898bbeaf7222fd3'
    minimum_host_gib = 4.25; max_wait_seconds = 10800
    policy = 'No model overlap, scientific changes, implicit retries, GPU phase or generation'
}
function Write-ExclusiveJson($name, $value) {
    $path = Join-Path $PSScriptRoot $name
    $stream = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes(($value | ConvertTo-Json -Depth 6))
        $stream.Write($bytes, 0, $bytes.Length)
    } finally { $stream.Dispose() }
}
Write-ExclusiveJson 'started.json' $receipt
try {
    while (([DateTime]::UtcNow - $started).TotalSeconds -lt $receipt.max_wait_seconds) {
        $native = Get-Process -Id $receipt.native_controller_pid -ErrorAction SilentlyContinue
        $nativeActive = ($null -ne $native -and $native.StartTime.ToUniversalTime().ToString('o') -eq $receipt.native_controller_start_utc)
        $memory = Get-CimInstance Win32_OperatingSystem
        $hostGiB = $memory.FreePhysicalMemory / 1MB
        $lease = Test-Path -LiteralPath (Join-Path $programmeRoot '.runtime.lock')
        [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); native_controller_live=$nativeActive; host_available_gib=$hostGiB; lease=$lease} |
            ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $PSScriptRoot 'events.jsonl') -Encoding utf8
        if (!$nativeActive -and !$lease -and $hostGiB -ge $receipt.minimum_host_gib) {
            if ((Get-FileHash -LiteralPath $configPath).Hash.ToLowerInvariant() -ne $receipt.config_sha256) { throw 'Frozen config changed' }
            $receipt.status = 'LAUNCHED'
            $receipt.launch_utc = [DateTime]::UtcNow.ToString('o')
            Write-ExclusiveJson 'launch.json' $receipt
            & $researchPython $caller --config $configPath phase --cell X0 --phase cpu --attempt 0 2>&1 |
                Tee-Object -FilePath (Join-Path $PSScriptRoot 'qualification.log')
            $receipt.exit_code = $LASTEXITCODE
            $receipt.status = if ($LASTEXITCODE -eq 0) { 'PHASE_RETURNED_REVIEW_REQUIRED' } else { 'STOPPED_FOR_REVIEW' }
            break
        }
        Start-Sleep -Seconds 30
    }
    if ($receipt.status -eq 'WAITING') { $receipt.status = 'QUEUE_BUDGET_EXHAUSTED' }
} catch {
    $receipt.status = 'STOPPED_FOR_REVIEW'
    $receipt.failure = $_.Exception.Message
} finally {
    $receipt.end_utc = [DateTime]::UtcNow.ToString('o')
    Write-ExclusiveJson 'queue_manifest.json' $receipt
    $receipt | ConvertTo-Json -Depth 6
}

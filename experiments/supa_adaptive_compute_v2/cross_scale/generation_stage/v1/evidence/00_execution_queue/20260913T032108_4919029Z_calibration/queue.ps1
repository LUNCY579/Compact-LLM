$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
$queueDirectory=$PSScriptRoot
$programmeRoot='C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2'
$researchPython='C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$exportManifest=Join-Path $programmeRoot 'cross_scale/generation_stage/v1/evidence/00_upstream_exports/selected_20260913_001/manifest.json'
$calibrationCaller=Join-Path $programmeRoot 'cross_scale/generation_stage/v1/scripts/calibrate_control.py'
$started=[DateTime]::UtcNow
$receipt=[ordered]@{status='WAITING';pid=$PID;start_utc=$started.ToString('o');policy='One unchanged B calibration attempt after host capacity; shared model lease remains enforced';min_host_gib=3.25;max_wait_seconds=1800;export_manifest=$exportManifest}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $queueDirectory 'started.json') -Encoding utf8
try {
    while (([DateTime]::UtcNow-$started).TotalSeconds -lt 1800) {
        $systemMemory=Get-CimInstance Win32_OperatingSystem
        $hostGiB=$systemMemory.FreePhysicalMemory/1MB
        $leasePresent=Test-Path -LiteralPath (Join-Path $programmeRoot '.runtime.lock')
        [ordered]@{utc=[DateTime]::UtcNow.ToString('o');available_gib=$hostGiB;lease=$leasePresent} | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $queueDirectory 'events.jsonl') -Encoding utf8
        if ($hostGiB -ge 3.25 -and !$leasePresent) {
            $receipt.status='LAUNCHED'
            $receipt.launch_utc=[DateTime]::UtcNow.ToString('o')
            $receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $queueDirectory 'launch.json') -Encoding utf8
            & $researchPython $calibrationCaller run --export-manifest $exportManifest 2>&1 | Tee-Object -FilePath (Join-Path $queueDirectory 'calibration.log')
            $receipt.exit_code=$LASTEXITCODE
            $receipt.status=if ($LASTEXITCODE -eq 0) {'COMPLETE'} else {'STOPPED_FOR_REVIEW'}
            break
        }
        Start-Sleep -Seconds 30
    }
    if ($receipt.status -eq 'WAITING') {$receipt.status='QUEUE_BUDGET_EXHAUSTED'}
} catch {
    $receipt.status='STOPPED_FOR_REVIEW'
    $receipt.failure=$_.Exception.Message
} finally {
    $receipt.end_utc=[DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $queueDirectory 'queue_manifest.json') -Encoding utf8
    $receipt | ConvertTo-Json
}

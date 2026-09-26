#requires -Version 7.0
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$HelperPath = Join-Path $PSScriptRoot 'process_exit_probe_helper.ps1'
$ReceiptPath = Join-Path $PSScriptRoot 'external_process_exit_verification.json'
$PowerShellPath = (Get-Process -Id $PID).Path
$Records = @()
foreach ($expectedExit in @(0, 7)) {
    $launcherObject = $null
    $Controller = $null
    try {
        $launcherObject = Start-Process -FilePath $PowerShellPath -WindowStyle Hidden -PassThru `
            -ArgumentList @('-NoProfile', '-File', ('"' + $HelperPath + '"'), '-RequestedExitCode', [string]$expectedExit)
        $observedPid = $launcherObject.Id
        $expectedStart = [DateTimeOffset]$launcherObject.StartTime.ToUniversalTime()
        # This is the live driver's external-process-object pattern verbatim.
        $Controller = [Diagnostics.Process]::GetProcessById($observedPid)
        $null = $Controller.Handle
        $sameStart = $Controller.StartTime.ToUniversalTime().Ticks -eq $expectedStart.UtcDateTime.Ticks
        if (-not $sameStart) { throw 'Helper PID/start time identity mismatch' }
        $externalStart = $Controller.StartTime.ToUniversalTime().ToString('o')
        $identityChecks = 0
        $stopwatch = [Diagnostics.Stopwatch]::StartNew()
        while (-not $Controller.HasExited) {
            $identity = Get-Process -Id $observedPid -ErrorAction SilentlyContinue
            if ($null -ne $identity -and $identity.StartTime.ToUniversalTime().Ticks -ne $expectedStart.UtcDateTime.Ticks) {
                throw 'Helper PID was reused'
            }
            $identityChecks += 1
            $null = $Controller.WaitForExit(60000)
        }
        $Controller.Refresh()
        $observedExit = $Controller.ExitCode
        $stopwatch.Stop()
        $Records += [ordered]@{
            expected_exit_code = $expectedExit
            observed_exit_code_from_external_object = $observedExit
            exit_code_matches = $observedExit -eq $expectedExit
            process_id = $observedPid
            expected_start_time_utc = $expectedStart.UtcDateTime.ToString('o')
            observed_start_time_utc = $externalStart
            start_ticks_match = $sameStart
            handle_retained_before_wait = $true
            external_process_object_used = $true
            wait_loop_identity_checks = $identityChecks
            wait_elapsed_milliseconds = $stopwatch.ElapsedMilliseconds
            has_exited_after_refresh = $Controller.HasExited
            hidden_window = $true
        }
    }
    finally {
        if ($null -ne $Controller) { $Controller.Dispose() }
        if ($null -ne $launcherObject) { $launcherObject.Dispose() }
    }
}
$passed = @($Records | Where-Object { -not $_.exit_code_matches -or -not $_.start_ticks_match -or -not $_.has_exited_after_refresh }).Count -eq 0
$receipt = [ordered]@{
    status = if ($passed) { 'PASS' } else { 'FAIL' }
    created_utc = [DateTime]::UtcNow.ToString('o')
    host_powershell = $PowerShellPath
    powershell_version = $PSVersionTable.PSVersion.ToString()
    scope = 'Two disposable hidden PowerShell helpers only; each sleeps briefly then exits 0 or 7. Driver and model processes untouched.'
    driver_executed = $false
    model_or_gpu_process_started = $false
    helper_sha256 = (Get-FileHash -LiteralPath $HelperPath -Algorithm SHA256).Hash.ToLowerInvariant()
    verification_script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    records = $Records
}
$receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
$receipt | ConvertTo-Json -Depth 10
if (-not $passed) { throw 'External process exit-code validation failed' }

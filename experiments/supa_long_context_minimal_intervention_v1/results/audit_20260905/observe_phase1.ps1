# Read-only, one-shot observer. latest.pt is queried for filesystem metadata ONLY.
param(
    [string]$LabRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [ValidateRange(0,20)][int]$TailLines = 5
)
$ErrorActionPreference = 'Stop'
$LabRoot = [IO.Path]::GetFullPath($LabRoot)
$notes = [Collections.Generic.List[string]]::new()
$jsonOptions = @{}
if ((Get-Command ConvertFrom-Json).Parameters.ContainsKey('DateKind')) { $jsonOptions.DateKind = 'String' }
function Read-ObservedJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try { return (Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json @jsonOptions) }
    catch { $notes.Add("JSON read failed for ${Path}: $($_.Exception.Message)"); return $null }
}
function Process-Start([int]$ProcessId) {
    try {
        $process = Get-Process -Id $ProcessId
        $started = $process.StartTime.ToUniversalTime()
        return [ordered]@{ pid=$ProcessId; start_utc=$started.ToString('o'); start_utc_ticks=[string]$started.Ticks }
    } catch { return $null }
}
$statusPath = Join-Path $LabRoot 'logs/resumable_phase1_status.json'
$status = Read-ObservedJson $statusPath
$controllerActual = $null
$expectedTicks = $null
if ($null -ne $status) {
    $controllerActual = Process-Start ([int]$status.controller_pid)
    try { $expectedTicks = [string]([DateTimeOffset]::Parse($status.controller_start_time_utc).UtcDateTime.Ticks) }
    catch { $notes.Add('Recorded controller UTC start could not be parsed.') }
}
$controllerVerified = $null -ne $controllerActual -and $null -ne $expectedTicks -and $controllerActual.start_utc_ticks -ceq $expectedTicks
$runnerPath = Join-Path $LabRoot 'scripts/phase1_run.py'
$modelProcesses = @()
try {
    $modelProcesses = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.Replace('/','\').IndexOf($runnerPath.Replace('/','\'), [StringComparison]::OrdinalIgnoreCase) -ge 0 } |
        ForEach-Object { $entry = Process-Start ([int]$_.ProcessId); if ($null -ne $entry) { $entry } } |
        Sort-Object { $_.pid })
} catch { $notes.Add("Python process inventory failed: $($_.Exception.Message)") }
$label = if ($status.detail.label) { [string]$status.detail.label } elseif ($status.detail.next) { [string]$status.detail.next } else { $null }
$stage = $null
$arm = $null
if ($label -match '^(phase1(?:_smoke)?(?:_a1)?)[_/]([ABC]|combine|audit)$') {
    $stage = $Matches[1]
    $labelAction = $Matches[2]
    if ($labelAction -match '^[ABC]$') { $arm = $labelAction }
} elseif ($status.detail.stage -match '^phase1(?:_smoke)?(?:_a1)?$') {
    $stage = [string]$status.detail.stage
    if ($status.detail.arm -match '^[ABC]$') { $arm = [string]$status.detail.arm }
}
$armPath = $segmentPath = $progressPath = $identityPath = $checkpointPath = $null
$identity = $latestAttempt = $latestValidation = $checkpointMetadata = $null
$eventCount = 0
$fragmentIgnored = $false
if ($stage -and $arm) {
    $armPath = Join-Path $LabRoot "results/$stage/arm_$arm"
    $identityPath = Join-Path $armPath 'identity.json'
    $identity = Read-ObservedJson $identityPath
    $segmentsPath = Join-Path $armPath 'segments'
    if (Test-Path -LiteralPath $segmentsPath -PathType Container) {
        $latest = Get-ChildItem -LiteralPath $segmentsPath -Directory | Sort-Object Name | Select-Object -Last 1
        if ($latest) { $segmentPath = $latest.FullName; $progressPath = Join-Path $segmentPath 'training_progress.jsonl' }
    }
    if ($progressPath -and (Test-Path -LiteralPath $progressPath -PathType Leaf)) {
        try {
            $raw = Get-Content -Raw -LiteralPath $progressPath
            $lines = @([regex]::Split([string]$raw, '\r?\n'))
            $limit = $lines.Count
            if ($raw -and -not $raw.EndsWith("`n")) { $limit--; $fragmentIgnored = $true }
            for ($index=0; $index -lt $limit; $index++) {
                if ([string]::IsNullOrWhiteSpace($lines[$index])) { continue }
                try { $event = $lines[$index] | ConvertFrom-Json }
                catch { $notes.Add("Malformed complete progress line $($index+1); only earlier intact records were used."); break }
                $eventCount++
                if ($event.event -eq 'optimizer_attempt') { $latestAttempt = $event }
                elseif ($event.event -eq 'validation') { $latestValidation = $event }
            }
        } catch { $notes.Add("Progress read failed: $($_.Exception.Message)") }
    }
    $checkpointPath = Join-Path $armPath 'checkpoints/latest.pt'
    try {
        if (Test-Path -LiteralPath $checkpointPath -PathType Leaf) {
            $metadata = Get-Item -LiteralPath $checkpointPath
            $checkpointMetadata = [ordered]@{ path=$metadata.FullName; exists=$true; bytes=$metadata.Length; last_write_utc=$metadata.LastWriteTimeUtc.ToString('o'); contents_read=$false }
        } else { $checkpointMetadata = [ordered]@{ path=$checkpointPath; exists=$false; contents_read=$false } }
    } catch { $notes.Add("Checkpoint metadata unavailable: $($_.Exception.Message)") }
}
$logPath = $status.detail.log
$logTail = @()
if ($logPath -and $TailLines -gt 0) {
    try {
        $logPath = [IO.Path]::GetFullPath([string]$logPath)
        if (-not $logPath.StartsWith($LabRoot.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Command log is outside the experiment directory.' }
        if (Test-Path -LiteralPath $logPath -PathType Leaf) { $logTail = @(Get-Content -LiteralPath $logPath -Tail $TailLines) }
    } catch { $notes.Add("Command log tail unavailable: $($_.Exception.Message)") }
}
[ordered]@{
    observed_utc=[DateTimeOffset]::UtcNow.ToString('o')
    status_path=$statusPath; recorded_state=$status.state; status_updated_utc=$status.updated_utc
    controller=[ordered]@{ recorded_pid=$status.controller_pid; expected_start_utc=$status.controller_start_time_utc; expected_start_utc_ticks=$expectedTicks; actual=$controllerActual; pid_and_start_verified=$controllerVerified }
    phase1_python_processes=$modelProcesses; phase1_python_process_count=$modelProcesses.Count
    label=$label; stage=$stage; arm=$arm; arm_path=$armPath; identity_path=$identityPath; segment_path=$segmentPath
    total_attempt_budget=$identity.config.max_steps; configured_validation_every=$identity.scientific_recipe.eval_every
    logged_progress=[ordered]@{ path=$progressPath; intact_event_count=$eventCount; trailing_fragment_ignored=$fragmentIgnored; attempt=$latestAttempt.attempt; applied=$latestAttempt.optimizer_updates_applied; skipped_scaler=$latestAttempt.steps_skipped_by_scaler_total; skipped_safety=$latestAttempt.steps_skipped_by_safety_check_total; examples_consumed=$latestAttempt.examples_consumed; most_recent_validation=$latestValidation; interpretation='Observed append-only segment log only; callbacks may lag committed progress. No checkpoint contents inspected.' }
    latest_checkpoint_metadata=$checkpointMetadata; command_log_path=$logPath; command_log_tail=$logTail
    notes=@($notes)
} | ConvertTo-Json -Depth 10 -Compress

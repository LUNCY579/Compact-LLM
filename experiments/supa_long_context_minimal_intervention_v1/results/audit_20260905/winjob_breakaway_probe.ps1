[CmdletBinding()]
param([ValidateSet('Probe', 'Child', 'Observe', 'Audit')][string]$Mode = 'Probe')

# Probe is the default. This file never launches model or controller scripts.
$ErrorActionPreference = 'Stop'
$probeAudit = $PSScriptRoot
$probeRoot = [IO.Path]::GetFullPath((Join-Path $probeAudit '..\..\..\..'))
$probeOutput = Join-Path $probeAudit 'winjob_powershell_probe_artifacts'

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
public static class PermittedJobLaunch {
  [StructLayout(LayoutKind.Sequential)] public struct BasicLimit {
    public long processTime, jobTime;
    public uint flags;
    public UIntPtr minWs, maxWs;
    public uint active;
    public UIntPtr affinity;
    public uint priority, scheduling;
  }
  [StructLayout(LayoutKind.Sequential)] public struct IoCounters {
    public ulong readOps, writeOps, otherOps, readBytes, writeBytes, otherBytes;
  }
  [StructLayout(LayoutKind.Sequential)] public struct ExtendedLimit {
    public BasicLimit basic; public IoCounters io;
    public UIntPtr processMemory, jobMemory, peakProcessMemory, peakJobMemory;
  }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] public struct StartupInfo {
    public uint cb; public string reserved, desktop, title;
    public uint x, y, xSize, ySize, xChars, yChars, fill, flags;
    public ushort show, reservedBytes;
    public IntPtr reservedPointer, stdin, stdout, stderr;
  }
  [StructLayout(LayoutKind.Sequential)] public struct ProcessInfo {
    public IntPtr process, thread; public uint pid, tid;
  }
  [StructLayout(LayoutKind.Sequential)] public struct FileTime { public uint low, high; }
  public class ProcessIdentity { public uint Pid; public string CreationFileTime100ns, CreationUtc; }
  public class JobState { public bool InAnyJob; public string FlagsHex; public uint Flags, ReturnedBytes; public bool BreakawayOk, SilentBreakawayOk, KillOnJobClose; }
  public class LaunchReceipt { public ProcessIdentity Identity; public JobState ParentJob; public string CreationFlagsHex; }
  [DllImport("kernel32.dll")] public static extern IntPtr GetCurrentProcess();
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool QueryInformationJobObject(IntPtr job, int kind, out ExtendedLimit result, uint size, out uint returned);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetProcessTimes(IntPtr process, out FileTime creation, out FileTime exit, out FileTime kernel, out FileTime user);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool CloseHandle(IntPtr handle);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GetExitCodeProcess(IntPtr handle, out uint code);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern bool CreateProcessW(string application, StringBuilder command, IntPtr processAttributes, IntPtr threadAttributes, bool inheritHandles, uint flags, IntPtr environment, string cwd, ref StartupInfo startup, out ProcessInfo process);
  public static void Check(bool ok) { if (!ok) throw new Win32Exception(Marshal.GetLastWin32Error()); }
  public static ProcessIdentity Identity(IntPtr process, uint pid) {
    FileTime c,e,k,u; Check(GetProcessTimes(process, out c, out e, out k, out u));
    ulong ticks = ((ulong)c.high << 32) | c.low;
    return new ProcessIdentity { Pid=pid, CreationFileTime100ns=ticks.ToString(), CreationUtc=DateTime.FromFileTimeUtc((long)ticks).ToString("o") };
  }
  public static bool InJob(IntPtr process) { bool assigned; Check(IsProcessInJob(process, IntPtr.Zero, out assigned)); return assigned; }
  public static JobState CurrentJob() {
    var result = new JobState { InAnyJob=InJob(GetCurrentProcess()) };
    if (!result.InAnyJob) return result;
    ExtendedLimit limits; uint returned; uint size=(uint)Marshal.SizeOf(typeof(ExtendedLimit));
    Check(QueryInformationJobObject(IntPtr.Zero, 9, out limits, size, out returned));
    if (returned != size) throw new InvalidOperationException("Unexpected Job Object structure length");
    uint flags=limits.basic.flags; result.Flags=flags; result.FlagsHex="0x"+flags.ToString("X8"); result.ReturnedBytes=returned;
    result.BreakawayOk=(flags & 0x0800)!=0; result.SilentBreakawayOk=(flags & 0x1000)!=0; result.KillOnJobClose=(flags & 0x2000)!=0;
    return result;
  }
  public static string Quote(string value) {
    if (value.Length>0 && value.IndexOfAny(new char[] {' ', '\t', '\n', '\v', '"'})<0) return value;
    var result=new StringBuilder("\""); int slashes=0;
    foreach (char c in value) {
      if (c=='\\') { slashes++; continue; }
      if (c=='"') result.Append('\\', slashes*2+1).Append('"');
      else result.Append('\\', slashes).Append(c);
      slashes=0;
    }
    return result.Append('\\', slashes*2).Append('"').ToString();
  }
  // Reusable launcher: checks explicit permission in the ACTUAL calling process.
  // No elevation, inherited handles, job changes, retries, or fallback routes.
  public static LaunchReceipt Start(string application, string[] arguments, string cwd) {
    JobState parent=CurrentJob();
    if (!parent.InAnyJob || !parent.BreakawayOk) throw new InvalidOperationException("Immediate BREAKAWAY_OK is required; launch aborted");
    var command=new StringBuilder(Quote(application));
    foreach (string argument in arguments) command.Append(' ').Append(Quote(argument));
    var startup=new StartupInfo { cb=(uint)Marshal.SizeOf(typeof(StartupInfo)) };
    ProcessInfo child;
    // CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW; child writes its own files.
    Check(CreateProcessW(application, command, IntPtr.Zero, IntPtr.Zero, false, 0x09000000, IntPtr.Zero, cwd, ref startup, out child));
    try { return new LaunchReceipt { Identity=Identity(child.process, child.pid), ParentJob=parent, CreationFlagsHex="0x09000000" }; }
    finally { CloseHandle(child.thread); CloseHandle(child.process); }
  }
}
'@

function Save-ProbeReceipt([string]$Name, $Data) {
    $target = Join-Path $probeOutput $Name
    $temporary = $target + '.tmp'
    $Data | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $target -Force
}
function Get-ProbeState {
    [ordered]@{ TimestampUtc = [DateTime]::UtcNow.ToString('o'); Identity = [PermittedJobLaunch]::Identity([PermittedJobLaunch]::GetCurrentProcess(), $PID); Job = [PermittedJobLaunch]::CurrentJob(); Executable = (Join-Path $PSHOME 'pwsh.exe'); Cwd = (Get-Location).Path }
}

try {
    switch ($Mode) {
        'Audit' { Get-ProbeState | ConvertTo-Json -Depth 8 }
        'Probe' {
            if ((Get-Location).Path -ne $probeRoot) { throw 'Run from repository root' }
            $parentState = Get-ProbeState
            if (-not $parentState.Job.BreakawayOk) { throw 'Immediate BREAKAWAY_OK is required; no launch attempted' }
            [void][IO.Directory]::CreateDirectory($probeOutput)
            $sentinel = [IO.File]::Open((Join-Path $probeOutput 'launch_once.json'), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            $sentinel.Dispose()
            Save-ProbeReceipt 'parent_preflight.json' $parentState
            $application = Join-Path $PSHOME 'pwsh.exe'
            $arguments = @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $PSCommandPath, '-Mode', 'Child')
            $launch = [PermittedJobLaunch]::Start($application, $arguments, $probeRoot)
            Save-ProbeReceipt 'launch_receipt.json' ([ordered]@{ TimestampUtc=[DateTime]::UtcNow.ToString('o'); Launch=$launch; Application=$application; Arguments=$arguments; InheritHandles=$false; ChildSelfLogs=$true })
            $deadline = [DateTime]::UtcNow.AddSeconds(4)
            while (-not (Test-Path -LiteralPath (Join-Path $probeOutput 'child_start.json')) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 50 }
            if (-not (Test-Path -LiteralPath (Join-Path $probeOutput 'child_start.json'))) { throw 'Child startup receipt missing; no relaunch' }
            [ordered]@{ ParentReturningUtc=[DateTime]::UtcNow.ToString('o'); Launch=$launch; Child=(Get-Content -LiteralPath (Join-Path $probeOutput 'child_start.json') -Raw | ConvertFrom-Json) } | ConvertTo-Json -Depth 8
        }
        'Child' {
            $state = Get-ProbeState
            Save-ProbeReceipt 'child_start.json' $state
            $elapsed = [Diagnostics.Stopwatch]::StartNew()
            for ($sequence=0; $sequence -le 20; $sequence++) {
                $wait = $sequence * 1000 - $elapsed.ElapsedMilliseconds
                if ($wait -gt 0) { Start-Sleep -Milliseconds $wait }
                $beat = [ordered]@{ Sequence=$sequence; TimestampUtc=[DateTime]::UtcNow.ToString('o'); Identity=$state.Identity; ElapsedSeconds=$elapsed.Elapsed.TotalSeconds }
                $beat | ConvertTo-Json -Compress -Depth 5 | Add-Content -LiteralPath (Join-Path $probeOutput 'heartbeats.jsonl') -Encoding utf8
                Save-ProbeReceipt 'latest_heartbeat.json' $beat
            }
            Save-ProbeReceipt 'child_completion.json' ([ordered]@{ Completed=$true; TimestampUtc=[DateTime]::UtcNow.ToString('o'); Identity=$state.Identity; HeartbeatCount=21; ElapsedSeconds=$elapsed.Elapsed.TotalSeconds; Job=[PermittedJobLaunch]::CurrentJob() })
        }
        'Observe' {
            $start = Get-Content -LiteralPath (Join-Path $probeOutput 'child_start.json') -Raw | ConvertFrom-Json
            $handle = [PermittedJobLaunch]::OpenProcess(0x00101000, $false, $start.Identity.Pid)
            [PermittedJobLaunch]::Check($handle -ne [IntPtr]::Zero)
            try {
                $identity = [PermittedJobLaunch]::Identity($handle, $start.Identity.Pid)
                if ($identity.CreationFileTime100ns -ne $start.Identity.CreationFileTime100ns) { throw 'PID identity mismatch' }
                $initialWait = [PermittedJobLaunch]::WaitForSingleObject($handle, 0)
                if ($initialWait -eq [uint32]::MaxValue) { [PermittedJobLaunch]::Check($false) }
                $observation = [ordered]@{ TimestampUtc=[DateTime]::UtcNow.ToString('o'); Context='Separate tool invocation after launcher returned'; Identity=$identity; AliveAfterToolReturn=($initialWait -eq 258); InAnyJob=[PermittedJobLaunch]::InJob($handle); LatestHeartbeat=(Get-Content -LiteralPath (Join-Path $probeOutput 'latest_heartbeat.json') -Raw | ConvertFrom-Json) }
                Save-ProbeReceipt 'observed_after_return.json' $observation
                $observation | ConvertTo-Json -Depth 8
                $finished = [PermittedJobLaunch]::WaitForSingleObject($handle, 25000)
                if ($finished -eq [uint32]::MaxValue) { [PermittedJobLaunch]::Check($false) }
                if ($finished -ne 0) { throw 'Finite probe did not finish within 25-second observation; no termination attempted' }
                [uint32]$exitCode = 0
                [PermittedJobLaunch]::Check([PermittedJobLaunch]::GetExitCodeProcess($handle, [ref]$exitCode))
                $final = [ordered]@{ TimestampUtc=[DateTime]::UtcNow.ToString('o'); Observation=$observation; ExitCode=$exitCode; Completion=(Get-Content -LiteralPath (Join-Path $probeOutput 'child_completion.json') -Raw | ConvertFrom-Json) }
                Save-ProbeReceipt 'final_receipt.json' $final
                $final | ConvertTo-Json -Depth 8
            } finally { [void][PermittedJobLaunch]::CloseHandle($handle) }
        }
    }
} catch {
    if (Test-Path -LiteralPath $probeOutput) { Save-ProbeReceipt ($Mode.ToLowerInvariant() + '_error.json') ([ordered]@{ TimestampUtc=[DateTime]::UtcNow.ToString('o'); Error=$_.Exception.ToString(); NoRetry=$true }) }
    throw
}

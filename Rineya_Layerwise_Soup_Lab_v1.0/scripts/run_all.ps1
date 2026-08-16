param(
    [Parameter(Mandatory=$true)][string]$Config,
    [string]$RunDir = "",
    [switch]$SkipSealed
)

$ErrorActionPreference = "Stop"
$arguments = @("run", "--config", $Config)
if ($RunDir) { $arguments += @("--run-dir", $RunDir) }
if ($SkipSealed) { $arguments += "--skip-sealed" }
rineya-soup @arguments

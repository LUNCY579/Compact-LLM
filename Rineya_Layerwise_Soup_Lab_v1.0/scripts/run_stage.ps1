param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$RunDir,
    [Parameter(Mandatory=$true)]
    [ValidateSet("preflight","endpoints","uniform","global","probability","layer_influence","layer_search","freeze","secondary_protocols","sealed","report")]
    [string]$Stage,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$arguments = @("stage", $Stage, "--config", $Config, "--run-dir", $RunDir)
if ($Force) { $arguments += "--force" }
rineya-soup @arguments

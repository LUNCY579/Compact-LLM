#requires -Version 7.0
[CmdletBinding()]
param([Parameter(Mandatory = $true)][int]$RequestedExitCode)
Start-Sleep -Milliseconds 750
exit $RequestedExitCode

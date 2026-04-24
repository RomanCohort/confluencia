param(
  [switch]$InstallDeps,
  [switch]$Clean,
  [string]$PythonExe
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent $root

$drugDir = Join-Path $workspace 'confluencia-2.0-drug'
$epiDir = Join-Path $workspace 'confluencia-2.0-epitope'

if (-not (Test-Path $drugDir)) { throw "Missing folder: $drugDir" }
if (-not (Test-Path $epiDir)) { throw "Missing folder: $epiDir" }

Write-Host '[1/2] Building confluencia-2.0-drug...'
$drugBuildArgs = @(
  '-ExecutionPolicy', 'Bypass',
  '-File', (Join-Path $drugDir 'build_full.ps1')
)
if ($InstallDeps) {
  $drugBuildArgs += '-InstallDeps'
}
if ($Clean) {
  $drugBuildArgs += '-Clean'
}
if ($PythonExe) {
  $drugBuildArgs += @('-PythonExe', $PythonExe)
}
& powershell @drugBuildArgs

Write-Host '[2/2] Building confluencia-2.0-epitope...'
$epiBuildArgs = @(
  '-ExecutionPolicy', 'Bypass',
  '-File', (Join-Path $epiDir 'build_full.ps1')
)
if ($InstallDeps) {
  $epiBuildArgs += '-InstallDeps'
}
if ($Clean) {
  $epiBuildArgs += '-Clean'
}
if ($PythonExe) {
  $epiBuildArgs += @('-PythonExe', $PythonExe)
}
& powershell @epiBuildArgs

Write-Host 'Both full builds completed.'

param(
  [string]$Version = 'full',
  [switch]$Build,
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

if ($Build) {
  Write-Host '[pre-step] Building both full packages before release...'
  & powershell -ExecutionPolicy Bypass -File (Join-Path $root 'build_confluencia2_full.ps1') -InstallDeps:$InstallDeps -Clean:$Clean -PythonExe $PythonExe
}

Write-Host '[1/2] Releasing confluencia-2.0-drug...'
& powershell -ExecutionPolicy Bypass -File (Join-Path $drugDir 'release_full.ps1') -Version $Version -Build:$false -InstallDeps:$false -PythonExe $PythonExe

Write-Host '[2/2] Releasing confluencia-2.0-epitope...'
& powershell -ExecutionPolicy Bypass -File (Join-Path $epiDir 'release_full.ps1') -Version $Version -Build:$false -InstallDeps:$false -PythonExe $PythonExe

Write-Host 'Both release packages completed.'
Write-Host "Drug zip folder: $drugDir\release"
Write-Host "Epitope zip folder: $epiDir\release"

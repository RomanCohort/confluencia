param(
  [string]$Version = 'full',
  [switch]$Build,
  [switch]$InstallDeps,
  [string]$PythonExe
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($Build) {
  $buildParams = @{}
  if ($InstallDeps) { $buildParams.InstallDeps = $true }
  if ($PythonExe) { $buildParams.PythonExe = $PythonExe }
  & .\build_full.ps1 @buildParams
}

$distDir = Join-Path $root 'dist\confluencia-2.0-drug'
if (-not (Test-Path $distDir)) {
  throw "Build output not found: $distDir"
}

$releaseDir = Join-Path $root 'release'
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
$zipName = "confluencia-2.0-drug-$Version-$stamp.zip"
$zipPath = Join-Path $releaseDir $zipName
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }

Compress-Archive -Path "$distDir\*" -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host "Release package: $zipPath"

param(
  [switch]$InstallDeps,
  [switch]$Clean,
  [string]$PythonExe
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not $PythonExe) {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    $PythonExe = $cmd.Source
  } else {
    throw 'Python not found. Please provide -PythonExe or add python to PATH.'
  }
}

if ($InstallDeps) {
  & $PythonExe -m pip install --upgrade pip
  & $PythonExe -m pip install -r requirements.txt
  & $PythonExe -m pip install pyinstaller
}

if ($Clean) {
  if (Test-Path build) { Remove-Item -Recurse -Force build }
  if (Test-Path dist) { Remove-Item -Recurse -Force dist }
}

& $PythonExe -m PyInstaller --noconfirm --clean confluencia-2.0-epitope.spec
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

Write-Host "Build done: $root\dist\confluencia-2.0-epitope"

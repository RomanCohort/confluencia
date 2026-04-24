param(
  [switch]$InstallDeps,
  [switch]$Clean,
  [string]$PythonExe
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$distRoot = Join-Path $root 'dist'
$distName = 'confluencia-2.0-drug'
$distTarget = Join-Path $distRoot $distName

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

if (Test-Path $distTarget) {
  $maxAttempts = 3
  $distPath = $distTarget
  # Try to stop any running process from this dist folder before cleanup.
  Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and ($_.Path -like "$distPath\*") } |
    Stop-Process -Force -ErrorAction SilentlyContinue
  for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    try {
      Get-ChildItem -LiteralPath $distPath -Recurse -Force -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Attributes = 'Normal' }
      Remove-Item -LiteralPath $distPath -Recurse -Force
      break
    } catch {
      if ($attempt -eq $maxAttempts) { $script:distLocked = $true }
      Start-Sleep -Seconds 2
    }
  }
}

$distPathArgs = @()
if ($distLocked) {
  $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $stagingRoot = Join-Path $distRoot "_staging_$timestamp"
  $distPathArgs = @('--distpath', $stagingRoot)
  Write-Host "Dist folder is locked. Using staging output: $stagingRoot"
}

& $PythonExe -m PyInstaller --noconfirm --clean @distPathArgs confluencia-2.0-drug.spec
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

if ($distLocked) {
  $stagingTarget = Join-Path $stagingRoot $distName
  try {
    if (Test-Path $distTarget) { Remove-Item -LiteralPath $distTarget -Recurse -Force }
    Move-Item -LiteralPath $stagingTarget -Destination $distTarget
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    Write-Host "Build done: $distTarget"
  } catch {
    Write-Host "Build done (staging): $stagingTarget"
    Write-Host "Note: Could not replace $distTarget because it is still locked."
  }
} else {
  Write-Host "Build done: $distTarget"
}

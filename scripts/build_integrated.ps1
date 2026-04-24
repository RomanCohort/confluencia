param(
  [ValidateSet('minimal','denoise','full')]
  [Alias('Profile')]
  [string]$BuildProfile = 'minimal',
  [switch]$OneFile,
  [switch]$Console,
  [switch]$InstallDeps,
  [switch]$SkipBuild,

  # build path / network convenience passthrough
  [string]$DriveLetter,
  [string]$DistPath,
  [string]$WorkPath,
  [string]$PyInstallerTempPath,
  [string]$PipIndexUrl,
  [string]$PipExtraIndexUrl,
  [string[]]$PipTrustedHost,
  [string]$PipFindLinks,
  [int]$PipRetries,
  [int]$PipTimeoutSec,
  [string]$PipCacheDir,
  [switch]$SkipPipUpgrade,
  [int]$MinFreeGB,
  [switch]$SkipDiskCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

# Avoid hardcoding non-ASCII paths here (Windows PowerShell 5.1 can mis-read UTF-8 script files).
# Locate the integrated project folder by finding its spec file.
$spec = Get-ChildItem -LiteralPath $root -Recurse -File -Filter 'IGEM_Integrated.spec' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $spec) {
  throw "Cannot locate IGEM_Integrated.spec under: $root"
}

$integratedDir = $spec.Directory.FullName
$script = Join-Path $integratedDir 'build_windows.ps1'
if (-not (Test-Path -LiteralPath $script)) {
  throw "Cannot locate build script: $script"
}

# Allow running in environments where scripts are blocked (process scope only).
try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force } catch {}

# Use a hashtable splat (more robust than an argv array; avoids argument shifting bugs).
$invokeParams = @{ BuildProfile = $BuildProfile }
if ($OneFile) { $invokeParams.OneFile = $true }
if ($Console) { $invokeParams.Console = $true }
if ($InstallDeps) { $invokeParams.InstallDeps = $true }
if ($SkipBuild) { $invokeParams.SkipBuild = $true }

if ($DriveLetter) { $invokeParams.DriveLetter = $DriveLetter }
if ($DistPath) { $invokeParams.DistPath = $DistPath }
if ($WorkPath) { $invokeParams.WorkPath = $WorkPath }
if ($PyInstallerTempPath) { $invokeParams.PyInstallerTempPath = $PyInstallerTempPath }

if ($PipIndexUrl) { $invokeParams.PipIndexUrl = $PipIndexUrl }
if ($PipExtraIndexUrl) { $invokeParams.PipExtraIndexUrl = $PipExtraIndexUrl }
if ($PipTrustedHost) { $invokeParams.PipTrustedHost = $PipTrustedHost }
if ($PipFindLinks) { $invokeParams.PipFindLinks = $PipFindLinks }
if ($PipRetries) { $invokeParams.PipRetries = $PipRetries }
if ($PipTimeoutSec) { $invokeParams.PipTimeoutSec = $PipTimeoutSec }
if ($PipCacheDir) { $invokeParams.PipCacheDir = $PipCacheDir }
if ($SkipPipUpgrade) { $invokeParams.SkipPipUpgrade = $true }

if ($MinFreeGB) { $invokeParams.MinFreeGB = $MinFreeGB }
if ($SkipDiskCheck) { $invokeParams.SkipDiskCheck = $true }

& $script @invokeParams

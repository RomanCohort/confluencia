param(
  [ValidateSet('minimal','denoise','full')]
  [Alias('Profile')]
  [string]$BuildProfile = 'minimal',
  [switch]$OneFile,
  [switch]$Rebuild,
  [string]$OutDir = 'release'
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
$script = Join-Path $integratedDir 'release_windows.ps1'
if (-not (Test-Path -LiteralPath $script)) {
  throw "Cannot locate release script: $script"
}

try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force } catch {}

$invokeParams = @{ BuildProfile = $BuildProfile; OutDir = $OutDir }
if ($OneFile) { $invokeParams.OneFile = $true }
if ($Rebuild) { $invokeParams.Rebuild = $true }

& $script @invokeParams

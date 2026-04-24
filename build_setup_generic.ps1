# Universal PyInstaller Runtime Setup Builder
# Builds a standalone exe that can be used with ANY PyInstaller app

$Host.UI.RawUI.WindowTitle = 'Runtime Setup Builder'

Write-Host ''
Write-Host '========================================' -ForegroundColor Cyan
Write-Host '  Universal Runtime Setup Builder' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan
Write-Host ''

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptFile = Join-Path $ScriptDir 'run_setup.py'
$OutputDir = Join-Path $ScriptDir 'RuntimeSetup_dist'
$SpecFile = Join-Path $ScriptDir 'run_setup.spec'

# Python check
Write-Host '[1/5] Checking Python...' -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
    Write-Host '  [ERROR] Python not found' -ForegroundColor Red
    Read-Host 'Enter to exit'
    exit 1
}
Write-Host "  $($python.Source)" -ForegroundColor Green

# PyInstaller check
Write-Host ''
Write-Host '[2/5] Checking PyInstaller...' -ForegroundColor Yellow
$pi = Get-Command pyinstaller -ErrorAction SilentlyContinue
if (-not $pi) {
    Write-Host '  Installing PyInstaller...' -ForegroundColor Yellow
    & python -m pip install pyinstaller --quiet
}
Write-Host '  OK' -ForegroundColor Green

# Clean
Write-Host ''
Write-Host '[3/5] Cleaning...' -ForegroundColor Yellow
if (Test-Path $OutputDir) { Remove-Item $OutputDir -Recurse -Force }
if (Test-Path $SpecFile) { Remove-Item $SpecFile -Force -ErrorAction SilentlyContinue }
if (Test-Path $ScriptDir\build) { Remove-Item $ScriptDir\build -Recurse -Force }
Write-Host '  Done' -ForegroundColor Green

# Spec
Write-Host ''
Write-Host '[4/5] Creating spec...' -ForegroundColor Yellow
$SpecContent = @"
# -*- mode: python ; coding: utf-8 -*-
a = Analysis(['run_setup.py'], pathex=['$($ScriptDir.Replace('\','/'))'], binaries=[], datas=[], hiddenimports=[], hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=['tkinter.test'], noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='RuntimeSetup', debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False, disable_windowed_traceback=False, argv_emulation=False, target_arch=None, codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name='RuntimeSetup')
"@
$SpecContent | Out-File -FilePath $SpecFile -Encoding UTF8
Write-Host '  Done' -ForegroundColor Green

# Build
Write-Host ''
Write-Host '[5/5] Building...' -ForegroundColor Yellow
Write-Host ''

Push-Location $ScriptDir
pyinstaller run_setup.spec --clean
$Result = $LASTEXITCODE
Pop-Location

if ($Result -ne 0) {
    Write-Host ''
    Write-Host '[ERROR] Build failed!' -ForegroundColor Red
    Read-Host 'Enter to exit'
    exit 1
}

# Verify
$ExePath = Join-Path $OutputDir 'RuntimeSetup.exe'
if (-not (Test-Path $ExePath)) {
    Write-Host '[ERROR] Output not found!' -ForegroundColor Red
    exit 1
}

$Size = (Get-Item $ExePath).Length / 1MB
Write-Host ''
Write-Host '========================================' -ForegroundColor Green
Write-Host '  Build Successful!' -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Green
Write-Host ''
Write-Host "  Output: $ExePath" -ForegroundColor White
Write-Host "  Size: $($Size.ToString('0.0')) MB" -ForegroundColor White
Write-Host ''
Write-Host '  Contents of RuntimeSetup_dist folder:' -ForegroundColor White
Write-Host '    RuntimeSetup.exe  - Run this on target computer' -ForegroundColor Gray
Write-Host ''
Write-Host '  This exe is GENERIC - works with ANY PyInstaller app!' -ForegroundColor Cyan
Write-Host ''

Start-Process explorer.exe -ArgumentList $OutputDir
Read-Host 'Enter to exit'
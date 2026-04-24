param(
  [ValidateSet('minimal','denoise','full')]
  [Alias('Profile')]
  [string]$BuildProfile = 'minimal'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$req = switch ($BuildProfile) {
  'full' { Join-Path $root 'requirements-shared-full.txt' }
  'denoise' { Join-Path $root 'requirements-shared-denoise.txt' }
  default { Join-Path $root 'requirements-shared-minimal.txt' }
}

if (-not (Test-Path $python)) {
  throw "未找到虚拟环境解释器：$python`n请先在工作区根目录创建 .venv（或把 .venv 放在根目录）。"
}
if (-not (Test-Path $req)) {
  throw "未找到依赖文件：$req"
}

& $python -m pip install -U pip setuptools wheel
& $python -m pip install -r $req

Write-Host "OK: 已在共用 .venv 中安装 $([System.IO.Path]::GetFileName($req))" -ForegroundColor Green

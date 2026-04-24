# 用途：提示用户粘贴 kaggle.json 内容并写入用户目录，然后调用下载脚本
# 运行示例（在项目根目录）：
# powershell -ExecutionPolicy Bypass -File .\scripts\setup_kaggle_and_download.ps1

$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
if (-not (Test-Path $kaggleDir)) { New-Item -ItemType Directory -Path $kaggleDir | Out-Null }
$kaggleFile = Join-Path $kaggleDir "kaggle.json"

Write-Host "请将你的 Kaggle API token (kaggle.json) 内容粘贴到下面，粘贴完成后单独一行输入 END，然后回车：" -ForegroundColor Cyan
$lines = @()
while ($true) {
    $line = Read-Host
    if ($line -eq 'END') { break }
    $lines += $line
}

$content = $lines -join "`n"
Set-Content -Path $kaggleFile -Value $content -Encoding UTF8
Write-Host "已写入 $kaggleFile" -ForegroundColor Green

# 设置权限（可选）
try {
    icacls $kaggleFile /inheritance:r | Out-Null
} catch { }

# 使用虚拟环境的 python 运行下载脚本
$venvPython = "D:/IGEM集成方案/.venv/Scripts/python.exe"
$downloadScript = "d:\\IGEM集成方案\\scripts\\download_kaggle_denseweight.py"
if (Test-Path $venvPython -and Test-Path $downloadScript) {
    Write-Host "开始调用虚拟环境 Python 下载数据..." -ForegroundColor Cyan
    & $venvPython $downloadScript
} else {
    Write-Host "未找到虚拟环境 Python 或下载脚本，请检查路径：" -ForegroundColor Yellow
    Write-Host "  $venvPython" -ForegroundColor Yellow
    Write-Host "  $downloadScript" -ForegroundColor Yellow
}

Write-Host "完成。如需手动检查 data 目录，请查看: D:\IGEM集成方案\新建文件夹\DLEPS-main\DLEPS-main\data" -ForegroundColor Green

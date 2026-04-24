param(
    [string]$OutDir = ".\logs\reproduce",
    [switch]$RunSmoke = $true
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Push-Location $root
try {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $metaPath = Join-Path $OutDir "drug_reproduce_$timestamp.txt"

    "timestamp=$timestamp" | Out-File -FilePath $metaPath -Encoding utf8
    "cwd=$root" | Out-File -FilePath $metaPath -Append -Encoding utf8
    "python_version=$((python --version) 2>&1)" | Out-File -FilePath $metaPath -Append -Encoding utf8
    "" | Out-File -FilePath $metaPath -Append -Encoding utf8
    "pip_freeze:" | Out-File -FilePath $metaPath -Append -Encoding utf8
    (python -m pip freeze) | Out-File -FilePath $metaPath -Append -Encoding utf8

    if ($RunSmoke) {
        python .\tests\smoke_test.py
    }

    Write-Host "Drug reproducibility pipeline finished. Log: $metaPath"
}
finally {
    Pop-Location
}

param(
    [string]$TargetDir = "",
    [string]$RepoUrl = "https://github.com/pineappleK/ED2Mol.git"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $TargetDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\external\ED2Mol"))
}

if (-not (Test-Path (Split-Path $TargetDir -Parent))) {
    New-Item -ItemType Directory -Path (Split-Path $TargetDir -Parent) -Force | Out-Null
}

if (-not (Test-Path $TargetDir)) {
    git clone $RepoUrl $TargetDir
} else {
    Write-Host "ED2Mol already exists at: $TargetDir"
}

Write-Host "Next steps:"
Write-Host "1) cd $TargetDir"
Write-Host "2) conda env create -f ed2mol_env.yml -n ed2mol"
Write-Host "3) conda activate ed2mol"
Write-Host "4) Download weights from ED2Mol release"
Write-Host "5) In Confluencia app, set ED2Mol repo/config path and python command"

param(
    [string]$PythonCmd = "",
    [string]$Ed2MolRepo = "",
    [string]$Ed2MolConfig = ""
)

$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($PythonCmd)) {
    $PythonCmd = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"))
}
if ([string]::IsNullOrWhiteSpace($Ed2MolRepo)) {
    $Ed2MolRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\external\ED2Mol"))
}
if ([string]::IsNullOrWhiteSpace($Ed2MolConfig)) {
    $Ed2MolConfig = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\external\ED2Mol\configs\hitopt.yml"))
}

function Write-Check {
    param(
        [string]$Name,
        [bool]$Ok,
        [string]$Detail
    )
    $tag = if ($Ok) { "[OK]" } else { "[FAIL]" }
    Write-Host "$tag $Name - $Detail"
}

Write-Host "=== Confluencia ED2Mol + Mamba Self-Check ==="

$pyOk = Test-Path -LiteralPath $PythonCmd
Write-Check "Python executable" $pyOk $PythonCmd

$repoOk = Test-Path -LiteralPath $Ed2MolRepo
Write-Check "ED2Mol repo dir" $repoOk $Ed2MolRepo

$genPath = Join-Path $Ed2MolRepo "Generate.py"
$genOk = Test-Path -LiteralPath $genPath
Write-Check "ED2Mol Generate.py" $genOk $genPath

$cfgOk = Test-Path -LiteralPath $Ed2MolConfig
Write-Check "ED2Mol config" $cfgOk $Ed2MolConfig

if ($pyOk) {
    try {
        $torchOut = & $PythonCmd -c "import torch; print(torch.__version__)" 2>&1
        Write-Check "torch import" ($LASTEXITCODE -eq 0) ($torchOut -join " ")
    }
    catch {
        Write-Check "torch import" $false $_.Exception.Message
    }

    try {
        $mambaOut = & $PythonCmd -c "import mamba_ssm; print('mamba_ssm ok')" 2>&1
        Write-Check "mamba_ssm import" ($LASTEXITCODE -eq 0) ($mambaOut -join " ")
    }
    catch {
        Write-Check "mamba_ssm import" $false $_.Exception.Message
    }
}
else {
    Write-Check "torch import" $false "Python executable missing"
    Write-Check "mamba_ssm import" $false "Python executable missing"
}

Write-Host "=== End Self-Check ==="

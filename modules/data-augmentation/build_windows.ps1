param(
  [switch]$OneFile,
  [switch]$Console
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py = "C:/Program Files/Python313/python.exe"

Write-Host "Using Python: $py"

# Ensure build tools are present
& $py -m pip install --upgrade pip | Out-Null
& $py -m pip install -r requirements.txt | Out-Null
& $py -m pip install pyinstaller | Out-Null

$exeName = "IGEM_DataAug_Denoise"

$args = @(
  "-m", "PyInstaller",
  "--name", $exeName,
  "--clean",
  "--noconfirm"
)

if ($Console) {
  $args += "--console"
} else {
  $args += "--noconsole"
}

if ($OneFile) {
  $args += "--onefile"
}

# Bundle the app entry + local modules
$args += @(
  "--collect-all", "streamlit",
  "--collect-all", "altair",
  "--collect-all", "matplotlib",
  "--collect-all", "pandas",
  "--collect-all", "openpyxl",
  "--add-data", "front.py;.",
  "--add-data", "backend.py;.",
  "--add-data", "README.md;.",
  "app_launcher.py"
)

Write-Host "Building..." ($args -join ' ')
& $py @args

Write-Host "Done. Output is under .\\dist\\$exeName\\ (or .\\dist\\$exeName.exe in onefile mode)."

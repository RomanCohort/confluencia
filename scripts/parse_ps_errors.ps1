param(
  [Parameter(Mandatory = $true)]
  [string]$Path
)

Write-Host ("Parsing: " + $Path) -ForegroundColor Cyan
try {
  $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $Path
  Write-Host ("SHA256:  " + $hash.Hash) -ForegroundColor DarkGray
} catch {
  Write-Host ("Get-FileHash failed: " + $_.Exception.Message) -ForegroundColor Yellow
}

try {
  $head = Get-Content -LiteralPath $Path -TotalCount 2
  Write-Host "--- head ---" -ForegroundColor DarkGray
  $head | ForEach-Object { Write-Host $_ }
} catch {}

$tokens = $null
$errors = $null

[System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors) | Out-Null

Write-Host "--- token dump (lines 200-235) ---" -ForegroundColor DarkGray
try {
  $tokens |
    Where-Object { $_.Extent.StartLineNumber -ge 200 -and $_.Extent.StartLineNumber -le 235 } |
    Select-Object -Property Kind, @{Name='Line';Expression={$_.Extent.StartLineNumber}}, @{Name='Col';Expression={$_.Extent.StartColumnNumber}}, Text |
    Format-Table -AutoSize | Out-String -Width 400 | Write-Host
} catch {}

Write-Host "--- token detail (lines 220-229) ---" -ForegroundColor DarkGray
try {
  $tokens |
    Where-Object { $_.Extent.StartLineNumber -ge 220 -and $_.Extent.StartLineNumber -le 229 } |
    Select-Object -First 200 |
    ForEach-Object {
      $extText = $_.Extent.Text -replace "`r", "\\r" -replace "`n", "\\n"
      Write-Host ("[{0}] {1}:{2}-{3}:{4}  {5}" -f $_.Kind, $_.Extent.StartLineNumber, $_.Extent.StartColumnNumber, $_.Extent.EndLineNumber, $_.Extent.EndColumnNumber, $extText)
    }
} catch {}

Write-Host "--- token dump (lines 520-532) ---" -ForegroundColor DarkGray
try {
  $tokens |
    Where-Object { $_.Extent.StartLineNumber -ge 520 -and $_.Extent.StartLineNumber -le 532 } |
    Select-Object -Property Kind, @{Name='Line';Expression={$_.Extent.StartLineNumber}}, @{Name='Col';Expression={$_.Extent.StartColumnNumber}}, Text |
    Format-Table -AutoSize | Out-String -Width 400 | Write-Host
} catch {}

if (-not $errors -or $errors.Count -eq 0) {
  Write-Host "No parse errors." -ForegroundColor Green
  exit 0
}

$errors | Select-Object -First 50 | ForEach-Object {
  Write-Host $_.Message -ForegroundColor Red
  Write-Host ("  at:  " + $_.Extent.Text)
  Write-Host ("  line:" + $_.Extent.StartLineNumber + " col:" + $_.Extent.StartColumnNumber)
  Write-Host ""
}

exit 1

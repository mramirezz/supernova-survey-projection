<#
  Abre 10 PowerShell y corre run_sn_list_multiband.py en paralelo.

  Uso:
    .\run_parallel_10.ps1
    ./run_parallel_10.ps1

  Nota:
    Si tu ExecutionPolicy bloquea scripts:
      powershell -ExecutionPolicy Bypass -File .\run_parallel_10.ps1
#>

$ErrorActionPreference = "Stop"

# ====== Ajusta aquí si quieres ======
$Survey = "ZTF"
$Seed = 42
$CondaEnv = "ALR_37"

# Rango a procesar (0-based, stop exclusivo)
$Start = 0
$Stop = $null   # si es $null, usa total filas del CSV

$MinDetections = 7
$MaxAttempts = 10

$Workers = 10
$CsvPath = Join-Path $PSScriptRoot "data\sn_list_to_project.csv"
# ===================================

if (-not (Test-Path $CsvPath)) {
  throw "No existe CSV: $CsvPath"
}

if ($Stop -eq $null) {
  # Cuenta filas (header no cuenta)
  $Stop = (Import-Csv -Path $CsvPath).Count
}

$Total = [int]($Stop - $Start)
if ($Total -le 0) {
  throw "Rango inválido: start=$Start stop=$Stop"
}

$Chunk = [int][math]::Ceiling($Total / [double]$Workers)

function New-WindowCommand([string]$title, [int]$start, [int]$stop) {
  $rootEsc = $PSScriptRoot.Replace("'", "''")
  $titleEsc = $title.Replace("'", "''")

  $cmd =
    "Set-Location -LiteralPath '$rootEsc'; " +
    "conda activate $CondaEnv; " +
    "`$env:PYTHONUNBUFFERED = '1'; " +
    "python run_sn_list_multiband.py --survey $Survey --seed $Seed --start $start --stop $stop --require-min-detections --min-detections $MinDetections --max-attempts $MaxAttempts; " +
    "Write-Host ''; Write-Host '[DONE] Press Enter to close...'; [void][System.Console]::ReadLine();"

  return $cmd
}

Write-Host "CSV rows total: $Stop"
Write-Host "Launching $Workers windows for range [$Start, $Stop) (total=$Total), chunk=$Chunk"
Write-Host ""

for ($i = 0; $i -lt $Workers; $i++) {
  $s = [int]($Start + $i * $Chunk)
  if ($s -ge $Stop) { break }
  $e = [int]([math]::Min($Stop, $s + $Chunk))

  $title = "SNList split $($i+1)/$Workers ($s-$e)"
  $cmd = New-WindowCommand -title $title -start $s -stop $e

  Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd
  ) -WorkingDirectory $PSScriptRoot | Out-Null
}

Write-Host ""
Write-Host "Cada ventana crea su propio batch_id (NO se pisan):"
Write-Host "  outputs\batch_runs\<batch_id>\"
Write-Host "  outputs\multiband_runs\${Survey}_<batch_id>\"
Write-Host ""

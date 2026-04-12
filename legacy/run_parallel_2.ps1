<#
  Ejecuta run_sn_list_multiband.py en 2 terminales PowerShell en paralelo.

  Uso:
    .\run_parallel_2.ps1
    ./run_parallel_2.ps1

  Nota:
    Si tu ExecutionPolicy bloquea scripts:
      powershell -ExecutionPolicy Bypass -File .\run_parallel_2.ps1

  Requisito:
    - `conda` disponible en PATH (Anaconda/Miniconda)
    - Este script abre PowerShell con tu Profile (sin `-NoProfile`) para que funcione `conda activate`
#>

$ErrorActionPreference = "Stop"

# ====== Ajusta aquí si quieres ======
$Survey = "ZTF"
$Seed = 42
$CondaEnv = "ALR_37"
$Start = 250
$Stop = 300
$MinDetections = 7
$MaxAttempts = 10
# ===================================

$Mid = [int](($Start + $Stop) / 2)
$Start1 = $Start
$Stop1 = $Mid
$Start2 = $Mid
$Stop2 = $Stop

function New-WindowCommand([string]$title, [int]$start, [int]$stop) {
  $rootEsc = $PSScriptRoot.Replace("'", "''")
  $titleEsc = $title.Replace("'", "''")

  # OJO: escapamos $ para que se evalúe en la ventana hija (no aquí)
  $cmd =
    "Set-Location -LiteralPath '$rootEsc'; " +
    "`$env:PYTHONUNBUFFERED = '1'; " +
    "conda activate $CondaEnv; " +
    "python run_sn_list_multiband.py --survey $Survey --seed $Seed --start $start --stop $stop --require-min-detections --min-detections $MinDetections --max-attempts $MaxAttempts; " +
    "Write-Host ''; Write-Host '[DONE] Press Enter to close...'; [void][System.Console]::ReadLine();"

  return $cmd
}

Write-Host "Launching 2 parallel PowerShell windows..."
Write-Host "  Split 1: --start $Start1 --stop $Stop1"
Write-Host "  Split 2: --start $Start2 --stop $Stop2"
Write-Host ""

$cmd1 = New-WindowCommand -title "SNList split 1 ($Start1-$Stop1)" -start $Start1 -stop $Stop1
$cmd2 = New-WindowCommand -title "SNList split 2 ($Start2-$Stop2)" -start $Start2 -stop $Stop2

Start-Process -FilePath "powershell.exe" -ArgumentList @(
  "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd1
) -WorkingDirectory $PSScriptRoot | Out-Null

Start-Process -FilePath "powershell.exe" -ArgumentList @(
  "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd2
) -WorkingDirectory $PSScriptRoot | Out-Null

Write-Host "Each split writes to its own batch folder (unique batch_id):"
Write-Host "  outputs\batch_runs\<batch_id>\"
Write-Host "  outputs\multiband_runs\${Survey}_<batch_id>\"
Write-Host ""

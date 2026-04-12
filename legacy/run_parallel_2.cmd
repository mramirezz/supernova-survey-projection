@echo off
setlocal

REM ============================================================
REM Ejecuta run_sn_list_multiband.py en 2 terminales en paralelo
REM - Abre 2 ventanas CMD
REM - Activa conda env ALR_37
REM - Corre 2 splits (start/limit)
REM ============================================================

cd /d "%~dp0"

REM --- Ajusta esto si quieres ---
set SURVEY=ZTF
set SEED=42
set MIN_DETECTIONS=7
set MAX_ATTEMPTS=2

REM Split #1: filas 0-1 (limit=2)
set START_1=0
set LIMIT_1=2

REM Split #2: filas 2-3 (limit=2)
set START_2=2
set LIMIT_2=2

echo Launching 2 parallel terminals...
echo   Split 1: --start %START_1% --limit %LIMIT_1%
echo   Split 2: --start %START_2% --limit %LIMIT_2%
echo.

start "SNList split 1 (start=%START_1% limit=%LIMIT_1%)" cmd /k ^
  "call conda activate ALR_37 && set PYTHONUNBUFFERED=1 && python run_sn_list_multiband.py --survey %SURVEY% --seed %SEED% --start %START_1% --limit %LIMIT_1% --require-min-detections --min-detections %MIN_DETECTIONS% --max-attempts %MAX_ATTEMPTS%"

start "SNList split 2 (start=%START_2% limit=%LIMIT_2%)" cmd /k ^
  "call conda activate ALR_37 && set PYTHONUNBUFFERED=1 && python run_sn_list_multiband.py --survey %SURVEY% --seed %SEED% --start %START_2% --limit %LIMIT_2% --require-min-detections --min-detections %MIN_DETECTIONS% --max-attempts %MAX_ATTEMPTS%"

echo.
echo Each split writes to its own batch folder:
echo   outputs\batch_runs\YYYYMMDD_HHMMSS_xxxxxxxx\
echo   outputs\multiband_runs\%SURVEY%_YYYYMMDD_HHMMSS_xxxxxxxx\
echo.
echo You can close this window; the 2 terminals will keep running.
pause


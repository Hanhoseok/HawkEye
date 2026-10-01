@echo off
REM storeguard dashboard launcher.
REM Works regardless of the PowerShell execution policy (.cmd is not a PS script).
REM
REM   scripts\run_dashboard.cmd
REM   scripts\run_dashboard.cmd --loop
REM   scripts\run_dashboard.cmd --port 8080 --loop
REM
REM Camera source is read from D:\computervision\.env  (CAM1_SOURCE).
REM Press Ctrl+C to stop.
REM
REM NOTE: keep this file ASCII-only. cmd.exe reads it in the OEM codepage and
REM       UTF-8 Korean text here breaks parsing.

setlocal
cd /d "D:\computervision"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] venv not found. See README section 3.2 to install.
    exit /b 1
)
if not exist ".env" (
    echo [INFO] .env missing - copying .env.example. Set CAM1_SOURCE, then rerun.
    copy /y ".env.example" ".env" >nul
    exit /b 1
)

echo Open http://127.0.0.1:8000 in a browser.  Ctrl+C to stop.

if exist "runs\baseline_r2p1d\best.pt" (
    ".venv\Scripts\python.exe" -m storeguard.server.app --run baseline_r2p1d %*
) else (
    echo [WARN] runs\baseline_r2p1d\best.pt not found - starting with MODEL_NOT_LOADED.
    ".venv\Scripts\python.exe" -m storeguard.server.app %*
)

endlocal

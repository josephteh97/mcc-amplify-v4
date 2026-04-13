@echo off
setlocal
:: =============================================================================
:: Run Revit Model Builder Add-in
:: =============================================================================
:: Starts Revit 2023 (if not already running) and waits for the add-in HTTP
:: server to become ready on port 5000.
::
:: Prerequisites (one-time setup):
::   Run  revit_server\RevitAddin\build_and_deploy.bat  as Administrator first.
::
:: Usage:
::   Double-click this file  OR  run from any directory — no Admin needed.
:: =============================================================================

set "REVIT_EXE=C:\Program Files\Autodesk\Revit 2023\Revit.exe"
set "HEALTH_URL=http://localhost:5000/health"
set /a MAX_WAIT_S=120
set /a POLL_S=5

echo.
echo  ╔══════════════════════════════════════╗
echo  ║   Revit Model Builder — Start        ║
echo  ║   Add-in HTTP server on port 5000    ║
echo  ╚══════════════════════════════════════╝
echo.

:: ── Check Revit executable exists ─────────────────────────────────────────────
if not exist "%REVIT_EXE%" (
    echo  ERROR: Revit 2023 not found at:
    echo         %REVIT_EXE%
    echo.
    echo  Edit REVIT_EXE at the top of this script to match your install path.
    pause & exit /b 1
)

:: ── Start Revit if not already running ────────────────────────────────────────
tasklist /FI "IMAGENAME eq Revit.exe" 2>nul | find /I "Revit.exe" >nul
if %errorlevel% equ 0 (
    echo  [1/2] Revit is already running — skipping launch.
) else (
    echo  [1/2] Launching Revit 2023...
    start "" "%REVIT_EXE%"
    echo         Waiting for Revit to load...
)

:: ── Poll port 5000 until the add-in HTTP server is ready ─────────────────────
echo  [2/2] Waiting for add-in server on port 5000...
echo.
set /a elapsed=0

:POLL
timeout /t %POLL_S% /nobreak >nul
set /a elapsed+=%POLL_S%

:: Check if port 5000 is listening
netstat -ano | findstr LISTENING | findstr ":5000 " >nul
if %errorlevel% neq 0 (
    echo  [%elapsed%/%MAX_WAIT_S%s] Port 5000 not ready yet...
    if %elapsed% lss %MAX_WAIT_S% goto POLL
    echo.
    echo  TIMEOUT: Add-in did not start within %MAX_WAIT_S% seconds.
    echo.
    echo  Possible causes:
    echo    - build_and_deploy.bat was not run (DLL not in Addins folder)
    echo    - Revit showed an "Always Load" security prompt — click it
    echo    - Check:  C:\RevitOutput\addin_startup.log
    echo.
    pause & exit /b 1
)

:: Do a quick HTTP health check
echo  Port 5000 is LISTENING. Checking /health endpoint...
powershell -NoProfile -Command ^
  "try { $r=(Invoke-WebRequest -Uri '%HEALTH_URL%' -UseBasicParsing -TimeoutSec 5).StatusCode; if($r -eq 200){exit 0}else{exit 1} } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo  ╔══════════════════════════════════════╗
    echo  ║  Add-in server is READY              ║
    echo  ║  http://localhost:5000/health  ✓     ║
    echo  ╚══════════════════════════════════════╝
    echo.
    echo  The Ubuntu backend can now send build requests to:
    echo    http://LT-HQ-277:5000/build-model
    echo.
) else (
    echo.
    echo  WARNING: Port is open but /health returned an error.
    echo  The add-in may still be initialising — wait a few seconds and retry.
    echo.
)

pause

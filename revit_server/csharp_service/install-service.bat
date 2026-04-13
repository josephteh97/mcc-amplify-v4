@echo off
REM Must run as Administrator

echo ============================================
echo Installing Revit API Service as Windows Service
echo ============================================

REM Check if running as admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Check if service exists
sc query RevitAPIService >nul 2>&1
if %errorlevel% equ 0 (
    echo Service already exists. Stopping and removing...
    sc stop RevitAPIService
    timeout /t 3 /nobreak >nul
    sc delete RevitAPIService
)

REM Get current directory
set "SERVICE_PATH=%~dp0bin\Release\net6.0\RevitService.exe"

REM Create service
echo Installing service...
sc create RevitAPIService ^
    binPath= "%SERVICE_PATH%" ^
    start= auto ^
    DisplayName= "Revit API Service"

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo Service installed successfully!
    echo ============================================
    echo.
    echo To start the service:
    echo   sc start RevitAPIService
    echo.
    echo To check status:
    echo   sc query RevitAPIService
    echo.
    echo To stop:
    echo   sc stop RevitAPIService
    echo.
) else (
    echo.
    echo ============================================
    echo Service installation FAILED!
    echo ============================================
)

pause

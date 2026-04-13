@echo off
echo ============================================
echo Starting Revit API Service
echo ============================================

REM Check if Revit is installed
if not exist "C:\Program Files\Autodesk\Revit 2023\Revit.exe" (
    if not exist "C:\Program Files\Autodesk\Revit 2022\Revit.exe" (
        echo ERROR: Revit 2022 or 2023 not found!
        echo Please install Revit first.
        pause
        exit /b 1
    )
)

REM Create output directory
if not exist "C:\RevitOutput" (
    echo Creating output directory: C:\RevitOutput
    mkdir "C:\RevitOutput"
)

REM Create logs directory
if not exist "logs" (
    mkdir "logs"
)

REM Check if built
if not exist "bin\Release\net6.0\RevitService.exe" (
    echo ERROR: Service not built yet!
    echo Please run build.bat first.
    pause
    exit /b 1
)

REM Run the service
echo Starting service on port 5000...
echo.
cd bin\Release\net6.0
RevitService.exe

pause

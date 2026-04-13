@echo off
:: =============================================================================
:: Build and deploy RevitModelBuilderAddin to Revit 2023
:: =============================================================================
:: Run this script on the Windows machine that has Revit 2023 installed.
:: Requires .NET SDK 4.8 or "Build Tools for Visual Studio 2022".
::
:: After successful deployment, restart Revit 2023 — the add-in loads
:: automatically and the HTTP server starts on port 5000.
:: =============================================================================

setlocal

set "ADDIN_DIR=%ProgramData%\Autodesk\Revit\Addins\2023"
set "OUTPUT_DIR=%~dp0bin\Release\net48"

echo.
echo === Revit Model Builder Add-in — Build and Deploy ===
echo.

:: ── Step 1: Build ─────────────────────────────────────────────────────────────
echo [1/3] Building project...
dotnet build "%~dp0RevitModelBuilder.csproj" -c Release ^
    || (echo BUILD FAILED && pause && exit /b 1)

:: ── Step 2: Create output directories ────────────────────────────────────────
echo [2/3] Creating directories...
if not exist "%ADDIN_DIR%" mkdir "%ADDIN_DIR%"
if not exist "C:\RevitOutput"  mkdir "C:\RevitOutput"

:: ── Step 3: Copy files to Revit's add-in folder ──────────────────────────────
echo [3/3] Deploying to %ADDIN_DIR%...

copy /Y "%OUTPUT_DIR%\RevitModelBuilderAddin.dll"  "%ADDIN_DIR%\" ^
    || (echo DLL copy failed && pause && exit /b 1)
copy /Y "%~dp0RevitModelBuilder.addin"             "%ADDIN_DIR%\" ^
    || (echo .addin copy failed && pause && exit /b 1)

:: Newtonsoft.Json.dll — copy from Revit's own folder so versions match
:: (only needed if the DLL is not already in Revit's bin directory)
if exist "%ADDIN_DIR%\Newtonsoft.Json.dll" goto :skip_json
if exist "%ProgramFiles%\Autodesk\Revit 2023\Newtonsoft.Json.dll" (
    copy /Y "%ProgramFiles%\Autodesk\Revit 2023\Newtonsoft.Json.dll" "%ADDIN_DIR%\"
)
:skip_json

echo.
echo ============================================================
echo  Deployment complete!
echo  Files in: %ADDIN_DIR%
echo.
echo  NEXT STEPS:
echo  1. Restart Revit 2023
echo  2. The HTTP server will start automatically on port 5000
echo  3. Test: open a browser and visit http://localhost:5000/health
echo ============================================================
echo.
pause

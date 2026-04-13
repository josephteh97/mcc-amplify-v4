# Revit API Service - C# Implementation

Windows service that exposes Revit API via HTTP for Ubuntu integration.

## Prerequisites

1. **Revit 2022 or 2023** with valid license
2. **.NET 6.0 SDK** - https://dotnet.microsoft.com/download/dotnet/6.0
3. **Administrator privileges** for service installation

## Quick Start

### Step 1: Update config.json

```json
{
  "apiSettings": {
    "apiKey": "YOUR-SECURE-KEY-HERE"
  }
}
```

**IMPORTANT:** This key must match `REVIT_SERVER_API_KEY` in Ubuntu .env file!

### Step 2: Build

```cmd
build.bat
```

### Step 3: Run (Development)

```cmd
run.bat
```

Service will start on: http://localhost:5000

### Step 4: Test

Open browser or use curl:
```
http://localhost:5000/health
```

Should return:
```json
{
  "status": "healthy",
  "service": "Revit API Service",
  "revit_initialized": true
}
```

## Production Deployment

### Install as Windows Service

```cmd
REM Right-click Command Prompt → Run as Administrator
install-service.bat
```

### Start Service

```cmd
sc start RevitAPIService
```

### Check Status

```cmd
sc query RevitAPIService
```

### View Logs

```
logs/revit-service.log
```

## Firewall Configuration

Allow port 5000:

```powershell
New-NetFirewallRule -DisplayName "Revit API Service" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
```

## Troubleshooting

### "Revit not found"

Update RevitService.csproj with your Revit installation path:

```xml
<Reference Include="RevitAPI">
  <HintPath>C:\Program Files\Autodesk\Revit 2023\RevitAPI.dll</HintPath>
</Reference>
```

### "Build failed"

Make sure .NET 6.0 SDK is installed:
```cmd
dotnet --version
```

### "Service won't start"

1. Check Revit license is active
2. Check logs: `logs/revit-service.log`
3. Run manually first: `run.bat`

## Testing from Ubuntu

```bash
# From Ubuntu terminal
curl http://WINDOWS_IP:5000/health

# Should return health status
```

## Architecture

```
Ubuntu (Linux)                    Windows Server
┌─────────────────┐              ┌──────────────────┐
│ FastAPI Backend │─────────────▶│ RevitService.exe │
│ Port 8000       │   HTTP/JSON  │ Port 5000        │
└─────────────────┘              └──────────────────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │ Revit API    │
                                  │ Creates .RVT │
                                  └──────────────┘
```

## API Endpoints

### GET /health
Check if service is running

### POST /build-model
Build Revit model from transaction JSON

Request:
```json
{
  "jobId": "abc123",
  "transactionJson": "{...}"
}
```

Response: Binary .RVT file

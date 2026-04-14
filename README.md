# MCC Amplify AI — Floor Plan to BIM

AI-powered system that converts PDF floor plans into native Revit (`.RVT`) BIM models and interactive 3D web previews (glTF/GLB).

---

## What the System Does

Upload a PDF architectural floor plan → receive a fully-formed, editable Revit file. No manual re-drawing, no IFC round-trips. The system:

1. Extracts precise vector geometry and text directly from the PDF.
2. Renders the page as a high-resolution image and runs a YOLOv11 detector to identify walls, doors, windows, columns, and rooms.
3. Fuses vector precision with ML detection results — wall endpoints are snapped to the nearest axis-aligned vector line.
4. Detects the structural column grid from vector geometry and its dimension annotations to derive the real-world coordinate scale. Scale text printed on the drawing (e.g. "1:100") is intentionally ignored as unreliable.
5. Sends the rendered image and fused elements to a vision-capable AI model (Google Gemini or Anthropic Claude) for semantic enrichment — building type, materials, room purposes, etc.
6. Generates Semantic 3D parameters (wall locations, door swings, window sill heights, slab boundaries) formatted as Revit API instructions.
7. Sends the instructions over the network to a Windows machine running Revit, where a C# Add-in creates all elements natively via the Revit API, and returns the resulting `.RVT` file.
8. Simultaneously exports a lightweight `.glb` (glTF binary) for instant web-based 3D preview, including walls, columns, doors, windows, and floor slabs.

---

## Architecture

```
Ubuntu (Linux) Machine                    Windows Machine
┌──────────────────────────────────────┐  ┌──────────────────────────────────┐
│  FastAPI Backend (port 8000)         │  │  Revit 2023                      │
│  ┌────────────────────────────────┐  │  │  ┌────────────────────────────┐  │
│  │  PDF Security Check            │  │  │  │  C# Add-in (RevitAddin)    │  │
│  │  Track A: Vector Extraction    │  │  │  │  TcpListener on TCP :5000  │  │
│  │  Track B: Raster Render+YOLO   │  │  │  │  Receives JSON transaction  │  │
│  │  Hybrid Fusion (vector snap)   │  │  │  │  Builds walls/doors/columns │  │
│  │  Grid-based Scale Detection    │  │  │  │  Returns .RVT file         │  │
│  │  Semantic AI (Gemini/Claude)   │──┼──┼─>│                            │  │
│  │  3D Geometry Generation        │  │  │  └────────────────────────────┘  │
│  │  RVT Exporter (RevitClient)    │<─┼──┼──                                │
│  │  glTF Exporter                 │  │  └──────────────────────────────────┘
│  └────────────────────────────────┘  │
│  Chat Agent (NVIDIA NIM / Gemini)    │
│  React + Three.js Frontend (5173)    │
└──────────────────────────────────────┘
```

Both machines must be on the same local network (or VPN). The Ubuntu machine is the primary — it hosts the web UI, runs all AI processing, and drives the Windows Revit machine.

---

## Full Workflow

### Step 1 — Prepare Windows (run once per session)

On the **Windows machine**, open PowerShell as Administrator and run:

```powershell
cd C:\path\to\mcc-amplify-ai\revit_server\RevitAddin
dotnet clean && dotnet build
Copy-Item RevitModelBuilder.addin, bin\Debug\net48\RevitModelBuilderAddin.dll `
    -Destination "C:\ProgramData\Autodesk\Revit\Addins\2023\"
Start-Process "C:\Program Files\Autodesk\Revit 2023\Revit.exe"
```

**When Revit opens:**
- Click **"Always Load"** on the security dialog to allow the Add-in.
- Open or create a project file — the Add-in requires an active document.

Once Revit has fully loaded, verify the service is reachable from Ubuntu:

```bash
curl http://LT-HQ-277:5000/health
# Expected: Revit Model Builder ready
```

### Step 2 — Configure network on Ubuntu (first-time only)

```bash
echo "191.168.124.64 LT-HQ-277" | sudo tee -a /etc/hosts
# Replace with your actual Windows IP (ipconfig on Windows) and hostname
```

### Step 3 — Start the Ubuntu system

```bash
# From the project root
./run.sh
```

This starts:
- **Backend** at `http://localhost:8000`
- **Frontend** at `http://localhost:5173`

The script waits for the backend to be ready before starting the frontend. Press `Ctrl+C` to stop both.

### Step 4 — Upload and process

1. Open `http://localhost:5173` in a browser.
2. Upload a **PDF** floor plan (max 50 MB).
3. Watch the real-time progress bar advance through all pipeline stages.
4. When processing completes:
   - Download the native **`.RVT`** file and open it in Revit — all walls, doors, windows, and columns will be editable native elements.
   - View the **3D web preview** (glTF) directly in the browser.

---

## Pipeline Stages

```mermaid
flowchart TD
    PDF(["📄 PDF Upload"])

    PDF --> SEC

    subgraph SEC["① Security & Size Check"]
        sec["SecurePDFRenderer
        Validate magic bytes · cap render DPI"]
    end

    SEC --> VEC & RAS

    subgraph PAR["② Parallel Tracks"]
        VEC["Track A · Vector Extraction
        VectorProcessor
        PyMuPDF paths + text spans"]
        RAS["Track B · Raster Render
        StreamingProcessor → YOLOv11
        walls · doors · windows · columns · rooms"]
    end

    VEC & RAS --> FUS

    subgraph FUSBOX["③ Hybrid Fusion"]
        FUS["HybridFusionPipeline
        Snap YOLO detections to vector lines
        Refine wall endpoints"]
    end

    FUS --> GRD

    subgraph GRDBOX["④ Grid Detection"]
        GRD["GridDetector
        Structural grid from PDF text labels
        Column schedule scan
        px → real-world mm scale"]
    end

    GRD --> SEM

    subgraph SEMBOX["⑤ Semantic AI  — first backend wins"]
        SEM["SemanticAnalyzer"]
        SEM -->|priority 1| OLL["Ollama
        gemma3:4b / qwen3-vl"]
        SEM -->|priority 2| GEM["Google Gemini API"]
        SEM -->|priority 3| ANT["Anthropic Claude API"]
    end

    OLL & GEM & ANT --> GEO

    subgraph GEOBOX["⑥ 3D Geometry Generation"]
        GEO["GeometryGenerator
        Image coords → Revit world mm
        Y-axis inversion · grid snapping
        Build Revit API recipe JSON"]
    end

    GEO --> RVTBOX & GLTBOX

    subgraph RVTBOX["⑦a  RVT Export  (Linux → Windows)"]
        EXP["RvtExporter · RevitClient
        HTTP POST to Revit 2023 C# Add-in
        Builds levels · grids · walls
        columns · doors · windows · slabs"]
        EXP --> WRN{"Revit warnings?"}
        WRN -->|"yes · attempt ≤ 2"| FIX["AI correction round
        SemanticAnalyzer patches recipe"]
        FIX -->|resend| EXP
        WRN -->|"no / accepted"| RVT["🏗 {pdf_stem}_{uuid}.rvt"]
    end

    subgraph GLTBOX["⑦b  glTF Export"]
        GLTF["GltfExporter
        .glb — walls · columns · floors · openings"]
    end

    RVT & GLTF --> DONE(["✅ BIM model + 3D preview ready for download"])
```

| # | Stage | Component | Notes |
|---|-------|-----------|-------|
| 1 | Security & size check | `SecurePDFRenderer` | Caps render DPI to prevent resource exhaustion |
| 2a | Vector extraction | `VectorProcessor` | Extracts paths + text spans from the PDF (PyMuPDF) |
| 2b | Raster render | `StreamingProcessor` | Renders first page as a numpy RGB image |
| 2c | Element detection | YOLO (inline) | YOLOv11 detects walls, doors, windows, columns, rooms |
| 3 | Hybrid fusion | `HybridFusionPipeline` | Snaps YOLO wall endpoints to nearest vector line |
| 4 | Grid detection | `GridDetector` | Derives real-world scale from structural grid lines + dimension annotations; scale text ignored |
| 5 | Semantic AI analysis | `SemanticAnalyzer` | Ollama (gemma3:4b) → Gemini → Claude; enriches elements with material, type, room purpose |
| 6 | 3D geometry | `GeometryGenerator` | Converts 2D elements to Revit API parameters (grid-based mm coords, Y-axis inverted for Revit) |
| 7 | BIM export | `RvtExporter` + `GltfExporter` | Sends to Windows Revit with closed-loop warning correction; also writes `.glb` |

---

## Project Structure

```
mcc-amplify-ai/
├── run.sh                              ← Start Ubuntu backend + frontend
├── scripts/
│   └── run.sh                          ← Same as above
│
├── backend/
│   ├── app.py                          ← FastAPI entry point
│   ├── .env                            ← Configuration (not committed)
│   ├── nvidia_key.txt                  ← NVIDIA NIM API key file (not committed)
│   ├── google_key.txt                  ← Google API key file (not committed)
│   ├── api/
│   │   ├── routes.py                   ← REST endpoints (upload, process, download)
│   │   └── websocket.py                ← Real-time progress updates
│   ├── chat_agent/
│   │   └── agent.py                    ← Chat agent (NVIDIA NIM DeepSeek / Gemini)
│   ├── core/
│   │   └── pipeline.py                 ← Thin wrapper around PipelineOrchestrator
│   ├── services/
│   │   ├── core/orchestrator.py        ← Main pipeline orchestrator
│   │   ├── pdf_processing/             ← VectorProcessor, StreamingProcessor
│   │   ├── security/                   ← SecurePDFRenderer, ResourceMonitor
│   │   ├── fusion/pipeline.py          ← HybridFusionPipeline (vector snapping)
│   │   ├── grid_detector.py            ← Structural grid detection, pixel→mm conversion
│   │   ├── semantic_analyzer.py        ← Multi-backend AI (Gemini / Claude)
│   │   ├── geometry_generator.py       ← 2D → Revit 3D parameter builder
│   │   ├── revit_client.py             ← HTTP client → Windows Revit Add-in
│   │   └── exporters/
│   │       ├── rvt_exporter.py         ← Sends to Windows, receives .RVT
│   │       └── gltf_exporter.py        ← Writes .glb (walls, doors, windows, floors)
│   └── utils/
│       └── api_keys.py                 ← Key resolution (env var → .txt file)
│
├── revit_server/
│   └── RevitAddin/                     ← C# Revit 2023 Add-in (build on Windows)
│       ├── App.cs                      ← TcpListener :5000 + ExternalEvent handler
│       ├── ModelBuilder.cs             ← Creates levels, grids, walls, columns, etc.
│       ├── RevitModelBuilder.addin     ← Revit add-in manifest
│       └── RevitAddin.csproj
│
├── frontend/                           ← React + Three.js web UI
│   └── src/
│       └── components/
│
├── ml/
│   └── weights/
│       └── yolov11_floorplan.pt        ← YOLO model weights
│
└── data/                               ← Runtime data (created automatically)
    ├── uploads/
    └── models/
        ├── rvt/                        ← Returned .RVT files
        └── gltf/                       ← Exported .glb files
```

---

## Configuration

Key settings in `backend/.env`:

```bash
# ── Chat Agent ────────────────────────────────────────────────────────────────
# NVIDIA NIM is default (free 1000 credits) — put key in backend/nvidia_key.txt
# Sign up: https://build.nvidia.com
CHAT_MODEL_BACKEND=nvidia_nim           # or: gemini_api

# ── Semantic AI Backend (pipeline Stage 5) ────────────────────────────────────
SEMANTIC_MODEL_BACKEND=gemini_api       # or: anthropic_api
# Put API keys in backend/google_key.txt or backend/nvidia_key.txt (gitignored)
# Env vars also accepted: GOOGLE_API_KEY, ANTHROPIC_API_KEY, NVIDIA_API_KEY

# ── Windows Revit Server ───────────────────────────────────────────────────────
WINDOWS_REVIT_SERVER=http://LT-HQ-277:5000
REVIT_SERVER_API_KEY=choose_a_shared_secret

# ── FastAPI ───────────────────────────────────────────────────────────────────
APP_HOST=0.0.0.0
APP_PORT=8000

# ── Upload Limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE=52428800   # 50 MB
ALLOWED_EXTENSIONS=pdf
```

### Architectural defaults (geometry_generator)

| Parameter | Default | Unit |
|-----------|---------|------|
| Wall height | 2800 | mm |
| Wall thickness | 200 | mm |
| Door height | 2100 | mm |
| Window height | 1500 | mm |
| Sill height | 900 | mm |
| Floor thickness | 200 | mm |

---

## System Requirements

### Ubuntu Machine
- Ubuntu 20.04 LTS or newer
- Python 3.10+
- Node.js 18+
- 16 GB RAM minimum (32 GB recommended for large PDFs)
- `tesseract-ocr`, `poppler-utils` installed

### Windows Machine
- Windows 10/11 Pro or Windows Server 2019+
- Revit 2023 (with valid license)
- .NET SDK 8.0 (for building) + .NET Framework 4.8 (for running)

---

## Troubleshooting

**`curl http://LT-HQ-277:5000/health` times out**
- Verify the Windows IP in `/etc/hosts` is correct.
- Check that Windows Firewall allows inbound TCP on port 5000:
  ```powershell
  netsh advfirewall firewall add rule name="RevitAddin5000" dir=in action=allow protocol=TCP localport=5000 profile=any
  ```
- Ensure Revit is not in a modal state (Options dialog, Print dialog) — these block the Revit API.

**Grid detection falls back to uniform grid**
- The system derives scale from structural column grid lines and dimension annotations in the PDF vector layer.
- If no grid is detected, a uniform 5×4 grid at 6000 mm bays is used as a fallback.
- Check the backend log: grid source and line count are logged at Stage 4.

**YOLO weights not found**
- Place the trained weights at `ml/weights/yolov11_floorplan.pt`.
- The pipeline continues without YOLO; only vector geometry is used downstream.

**Backend won't start**
```bash
which python && python -c "import fastapi, loguru; print('OK')"
tail -50 logs/app.log
```

**RVT file empty / Revit error**
- Check `C:\RevitOutput\addin_startup.log` and `C:\RevitOutput\build_log.txt` on Windows.
- Confirm the Add-in loaded: check the Add-ins tab in the Revit ribbon.
- Run Revit as Administrator if permission errors appear.

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| Processing time | 30–90 s per floor plan |
| Wall detection accuracy | 85–95 % |
| Door / window detection | 80–90 % |
| Column detection | 75–90 % |
| Max file size | 50 MB |

---

## Closed-loop Revit Feedback System

 Full closed-loop flow:                                                                                                                                                                                         
                                                                                                                                                                                                                 
  Revit builds model
         ↓
  WarningCollector (C#) silently dismisses dialogs
         ↓ (X-Revit-Warnings header)
  revit_client.py parses warnings list
         ↓
  orchestrator: if warnings exist AND attempt < 3:
      semantic_ai.analyze_revit_warnings(warnings, recipe)
         ↓ (AI returns corrections JSON)
      _apply_revit_corrections patches recipe
         ↓
      resend to Revit (max 2 correction rounds)
         ↓
  Accept result (warnings logged but not fatal)

  Key behaviors:

  - No popup dialogs — IFailuresPreprocessor.PreprocessFailures() calls fa.DeleteWarning(msg) on every warning, so Revit never shows the dialog to the user
  - Fatal errors (e.g., geometry that Revit can't build) still cause transaction rollback and return a 500 from the C# server — the Python side raises an exception as before
  - Max 2 correction rounds — prevents infinite loops; after round 2, whatever Revit built is accepted
  - Safety guardrails on corrections — _apply_revit_corrections() only patches whitelisted numeric fields (width, depth, height, thickness, etc.) so the AI cannot corrupt structural keys like level or id



## Support

- GitHub Issues: <https://github.com/josephteh97/mcc-amplify-ai/issues>
- API docs (when backend is running): <http://localhost:8000/api/docs>

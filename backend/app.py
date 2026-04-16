import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Main FastAPI Application
Amplify-Like Floor Plan to BIM System
"""

import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os
from dotenv import load_dotenv
from loguru import logger

from pipeline import FloorPlanPipeline
from api.routes import router as api_router
from api.websocket import manager as ws_manager, ws_router
from utils.logger import setup_logger
from chat_agent.agent import ChatAgent
from fastapi import WebSocketDisconnect

# Load environment variables
load_dotenv()

# Setup logging
setup_logger()

# Chat agent — singleton shared across all WebSocket sessions
chat_agent = ChatAgent()


_revit_healthy = False  # shared flag for frontend status queries


async def _revit_heartbeat(client, interval: int = 30):
    """Periodically check Revit server health so stale connections are detected early."""
    global _revit_healthy
    while True:
        try:
            _revit_healthy = await client.check_health()
            if not _revit_healthy:
                logger.warning("Revit heartbeat: server unreachable")
        except Exception as e:
            _revit_healthy = False
            logger.warning(f"Revit heartbeat error: {e}")
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    global _revit_healthy

    # Startup
    logger.info("Starting Amplify Floor Plan AI System")

    # Create necessary directories
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/models/revit_transactions").mkdir(parents=True, exist_ok=True)
    Path("data/models/rvt").mkdir(parents=True, exist_ok=True)
    Path("data/models/gltf").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Test Windows Revit server connection
    from services.revit_client import RevitClient
    revit_client = RevitClient()
    _revit_healthy = await revit_client.check_health()

    if _revit_healthy:
        logger.info("✓ Connected to Windows Revit server")
    else:
        logger.warning("✗ Cannot connect to Windows Revit server - RVT export will fail")

    # Start periodic heartbeat (every 30 s)
    heartbeat_interval = int(os.getenv("REVIT_HEARTBEAT_INTERVAL", "30"))
    heartbeat_task = asyncio.create_task(_revit_heartbeat(revit_client, heartbeat_interval))

    logger.info("System ready!")

    yield  # Application runs here

    # Shutdown
    heartbeat_task.cancel()
    logger.info("Shutting down Amplify Floor Plan AI System")
    await ws_manager.disconnect_all()


# Create FastAPI app with lifespan
app = FastAPI(
    title="Amplify Floor Plan AI",
    description="AI-powered PDF floor plan to native Revit (.RVT) conversion",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)


# ── Chat WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws/chat/{user_id}")
async def chat_websocket(websocket: WebSocket, user_id: str):
    """
    Bidirectional chat endpoint.

    Incoming JSON:
      { "type": "user_message", "message": "...", "context": { "job_id": "..." } }

    Outgoing JSON:
      { "type": "agent_message", "message": "...", "metadata": { ... } }
    """
    await websocket.accept()
    await chat_agent.on_connect(user_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "user_message":
                reply = await chat_agent.handle_message(
                    user_id=user_id,
                    message=data.get("message", ""),
                    context_data=data.get("context", {}),
                )
                await websocket.send_json({
                    "type": "agent_message",
                    "message": reply,
                    "metadata": {"auto_generated": False},
                })

    except WebSocketDisconnect:
        chat_agent.on_disconnect(user_id)
    except Exception as exc:
        logger.error(f"Chat WebSocket error for {user_id}: {exc}")
        chat_agent.on_disconnect(user_id)

@app.get("/api/revit-health")
async def revit_health():
    """Return Revit server connectivity status (updated by heartbeat)."""
    return {"revit_available": _revit_healthy}


@app.get("/")
async def root():
    """Root endpoint - serves frontend"""
    # Check if frontend build exists
    if Path("../frontend/dist/index.html").exists():
        return FileResponse("../frontend/dist/index.html")
    else:
        # Fallback for development mode
        return {
            "message": "Backend is running. For frontend, run 'npm run dev' in the frontend directory and visit http://localhost:5173",
            "docs": "/api/docs"
        }

# Serve static files (frontend build)
if Path("../frontend/dist").exists():
    app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
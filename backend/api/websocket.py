"""
WebSocket Manager for Real-time Progress Updates
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
from loguru import logger
import json


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        
        self.active_connections[job_id].add(websocket)
        logger.info(f"WebSocket connected for job {job_id}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove WebSocket connection"""
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        
        logger.info(f"WebSocket disconnected for job {job_id}")
    
    async def send_progress(self, job_id: str, data: dict):
        """Send progress update to all connected clients for a job"""
        if job_id in self.active_connections:
            message = json.dumps(data, default=str)
            
            for connection in self.active_connections[job_id].copy():
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    self.disconnect(connection, job_id)
    
    async def disconnect_all(self):
        """Disconnect all clients"""
        for job_id, connections in self.active_connections.items():
            for connection in connections:
                await connection.close()
        
        self.active_connections.clear()


# Global instance
manager = ConnectionManager()


# WebSocket endpoint
from fastapi import APIRouter

ws_router = APIRouter()


@ws_router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time pipeline progress.

    The server pushes JSON messages whenever the pipeline calls on_progress()
    or completes/fails.  Clients may send any text (ping) — it is ignored.
    Message shape:
      { "type": "progress"|"completed"|"failed",
        "job_id": str, "progress": int, "message": str }
    """
    await manager.connect(websocket, job_id)
    try:
        while True:
            await websocket.receive_text()   # keep-alive; ignore client messages
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

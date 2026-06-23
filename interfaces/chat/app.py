"""
Chat Application

FastAPI application factory and route registration.
All WebSocket handling is delegated to handler.py.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware

from logger import log
from observability import audit
from interfaces.chat.handler import handle_websocket
from interfaces.chat.websocket import ConnectionManager


# ══════════════════════════════════════════════════════════════════════════════
# Connection Manager Singleton
# ══════════════════════════════════════════════════════════════════════════════

_manager: Optional[ConnectionManager] = None


def get_manager() -> ConnectionManager:
    """Get the singleton ConnectionManager."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager


# ══════════════════════════════════════════════════════════════════════════════
# Application Lifecycle
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    log.info("Chat interface starting...")
    audit.log_raw("interface", "startup", "chat", "started")

    cleanup_task = asyncio.create_task(_cleanup_loop())

    yield

    cleanup_task.cancel()
    audit.log_raw("interface", "shutdown", "chat", "completed")
    log.info("Chat interface stopped")


async def _cleanup_loop():
    """Background task to clean up idle sessions."""
    manager = get_manager()
    while True:
        await asyncio.sleep(300)
        removed = manager.cleanup_idle_sessions()
        if removed > 0:
            log.info(f"Cleaned up {removed} idle sessions")


# ══════════════════════════════════════════════════════════════════════════════
# Application Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_chat_app() -> FastAPI:
    """Create the FastAPI chat application."""
    app = FastAPI(
        title="CogniCore Chat",
        description="WebSocket server for real-time chat conversations",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)

    return app


def _register_routes(app: FastAPI):
    """Register health and WebSocket routes."""

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        manager = get_manager()
        return {"status": "ok", "connections": manager.get_status()}

    @app.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        user_id: Optional[str] = Query(None),
        conversation_id: Optional[str] = Query(None),
    ):
        """WebSocket endpoint for chat — delegates to handler."""
        await handle_websocket(
            websocket=websocket,
            manager=get_manager(),
            user_id=user_id,
            conversation_id=conversation_id,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton App
# ══════════════════════════════════════════════════════════════════════════════

_app: Optional[FastAPI] = None


def get_chat_app() -> FastAPI:
    """Get the singleton FastAPI chat app."""
    global _app
    if _app is None:
        _app = create_chat_app()
    return _app

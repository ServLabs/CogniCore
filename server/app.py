"""
Server Application

FastAPI application for WebSocket chat/agent conversations.
This is the primary interface for end-user interactions.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from core import config, log
from audit import audit
from server.websocket import ConnectionManager, Session
from server.streaming import EventStream


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
    # Startup
    log.info("Server starting...")
    audit.log_raw("server", "startup", "app", "started")
    
    # Start background tasks
    cleanup_task = asyncio.create_task(_cleanup_loop())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    audit.log_raw("server", "shutdown", "app", "completed")
    log.info("Server stopped")


async def _cleanup_loop():
    """Background task to clean up idle sessions."""
    manager = get_manager()
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        removed = manager.cleanup_idle_sessions()
        if removed > 0:
            log.info(f"Cleaned up {removed} idle sessions")


# ══════════════════════════════════════════════════════════════════════════════
# Application Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """
    Create the FastAPI application.
    
    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="CogniCore Server",
        description="WebSocket server for real-time chat/agent conversations",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    _register_routes(app)
    
    return app


def _register_routes(app: FastAPI):
    """Register all routes."""
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        manager = get_manager()
        return {
            "status": "ok",
            "connections": manager.get_status(),
        }
    
    @app.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        user_id: Optional[str] = Query(None),
        conversation_id: Optional[str] = Query(None),
    ):
        """
        WebSocket endpoint for chat.
        
        Query params:
        - user_id: Optional user identifier
        - conversation_id: Optional conversation to resume
        """
        manager = get_manager()
        session = await manager.connect(websocket, user_id)
        
        if conversation_id:
            session.conversation_id = conversation_id
        
        try:
            # Send connection confirmation
            await websocket.send_json({
                "type": "connected",
                "session_id": session.session_id,
                "conversation_id": session.conversation_id,
            })
            
            # Message loop
            while True:
                data = await websocket.receive_text()
                await _handle_message(session, data)
        
        except WebSocketDisconnect:
            manager.disconnect(session.session_id)
        
        except Exception as e:
            log.error(f"WebSocket error: {e}", exc_info=True)
            manager.disconnect(session.session_id)


async def _handle_message(session: Session, raw_message: str):
    """
    Handle an incoming WebSocket message.
    
    Args:
        session: User session.
        raw_message: Raw message string.
    """
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError:
        await _send_error(session, "Invalid JSON")
        return
    
    session.touch()
    
    msg_type = message.get("type", "message")
    
    match msg_type:
        case "message":
            await _handle_user_message(session, message)
        
        case "cancel":
            await _handle_cancel(session)
        
        case "ping":
            if session.websocket:
                await session.websocket.send_json({"type": "pong"})
        
        case "set_conversation":
            session.conversation_id = message.get("conversation_id")
            if session.websocket:
                await session.websocket.send_json({
                    "type": "conversation_set",
                    "conversation_id": session.conversation_id,
                })
        
        case _:
            await _send_error(session, f"Unknown message type: {msg_type}")


async def _handle_user_message(session: Session, message: dict[str, Any]):
    """
    Handle a user chat message.
    
    Routes to pipeline and streams events back.
    
    Args:
        session: User session.
        message: Parsed message.
    """
    text = message.get("text", "")
    if not text:
        await _send_error(session, "Empty message")
        return
    
    audit.log_raw(
        "server",
        "message_received",
        "websocket",
        "started",
        session_id=session.session_id,
        details={"length": len(text)},
    )
    
    # Create event stream
    stream = session.create_stream()
    
    # Start streaming task
    stream_task = asyncio.create_task(
        _stream_events(session, stream)
    )
    
    try:
        # Process message through pipeline
        # This would integrate with CEN/Response pipeline
        await _process_message(session, text, stream)
    
    except Exception as e:
        log.error(f"Message processing error: {e}", exc_info=True)
        stream.emit_error(str(e), recoverable=False)
        stream.emit_done({"error": str(e)})
    
    finally:
        # Wait for streaming to complete
        await stream_task


async def _process_message(
    session: Session,
    text: str,
    stream: EventStream,
):
    """
    Process a user message through the pipeline.
    
    Args:
        session: User session.
        text: User message text.
        stream: Event stream for this request.
    """
    import time
    start_time = time.monotonic()
    
    # Emit thinking event
    stream.emit_thinking("Analyzing your request...")
    
    try:
        # Import and use response pipeline
        from response import get_pipeline
        from memory.types.wm import add_user_message, add_assistant_message
        
        pipeline = get_pipeline()
        
        # Store user message in working memory
        if session.user_id and session.conversation_id:
            await add_user_message(
                session.user_id,
                session.conversation_id,
                text,
            )
        
        # Process through pipeline
        result = await pipeline.process(
            query=text,
            user_id=session.user_id,
            conversation_id=session.conversation_id,
            stream=stream,
        )
        
        # Store assistant response in working memory
        if session.user_id and session.conversation_id and result.response:
            await add_assistant_message(
                session.user_id,
                session.conversation_id,
                result.response,
            )
        
        # Emit response
        stream.emit_response_chunk(result.response)
        
        elapsed_ms = (time.monotonic() - start_time) * 1000
        
        stream.emit_done({
            "processing_time_ms": elapsed_ms,
            "used_deep_pipeline": result.used_deep_pipeline,
            "gate_decision": result.gate_decision,
        })
        
        audit.log_raw(
            "server",
            "message_processed",
            "websocket",
            "completed",
            session_id=session.session_id,
            duration_ms=elapsed_ms,
        )
    
    except Exception as e:
        stream.emit_error(str(e), recoverable=False)
        stream.emit_done({"error": str(e)})
        raise


async def _stream_events(session: Session, stream: EventStream):
    """
    Stream events from EventStream to WebSocket.
    
    Args:
        session: User session.
        stream: Event stream to consume.
    """
    if not session.websocket:
        return
    
    async for event in stream:
        try:
            await session.websocket.send_text(event.to_json())
        except Exception:
            break


async def _handle_cancel(session: Session):
    """Handle a cancel request."""
    session.close_stream()
    
    if session.websocket:
        await session.websocket.send_json({
            "type": "cancelled",
            "text": "Request cancelled",
        })
    
    audit.log_raw(
        "server",
        "request_cancel",
        "websocket",
        "completed",
        session_id=session.session_id,
    )


async def _send_error(session: Session, error: str):
    """Send an error message to the client."""
    if session.websocket:
        await session.websocket.send_json({
            "type": "error",
            "text": error,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Singleton App
# ══════════════════════════════════════════════════════════════════════════════

_app: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """Get the singleton FastAPI app."""
    global _app
    if _app is None:
        _app = create_app()
    return _app

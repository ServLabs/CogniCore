"""
WebSocket API Server

Streams the agent's internal process to clients in real-time.
Users see thinking, tool calls, memory recalls, decisions as they happen.
"""

import asyncio
import json
from typing import Any, Optional

from config import config
from agent.sessions import SessionManager, Session
from agent.streaming import EventStream
from audit import audit


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Server
# ══════════════════════════════════════════════════════════════════════════════

class WebSocketServer:
    """
    WebSocket server for real-time agent communication.
    
    Provides:
    - Connection management
    - Message routing to CEN
    - Event streaming to clients
    """
    
    def __init__(self, session_manager: SessionManager):
        """
        Initialize WebSocket server.
        
        Args:
            session_manager: Session manager instance.
        """
        self.sessions = session_manager
        self._server = None
        self._connections: dict[str, Any] = {}  # session_id -> websocket
    
    async def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        """
        Start the WebSocket server.
        
        Args:
            host: Host to bind to.
            port: Port to bind to.
        """
        try:
            import websockets
            
            self._server = await websockets.serve(
                self._handle_connection,
                host,
                port,
            )
            
            audit.log_raw(
                "agent",
                "websocket_start",
                "api",
                "completed",
                details={"host": host, "port": port},
            )
        except ImportError:
            # websockets not installed - log warning
            audit.log_raw(
                "agent",
                "websocket_start",
                "api",
                "failed",
                error="websockets package not installed",
            )
    
    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
    
    async def _handle_connection(self, websocket, path: str) -> None:
        """
        Handle a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection.
            path: Connection path.
        """
        session: Optional[Session] = None
        
        try:
            # Create session for this connection
            session = self.sessions.create_session()
            self._connections[session.session_id] = websocket
            
            audit.log_raw(
                "agent",
                "connection_open",
                "api",
                "completed",
                session_id=session.session_id,
            )
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(session, websocket, message)
        
        except Exception as e:
            audit.log_raw(
                "agent",
                "connection_error",
                "api",
                "failed",
                error=str(e),
                session_id=session.session_id if session else None,
            )
        
        finally:
            # Cleanup
            if session:
                self._connections.pop(session.session_id, None)
                self.sessions.remove_session(session.session_id)
                
                audit.log_raw(
                    "agent",
                    "connection_close",
                    "api",
                    "completed",
                    session_id=session.session_id,
                )
    
    async def _handle_message(
        self,
        session: Session,
        websocket,
        raw_message: str,
    ) -> None:
        """
        Handle an incoming message.
        
        Args:
            session: User session.
            websocket: WebSocket connection.
            raw_message: Raw message string.
        """
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON")
            return
        
        session.touch()
        
        msg_type = message.get("type", "message")
        
        match msg_type:
            case "message":
                await self._handle_user_message(session, websocket, message)
            
            case "cancel":
                await self._handle_cancel(session)
            
            case "ping":
                await websocket.send(json.dumps({"type": "pong"}))
            
            case _:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")
    
    async def _handle_user_message(
        self,
        session: Session,
        websocket,
        message: dict[str, Any],
    ) -> None:
        """
        Handle a user message.
        
        Routes to CEN and streams events back.
        
        Args:
            session: User session.
            websocket: WebSocket connection.
            message: Parsed message.
        """
        text = message.get("text", "")
        if not text:
            await self._send_error(websocket, "Empty message")
            return
        
        # Create event stream
        stream = session.create_stream()
        
        # Start streaming task
        stream_task = asyncio.create_task(
            self._stream_events(websocket, stream)
        )
        
        try:
            # Route to response pipeline
            from response import process_message
            from control.salience import AgentEvent, get_salience_network
            
            # Ingest event to salience network
            salience = get_salience_network()
            await salience.ingest_event(AgentEvent(
                type="user_message",
                payload={
                    "message": text,
                    "user_id": session.user_id,
                    "convo_id": session.conversation_id,
                    "session_id": session.session_id,
                    "stream": stream,
                },
            ))
            
            # Process message directly for now
            result = await process_message(
                message=text,
                user_id=session.user_id,
                convo_id=session.conversation_id,
            )
            
            # Emit response
            stream.emit_response_chunk(result.response)
            stream.emit_done({
                "processing_time_ms": result.processing_time_ms,
                "used_deep_pipeline": result.used_deep_pipeline,
            })
        
        except Exception as e:
            stream.emit_error(str(e), recoverable=False)
            stream.emit_done({"error": str(e)})
        
        finally:
            # Wait for streaming to complete
            await stream_task
    
    async def _handle_cancel(self, session: Session) -> None:
        """
        Handle a cancel request.
        
        Args:
            session: User session.
        """
        session.close_stream()
        
        audit.log_raw(
            "agent",
            "request_cancel",
            "api",
            "completed",
            session_id=session.session_id,
        )
    
    async def _stream_events(
        self,
        websocket,
        stream: EventStream,
    ) -> None:
        """
        Stream events from EventStream to WebSocket.
        
        Args:
            websocket: WebSocket connection.
            stream: Event stream to consume.
        """
        async for event in stream:
            try:
                await websocket.send(event.to_json())
            except Exception:
                break
    
    async def _send_error(self, websocket, error: str) -> None:
        """Send an error message to the client."""
        await websocket.send(json.dumps({
            "type": "error",
            "text": error,
        }))
    
    def get_status(self) -> dict[str, Any]:
        """Get server status."""
        return {
            "running": self._server is not None,
            "connections": len(self._connections),
            "sessions": self.sessions.get_status(),
        }

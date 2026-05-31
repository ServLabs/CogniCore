"""
WebSocket Connection Manager

Manages WebSocket connections for real-time chat.
Handles connection lifecycle, message routing, and broadcasting.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

from audit import audit
from core import log
from server.streaming import EventStream


@dataclass
class Session:
    """
    A user session.
    
    Tracks user state, current conversation, and WebSocket connection.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    websocket: Optional[WebSocket] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Current event stream (None if no active request)
    _current_stream: Optional[EventStream] = field(default=None, repr=False)
    
    def create_stream(self) -> EventStream:
        """Create a new event stream for this session."""
        convo_id = self.conversation_id or str(uuid.uuid4())
        self._current_stream = EventStream(self.session_id, convo_id)
        return self._current_stream
    
    def get_stream(self) -> Optional[EventStream]:
        """Get the current event stream."""
        return self._current_stream
    
    def close_stream(self) -> None:
        """Close the current event stream."""
        if self._current_stream and not self._current_stream.is_closed:
            self._current_stream.emit_done()
        self._current_stream = None
    
    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)


class ConnectionManager:
    """
    Manages all WebSocket connections.
    
    Provides:
    - Connection lifecycle management
    - Session tracking
    - Message broadcasting
    - Idle session cleanup
    """
    
    def __init__(self, idle_timeout_seconds: int = 3600):
        """
        Initialize connection manager.
        
        Args:
            idle_timeout_seconds: Seconds before idle session is cleaned up.
        """
        self._sessions: dict[str, Session] = {}
        self._user_sessions: dict[str, str] = {}  # user_id -> session_id
        self.idle_timeout_seconds = idle_timeout_seconds
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
    ) -> Session:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket.
            user_id: Optional user ID.
            
        Returns:
            New Session.
        """
        await websocket.accept()
        
        session = Session(
            user_id=user_id,
            websocket=websocket,
        )
        
        self._sessions[session.session_id] = session
        
        if user_id:
            self._user_sessions[user_id] = session.session_id
        
        audit.log_raw(
            "server",
            "connection_open",
            "websocket",
            "completed",
            session_id=session.session_id,
            user_id=user_id,
        )
        
        log.info(f"WebSocket connected: session={session.session_id}")
        
        return session
    
    def disconnect(self, session_id: str) -> None:
        """
        Handle WebSocket disconnection.
        
        Args:
            session_id: Session ID to disconnect.
        """
        session = self._sessions.pop(session_id, None)
        
        if session:
            session.close_stream()
            
            if session.user_id:
                self._user_sessions.pop(session.user_id, None)
            
            audit.log_raw(
                "server",
                "connection_close",
                "websocket",
                "completed",
                session_id=session_id,
            )
            
            log.info(f"WebSocket disconnected: session={session_id}")
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_session_by_user(self, user_id: str) -> Optional[Session]:
        """Get a session by user ID."""
        session_id = self._user_sessions.get(user_id)
        if session_id:
            return self._sessions.get(session_id)
        return None
    
    async def send_json(self, session_id: str, data: dict[str, Any]) -> bool:
        """
        Send JSON data to a specific session.
        
        Args:
            session_id: Target session ID.
            data: Data to send.
            
        Returns:
            True if sent successfully.
        """
        session = self._sessions.get(session_id)
        if session and session.websocket:
            try:
                await session.websocket.send_json(data)
                return True
            except Exception as e:
                log.warning(f"Failed to send to session {session_id}: {e}")
        return False
    
    async def send_text(self, session_id: str, text: str) -> bool:
        """
        Send text data to a specific session.
        
        Args:
            session_id: Target session ID.
            text: Text to send.
            
        Returns:
            True if sent successfully.
        """
        session = self._sessions.get(session_id)
        if session and session.websocket:
            try:
                await session.websocket.send_text(text)
                return True
            except Exception as e:
                log.warning(f"Failed to send to session {session_id}: {e}")
        return False
    
    async def broadcast(self, data: dict[str, Any]) -> int:
        """
        Broadcast data to all connected sessions.
        
        Args:
            data: Data to broadcast.
            
        Returns:
            Number of sessions reached.
        """
        count = 0
        for session_id in list(self._sessions.keys()):
            if await self.send_json(session_id, data):
                count += 1
        return count
    
    def cleanup_idle_sessions(self) -> int:
        """
        Remove idle sessions.
        
        Returns:
            Number of sessions removed.
        """
        now = datetime.now(timezone.utc)
        to_remove = []
        
        for session_id, session in self._sessions.items():
            idle_seconds = (now - session.last_activity).total_seconds()
            if idle_seconds > self.idle_timeout_seconds:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            self.disconnect(session_id)
        
        return len(to_remove)
    
    @property
    def active_connections(self) -> int:
        """Count of active connections."""
        return len(self._sessions)
    
    def get_status(self) -> dict[str, Any]:
        """Get connection manager status."""
        return {
            "active_connections": len(self._sessions),
            "users_with_sessions": len(self._user_sessions),
            "idle_timeout_seconds": self.idle_timeout_seconds,
        }

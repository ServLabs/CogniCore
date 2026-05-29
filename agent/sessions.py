"""
Session Management

Multi-user session management. Each connected client gets a session
that tracks their state, conversation, and event stream.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from agent.streaming import EventStream


# ══════════════════════════════════════════════════════════════════════════════
# Session
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Session:
    """
    A user session.
    
    Tracks user state, current conversation, and event stream.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Current event stream (None if no active request)
    _current_stream: Optional[EventStream] = field(default=None, repr=False)
    
    def create_stream(self) -> EventStream:
        """
        Create a new event stream for this session.
        
        Returns:
            New EventStream.
        """
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
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Session Manager
# ══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """
    Manages all active sessions.
    
    Provides:
    - Session creation and lookup
    - Session cleanup (idle timeout)
    - User-to-session mapping
    """
    
    def __init__(self, idle_timeout_seconds: int = 3600):
        """
        Initialize session manager.
        
        Args:
            idle_timeout_seconds: Seconds before idle session is cleaned up.
        """
        self._sessions: dict[str, Session] = {}
        self._user_sessions: dict[str, str] = {}  # user_id -> session_id
        self.idle_timeout_seconds = idle_timeout_seconds
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            user_id: Optional user ID.
            conversation_id: Optional conversation ID.
            metadata: Optional metadata.
            
        Returns:
            New Session.
        """
        session = Session(
            user_id=user_id,
            conversation_id=conversation_id,
            metadata=metadata or {},
        )
        
        self._sessions[session.session_id] = session
        
        if user_id:
            self._user_sessions[user_id] = session.session_id
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session ID.
            
        Returns:
            Session or None if not found.
        """
        return self._sessions.get(session_id)
    
    def get_session_by_user(self, user_id: str) -> Optional[Session]:
        """
        Get a session by user ID.
        
        Args:
            user_id: User ID.
            
        Returns:
            Session or None if not found.
        """
        session_id = self._user_sessions.get(user_id)
        if session_id:
            return self._sessions.get(session_id)
        return None
    
    def get_or_create_session(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """
        Get existing session or create new one.
        
        Args:
            user_id: Optional user ID.
            session_id: Optional session ID.
            
        Returns:
            Session.
        """
        # Try by session_id first
        if session_id:
            session = self.get_session(session_id)
            if session:
                session.touch()
                return session
        
        # Try by user_id
        if user_id:
            session = self.get_session_by_user(user_id)
            if session:
                session.touch()
                return session
        
        # Create new
        return self.create_session(user_id=user_id)
    
    def remove_session(self, session_id: str) -> None:
        """
        Remove a session.
        
        Args:
            session_id: Session ID to remove.
        """
        session = self._sessions.pop(session_id, None)
        if session and session.user_id:
            self._user_sessions.pop(session.user_id, None)
    
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
            self.remove_session(session_id)
        
        return len(to_remove)
    
    def list_sessions(self) -> list[Session]:
        """List all active sessions."""
        return list(self._sessions.values())
    
    def count(self) -> int:
        """Count active sessions."""
        return len(self._sessions)
    
    def get_status(self) -> dict[str, Any]:
        """Get session manager status."""
        return {
            "active_sessions": len(self._sessions),
            "users_with_sessions": len(self._user_sessions),
            "idle_timeout_seconds": self.idle_timeout_seconds,
        }

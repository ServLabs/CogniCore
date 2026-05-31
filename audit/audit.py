"""
Audit Logger

Universal audit logger. Append-only JSONL files.
Provides decorator, context manager, and direct logging.
"""

import asyncio
import functools
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core import config


# ══════════════════════════════════════════════════════════════════════════════
# Audit Event Schema
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AuditEvent:
    """
    Universal audit event schema.
    
    Every action in the system produces an AuditEvent.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    component: str = ""       # "response", "memory", "control", "connector", "mml", etc.
    action: str = ""          # "llm_call", "memory_read", "sub_agent_spawn", etc.
    actor: str = ""           # "cen", "response.thinking", "mml.consolidation", "user"
    status: str = ""          # "started", "completed", "failed", "denied"
    target: str = ""          # what was acted upon: "sfm", "snowflake", "gpt-4o", etc.
    details: dict[str, Any] = field(default_factory=dict)  # action-specific payload
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "component": self.component,
            "action": self.action,
            "actor": self.actor,
            "status": self.status,
            "target": self.target,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Audit Logger
# ══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Universal audit logger.
    
    Provides:
    - Append-only JSONL file storage
    - Daily log rotation
    - Context manager for tracking duration
    - Decorator for automatic logging
    """
    
    def __init__(self):
        """Initialize audit logger."""
        self._audit_dir = config.paths.audit_dir
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = None
        self._current_date: Optional[str] = None
    
    def _get_file(self):
        """Get current log file, rotating daily."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._current_file:
                self._current_file.close()
            path = self._audit_dir / f"{today}.jsonl"
            self._current_file = open(path, "a", buffering=1)  # line-buffered
            self._current_date = today
        return self._current_file
    
    def log(self, event: AuditEvent) -> None:
        """
        Write an audit event.
        
        Non-blocking, append-only.
        
        Args:
            event: AuditEvent to log.
        """
        f = self._get_file()
        f.write(json.dumps(event.to_dict(), default=str) + "\n")
    
    def log_raw(
        self,
        component: str,
        action: str,
        actor: str,
        status: str,
        target: str = "",
        details: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        """
        Convenience method — log without constructing AuditEvent manually.
        
        Args:
            component: System component.
            action: Action being performed.
            actor: Who/what is performing the action.
            status: Status of the action.
            target: What is being acted upon.
            details: Additional details.
            duration_ms: Duration in milliseconds.
            error: Error message if failed.
            user_id: User ID if applicable.
            session_id: Session ID if applicable.
            conversation_id: Conversation ID if applicable.
        """
        event = AuditEvent(
            component=component,
            action=action,
            actor=actor,
            status=status,
            target=target,
            details=details or {},
            duration_ms=duration_ms,
            error=error,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )
        self.log(event)
    
    @asynccontextmanager
    async def track(
        self,
        component: str,
        action: str,
        actor: str,
        target: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **extra,
    ):
        """
        Context manager that logs start + end with duration.
        
        Args:
            component: System component.
            action: Action being performed.
            actor: Who/what is performing the action.
            target: What is being acted upon.
            user_id: User ID if applicable.
            session_id: Session ID if applicable.
            **extra: Additional details.
            
        Yields:
            AuditEvent for modification.
        """
        event = AuditEvent(
            component=component,
            action=action,
            actor=actor,
            status="started",
            target=target,
            details=extra,
            user_id=user_id,
            session_id=session_id,
        )
        self.log(event)
        
        start = time.monotonic()
        try:
            yield event
            elapsed = (time.monotonic() - start) * 1000
            event.status = "completed"
            event.duration_ms = elapsed
            self.log(event)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            event.status = "failed"
            event.duration_ms = elapsed
            event.error = str(e)
            self.log(event)
            raise
    
    def close(self) -> None:
        """Close the current log file."""
        if self._current_file:
            self._current_file.close()
            self._current_file = None


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

audit = AuditLogger()


# ══════════════════════════════════════════════════════════════════════════════
# Decorator
# ══════════════════════════════════════════════════════════════════════════════

def audited(component: str, action: str, actor: str = ""):
    """
    Decorator that wraps any function with audit logging.
    
    Automatically logs start, completion/failure, and duration.
    
    Args:
        component: System component.
        action: Action being performed.
        actor: Who/what is performing the action (defaults to function name).
        
    Returns:
        Decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _actor = actor or func.__qualname__
            start = time.monotonic()
            audit.log_raw(component, action, _actor, "started")
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.monotonic() - start) * 1000
                audit.log_raw(component, action, _actor, "completed", duration_ms=elapsed)
                return result
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                audit.log_raw(component, action, _actor, "failed", error=str(e), duration_ms=elapsed)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _actor = actor or func.__qualname__
            start = time.monotonic()
            audit.log_raw(component, action, _actor, "started")
            try:
                result = func(*args, **kwargs)
                elapsed = (time.monotonic() - start) * 1000
                audit.log_raw(component, action, _actor, "completed", duration_ms=elapsed)
                return result
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                audit.log_raw(component, action, _actor, "failed", error=str(e), duration_ms=elapsed)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

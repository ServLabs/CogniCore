"""
Event Streaming

Real-time event streaming from pipeline to WebSocket clients.
Users see thinking, tool calls, memory recalls, decisions as they happen.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class StreamEvent:
    """
    Event streamed to the client during request processing.
    
    Every step in the pipeline emits a typed event.
    """
    type: str              # event type (thinking, memory_recall, tool_call, etc.)
    text: str              # human-readable description for UI
    stage: str = ""        # pipeline stage: "gate" | "thinking" | "recall" | "execution" | "synthesis"
    details: dict[str, Any] = field(default_factory=dict)  # type-specific payload
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sequence: int = 0      # ordering guarantee within a request
    is_final: bool = False # True for the last event in a request
    
    def to_json(self) -> str:
        """Serialize to JSON for WebSocket transmission."""
        return json.dumps({
            "type": self.type,
            "text": self.text,
            "stage": self.stage,
            "details": self.details,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "is_final": self.is_final,
        }, default=str)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text,
            "stage": self.stage,
            "details": self.details,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "is_final": self.is_final,
        }


class EventStream:
    """
    Bridges the pipeline to the WebSocket.
    
    Thread-safe async queue that pipeline stages emit to,
    and WebSocket handler consumes from.
    """
    
    def __init__(self, session_id: str, conversation_id: str):
        """
        Initialize event stream.
        
        Args:
            session_id: Session ID for this stream.
            conversation_id: Conversation ID for this stream.
        """
        self.session_id = session_id
        self.conversation_id = conversation_id
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._sequence = 0
        self._closed = False
    
    def emit(
        self,
        type: str,
        text: str,
        stage: str = "",
        **details,
    ) -> None:
        """
        Emit an event to the stream.
        
        Called by pipeline stages.
        
        Args:
            type: Event type.
            text: Human-readable description.
            stage: Pipeline stage.
            **details: Additional details.
        """
        if self._closed:
            return
        
        self._sequence += 1
        event = StreamEvent(
            type=type,
            text=text,
            stage=stage,
            details=details,
            sequence=self._sequence,
        )
        self._queue.put_nowait(event)
    
    def emit_thinking(self, text: str, stage: str = "thinking") -> None:
        """Emit a thinking event."""
        self.emit("thinking", text, stage=stage)
    
    def emit_tool_call(
        self,
        tool: str,
        description: str,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit a tool call event."""
        self.emit(
            "tool_call",
            description,
            stage="execution",
            tool=tool,
            params=params or {},
        )
    
    def emit_tool_result(
        self,
        tool: str,
        summary: str,
        duration_ms: float = 0,
    ) -> None:
        """Emit a tool result event."""
        self.emit(
            "tool_result",
            summary,
            stage="execution",
            tool=tool,
            duration_ms=duration_ms,
        )
    
    def emit_memory(
        self,
        text: str,
        source: str = "",
        count: int = 0,
    ) -> None:
        """Emit a memory recall event."""
        self.emit(
            "memory_recall",
            text,
            stage="recall",
            source=source,
            count=count,
        )
    
    def emit_decision(self, text: str) -> None:
        """Emit a decision event."""
        self.emit("decision", text, stage="decision")
    
    def emit_error(self, text: str, recoverable: bool = True) -> None:
        """Emit an error event."""
        self.emit("error", text, stage="", recoverable=recoverable)
    
    def emit_response_chunk(self, text: str) -> None:
        """Emit a response chunk (for streaming final response)."""
        self.emit("response", text, stage="synthesis")
    
    def emit_done(self, summary: Optional[dict[str, Any]] = None) -> None:
        """Signal request complete."""
        event = StreamEvent(
            type="done",
            text="",
            details=summary or {},
            sequence=self._sequence + 1,
            is_final=True,
        )
        self._queue.put_nowait(event)
        self._closed = True
    
    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """
        Async iterator for consuming events.
        
        WebSocket handler uses this.
        """
        while True:
            event = await self._queue.get()
            yield event
            if event.is_final:
                break
    
    @property
    def is_closed(self) -> bool:
        """Check if stream is closed."""
        return self._closed

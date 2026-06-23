"""
Event Streaming

Real-time event streaming from pipeline to WebSocket clients.
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
    """
    type: str
    text: str
    stage: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sequence: int = 0
    is_final: bool = False
    
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
        self.session_id = session_id
        self.conversation_id = conversation_id
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._sequence = 0
        self._closed = False
        self._done_event = asyncio.Event()
    
    def emit(self, type: str, text: str, stage: str = "", **details) -> None:
        """Emit an event to the stream."""
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
    
    def emit_tool_call(self, tool: str, description: str, params: Optional[dict] = None) -> None:
        """Emit a tool call event."""
        self.emit("tool_call", description, stage="execution", tool=tool, params=params or {})
    
    def emit_tool_result(self, tool: str, summary: str, duration_ms: float = 0) -> None:
        """Emit a tool result event."""
        self.emit("tool_result", summary, stage="execution", tool=tool, duration_ms=duration_ms)
    
    def emit_memory(self, text: str, source: str = "", count: int = 0) -> None:
        """Emit a memory recall event."""
        self.emit("memory_recall", text, stage="recall", source=source, count=count)
    
    def emit_decision(self, text: str) -> None:
        """Emit a decision event."""
        self.emit("decision", text, stage="decision")
    
    def emit_error(self, text: str, recoverable: bool = True) -> None:
        """Emit an error event."""
        self.emit("error", text, stage="", recoverable=recoverable)
    
    def emit_response_chunk(self, text: str) -> None:
        """Emit a response chunk."""
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
        self._done_event.set()
    
    async def wait_done(self) -> None:
        """Block until the stream emits a done event."""
        await self._done_event.wait()
    
    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Async iterator for consuming events."""
        while True:
            event = await self._queue.get()
            yield event
            if event.is_final:
                break
    
    @property
    def is_closed(self) -> bool:
        return self._closed

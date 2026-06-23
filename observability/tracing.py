"""
Distributed Tracing

Lightweight tracing inspired by OpenTelemetry / Arize.
Every pipeline invocation creates a Trace containing Spans.

A Span represents a single operation (LLM call, memory fetch, tool execution, etc.)
with timing, metadata, and parent-child relationships.

Usage:
    from observability.tracing import Tracer, get_tracer

    tracer = get_tracer()
    async with tracer.start_trace("pipeline", user_id="u1") as trace:
        async with trace.span("gate", component="response") as s:
            result = await gate.classify(msg)
            s.set_output({"is_gate": result.is_gate})
        
        async with trace.span("llm_call", component="genai", target="gpt-4o") as s:
            response = await genai.ask(payload)
            s.set_output({"tokens": len(response)})
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpanData:
    """Immutable span record for storage."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str             # "gate", "think", "recall", "decision", "llm_call", "tool_call", etc.
    component: str             # "response", "genai", "memory", "connector", etc.
    target: str                # what was acted on: model name, tool name, memory type
    start_ts: datetime
    end_ts: Optional[datetime] = None
    duration_ms: float = 0.0
    status: str = "ok"         # "ok", "error"
    input_summary: Optional[str] = None   # truncated input for debugging
    output_summary: Optional[str] = None  # truncated output for debugging
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_tuple(self) -> tuple:
        """Convert to tuple for DuckDB insert."""
        import json
        return (
            self.trace_id,
            self.span_id,
            self.parent_span_id,
            self.operation,
            self.component,
            self.target,
            self.start_ts,
            self.end_ts,
            self.duration_ms,
            self.status,
            self.input_summary,
            self.output_summary,
            json.dumps(self.metadata) if self.metadata else None,
            self.error,
        )


@dataclass
class TraceData:
    """Immutable trace record for storage."""
    trace_id: str
    start_ts: datetime
    end_ts: Optional[datetime] = None
    total_duration_ms: float = 0.0
    operation: str = "pipeline"    # top-level operation name
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    status: str = "ok"
    span_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_tuple(self) -> tuple:
        """Convert to tuple for DuckDB insert."""
        import json
        return (
            self.trace_id,
            self.start_ts,
            self.end_ts,
            self.total_duration_ms,
            self.operation,
            self.user_id,
            self.session_id,
            self.conversation_id,
            self.status,
            self.span_count,
            json.dumps(self.metadata) if self.metadata else None,
            self.error,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Span Context Manager
# ══════════════════════════════════════════════════════════════════════════════

class Span:
    """
    A single timed operation within a trace.

    Use as async context manager. Automatically captures duration.
    """

    def __init__(
        self,
        trace_id: str,
        operation: str,
        component: str = "",
        target: str = "",
        parent_span_id: Optional[str] = None,
    ):
        self.span_id = str(uuid.uuid4())
        self.trace_id = trace_id
        self.operation = operation
        self.component = component
        self.target = target
        self.parent_span_id = parent_span_id
        self._start_time: float = 0.0
        self._start_ts: datetime = datetime.now(timezone.utc)
        self.duration_ms: float = 0.0
        self.status: str = "ok"
        self.error: Optional[str] = None
        self.metadata: dict[str, Any] = {}
        self._input_summary: Optional[str] = None
        self._output_summary: Optional[str] = None

    def set_input(self, summary: str) -> None:
        """Set truncated input for debugging (max 500 chars)."""
        self._input_summary = summary[:500] if summary else None

    def set_output(self, data: Any) -> None:
        """Set output metadata and summary."""
        if isinstance(data, dict):
            self.metadata.update(data)
            self._output_summary = str(data)[:500]
        elif isinstance(data, str):
            self._output_summary = data[:500]

    def set_metadata(self, **kwargs: Any) -> None:
        """Set additional metadata."""
        self.metadata.update(kwargs)

    async def __aenter__(self) -> "Span":
        self._start_time = time.perf_counter()
        self._start_ts = datetime.now(timezone.utc)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration_ms = (time.perf_counter() - self._start_time) * 1000
        if exc_type:
            self.status = "error"
            self.error = str(exc_val) if exc_val else str(exc_type)

    def to_data(self) -> SpanData:
        """Convert to storage-ready SpanData."""
        return SpanData(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            operation=self.operation,
            component=self.component,
            target=self.target,
            start_ts=self._start_ts,
            end_ts=datetime.now(timezone.utc),
            duration_ms=self.duration_ms,
            status=self.status,
            input_summary=self._input_summary,
            output_summary=self._output_summary,
            metadata=self.metadata,
            error=self.error,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Trace Context Manager
# ══════════════════════════════════════════════════════════════════════════════

class Trace:
    """
    A complete trace (one per pipeline invocation).

    Collects spans and writes them all to storage on completion.
    """

    def __init__(
        self,
        operation: str = "pipeline",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ):
        self.trace_id = str(uuid.uuid4())
        self.operation = operation
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.spans: list[Span] = []
        self._start_time: float = 0.0
        self._start_ts: datetime = datetime.now(timezone.utc)
        self.total_duration_ms: float = 0.0
        self.status: str = "ok"
        self.error: Optional[str] = None
        self.metadata: dict[str, Any] = {}
        self._current_span: Optional[Span] = None

    def span(
        self,
        operation: str,
        component: str = "",
        target: str = "",
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """Create a new span within this trace."""
        s = Span(
            trace_id=self.trace_id,
            operation=operation,
            component=component,
            target=target,
            parent_span_id=parent_span_id or (self._current_span.span_id if self._current_span else None),
        )
        self.spans.append(s)
        return s

    def set_metadata(self, **kwargs: Any) -> None:
        """Set trace-level metadata."""
        self.metadata.update(kwargs)

    async def __aenter__(self) -> "Trace":
        self._start_time = time.perf_counter()
        self._start_ts = datetime.now(timezone.utc)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.total_duration_ms = (time.perf_counter() - self._start_time) * 1000
        if exc_type:
            self.status = "error"
            self.error = str(exc_val) if exc_val else str(exc_type)

    def to_data(self) -> TraceData:
        """Convert to storage-ready TraceData."""
        return TraceData(
            trace_id=self.trace_id,
            start_ts=self._start_ts,
            end_ts=datetime.now(timezone.utc),
            total_duration_ms=self.total_duration_ms,
            operation=self.operation,
            user_id=self.user_id,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            status=self.status,
            span_count=len(self.spans),
            metadata=self.metadata,
            error=self.error,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tracer (singleton entry point)
# ══════════════════════════════════════════════════════════════════════════════

class Tracer:
    """
    Central tracer — creates traces and flushes them to storage.

    Storage is the Analytics DuckDB (traces + spans tables).
    """

    def __init__(self):
        self._buffer: list[tuple[TraceData, list[SpanData]]] = []
        self._initialized = False

    async def _init_tables(self) -> None:
        """Create traces/spans tables in DuckDB if not exist."""
        if self._initialized:
            return

        import duckdb
        from pathlib import Path

        db_path = str(config.paths.analytics_db)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = duckdb.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id VARCHAR PRIMARY KEY,
                    start_ts TIMESTAMP NOT NULL,
                    end_ts TIMESTAMP,
                    total_duration_ms DOUBLE NOT NULL,
                    operation VARCHAR NOT NULL,
                    user_id VARCHAR,
                    session_id VARCHAR,
                    conversation_id VARCHAR,
                    status VARCHAR NOT NULL DEFAULT 'ok',
                    span_count INTEGER NOT NULL DEFAULT 0,
                    metadata JSON,
                    error VARCHAR
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    trace_id VARCHAR NOT NULL,
                    span_id VARCHAR NOT NULL,
                    parent_span_id VARCHAR,
                    operation VARCHAR NOT NULL,
                    component VARCHAR NOT NULL,
                    target VARCHAR,
                    start_ts TIMESTAMP NOT NULL,
                    end_ts TIMESTAMP,
                    duration_ms DOUBLE NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'ok',
                    input_summary VARCHAR,
                    output_summary VARCHAR,
                    metadata JSON,
                    error VARCHAR,
                    PRIMARY KEY (trace_id, span_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_start
                ON traces(start_ts)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spans_trace
                ON spans(trace_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spans_operation
                ON spans(operation, start_ts)
            """)
        finally:
            conn.close()

        self._initialized = True

    def start_trace(
        self,
        operation: str = "pipeline",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> "TracerContext":
        """Start a new trace. Use as async context manager."""
        trace = Trace(
            operation=operation,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )
        return TracerContext(self, trace)

    async def flush(self, trace_data: TraceData, span_data: list[SpanData]) -> None:
        """Write a completed trace + spans to DuckDB."""
        await self._init_tables()

        import duckdb

        conn = duckdb.connect(str(config.paths.analytics_db))
        try:
            conn.execute(
                "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                trace_data.to_tuple(),
            )
            if span_data:
                conn.executemany(
                    "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [s.to_tuple() for s in span_data],
                )
        finally:
            conn.close()


class TracerContext:
    """Async context manager that auto-flushes on exit."""

    def __init__(self, tracer: Tracer, trace: Trace):
        self._tracer = tracer
        self._trace = trace

    async def __aenter__(self) -> Trace:
        await self._trace.__aenter__()
        return self._trace

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._trace.__aexit__(exc_type, exc_val, exc_tb)
        # Flush trace + all spans to storage
        trace_data = self._trace.to_data()
        span_data = [s.to_data() for s in self._trace.spans]
        try:
            await self._tracer.flush(trace_data, span_data)
        except Exception:
            pass  # Never let tracing break the pipeline


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_tracer_instance: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the singleton Tracer instance."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = Tracer()
    return _tracer_instance

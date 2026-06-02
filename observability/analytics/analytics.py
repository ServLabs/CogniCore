"""
Analytics Layer

Cross-cutting metrics and performance tracking across the entire agent.
Append-only numerical data for admin/developer dashboards.

This module provides:
- Metric recording (fire-and-forget append)
- Eval score tracking
- Buffered batch writes to DuckDB
- Cached dashboard queries

Key properties:
- Append-only: Every metric is an immutable event
- Numbers-focused: Latencies, counts, rates, scores
- Write-heavy, read-light: Constant appends, periodic dashboard reads
- DuckDB: Columnar, analytical, single-file
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable

from core import config


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Components that can emit metrics
COMPONENTS = frozenset({
    "abm", "pm", "wm", "em", "am", "mm", "sfm", "lfm", "meta",
    "llm", "system", "mml", "recall", "learning",
})

# Eval types
EVAL_TYPES = frozenset({
    "schedule_adherence",
    "search_quality",
    "response_quality",
    "procedure_success",
    "sentiment_accuracy",
    "recall_relevance",
})

# Buffer settings
DEFAULT_BUFFER_SIZE = 100  # Flush after N events
DEFAULT_FLUSH_INTERVAL = 5.0  # Flush every N seconds
CACHE_TTL_SECONDS = 300  # 5 minute cache for reads


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricEvent:
    """A single metric event."""
    ts: datetime
    component: str
    metric_name: str
    metric_value: float
    dimensions: dict[str, Any] = field(default_factory=dict)
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for DuckDB insert."""
        return (
            self.ts,
            self.component,
            self.metric_name,
            self.metric_value,
            json.dumps(self.dimensions) if self.dimensions else None,
        )


@dataclass
class EvalEvent:
    """An evaluation score event."""
    ts: datetime
    eval_type: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for DuckDB insert."""
        return (
            self.ts,
            self.eval_type,
            self.score,
            json.dumps(self.details) if self.details else None,
        )


@dataclass
class CachedResult:
    """Cached query result."""
    data: Any
    cached_at: datetime
    
    def is_valid(self, ttl_seconds: int = CACHE_TTL_SECONDS) -> bool:
        """Check if cache is still valid."""
        age = (datetime.now(timezone.utc) - self.cached_at).total_seconds()
        return age < ttl_seconds


# ══════════════════════════════════════════════════════════════════════════════
# Analytics Manager
# ══════════════════════════════════════════════════════════════════════════════

class Analytics:
    """
    Analytics layer for metrics and eval tracking.
    
    Provides:
    - Fire-and-forget metric recording
    - Buffered batch writes to DuckDB
    - Cached dashboard queries
    
    Attributes:
        db_path: Path to DuckDB database.
        buffer_size: Events to buffer before flush.
        flush_interval: Seconds between auto-flushes.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
    ):
        """
        Initialize Analytics.
        
        Args:
            db_path: Path to DuckDB database.
            buffer_size: Events to buffer before flush.
            flush_interval: Seconds between auto-flushes.
        """
        self.db_path = db_path or str(config.paths.analytics_db)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Buffers
        self._metric_buffer: list[MetricEvent] = []
        self._eval_buffer: list[EvalEvent] = []
        self._buffer_lock = threading.Lock()
        
        # Cache
        self._query_cache: dict[str, CachedResult] = {}
        self._cache_lock = threading.Lock()
        
        # Background flush
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        
        self._initialized = False
    
    # ── Initialization ──
    
    async def _init(self) -> None:
        """Initialize DuckDB tables."""
        if self._initialized:
            return
        
        import duckdb
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = duckdb.connect(self.db_path)
        try:
            # Metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    ts TIMESTAMP NOT NULL,
                    component VARCHAR NOT NULL,
                    metric_name VARCHAR NOT NULL,
                    metric_value DOUBLE NOT NULL,
                    dimensions JSON
                )
            """)
            
            # Eval scores table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_scores (
                    ts TIMESTAMP NOT NULL,
                    eval_type VARCHAR NOT NULL,
                    score DOUBLE NOT NULL,
                    details JSON
                )
            """)
            
            # Indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_ts 
                ON metrics(ts)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_component 
                ON metrics(component, metric_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval_ts 
                ON eval_scores(ts)
            """)
            
        finally:
            conn.close()
        
        self._initialized = True
    
    async def start(self) -> None:
        """Start background flush task."""
        await self._init()
        
        if self._running:
            return
        
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self) -> None:
        """Stop background flush and flush remaining buffer."""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush()
    
    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush()
    
    async def _flush(self) -> None:
        """Flush buffers to DuckDB."""
        metrics_to_flush = []
        evals_to_flush = []
        
        with self._buffer_lock:
            if self._metric_buffer:
                metrics_to_flush = self._metric_buffer.copy()
                self._metric_buffer.clear()
            if self._eval_buffer:
                evals_to_flush = self._eval_buffer.copy()
                self._eval_buffer.clear()
        
        if not metrics_to_flush and not evals_to_flush:
            return
        
        import duckdb
        
        conn = duckdb.connect(self.db_path)
        try:
            if metrics_to_flush:
                conn.executemany(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                    [m.to_tuple() for m in metrics_to_flush]
                )
            
            if evals_to_flush:
                conn.executemany(
                    "INSERT INTO eval_scores VALUES (?, ?, ?, ?)",
                    [e.to_tuple() for e in evals_to_flush]
                )
        finally:
            conn.close()
    
    # ── Public API: Recording ──
    
    async def record_metric(
        self,
        component: str,
        metric_name: str,
        metric_value: float,
        dimensions: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record a metric event.
        
        Fire-and-forget - returns immediately, buffers for batch write.
        
        Args:
            component: Source component (pm, wm, em, etc.).
            metric_name: Metric name (e.g., search_latency_ms).
            metric_value: Numerical value.
            dimensions: Optional tags/dimensions.
        """
        await self._init()
        
        event = MetricEvent(
            ts=datetime.now(timezone.utc),
            component=component,
            metric_name=metric_name,
            metric_value=metric_value,
            dimensions=dimensions or {},
        )
        
        with self._buffer_lock:
            self._metric_buffer.append(event)
            
            # Flush if buffer full
            if len(self._metric_buffer) >= self.buffer_size:
                asyncio.create_task(self._flush())
    
    async def record_eval(
        self,
        eval_type: str,
        score: float,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record an evaluation score.
        
        Args:
            eval_type: Type of evaluation.
            score: Score (0.0-1.0).
            details: Optional breakdown/details.
        """
        await self._init()
        
        event = EvalEvent(
            ts=datetime.now(timezone.utc),
            eval_type=eval_type,
            score=score,
            details=details or {},
        )
        
        with self._buffer_lock:
            self._eval_buffer.append(event)
            
            if len(self._eval_buffer) >= self.buffer_size:
                asyncio.create_task(self._flush())
    
    # ── Public API: Querying ──
    
    async def query_metrics(
        self,
        sql: str,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Execute a SQL query on metrics.
        
        Results are cached for 5 minutes by default.
        
        Args:
            sql: SQL query string.
            use_cache: Whether to use cached results.
            
        Returns:
            List of result rows as dicts.
        """
        await self._init()
        
        # Check cache
        cache_key = sql
        if use_cache:
            with self._cache_lock:
                if cache_key in self._query_cache:
                    cached = self._query_cache[cache_key]
                    if cached.is_valid():
                        return cached.data
        
        import duckdb
        
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            result = conn.execute(sql).fetchall()
            columns = [desc[0] for desc in conn.description]
            rows = [dict(zip(columns, row)) for row in result]
        finally:
            conn.close()
        
        # Cache result
        with self._cache_lock:
            self._query_cache[cache_key] = CachedResult(
                data=rows,
                cached_at=datetime.now(timezone.utc),
            )
        
        return rows
    
    async def get_dashboard_stats(
        self,
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get dashboard statistics.
        
        Args:
            hours: Hours of data to include.
            
        Returns:
            Dashboard stats dict.
        """
        await self._init()
        
        import duckdb
        
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            # Metrics summary by component
            metrics_by_component = conn.execute(f"""
                SELECT 
                    component,
                    metric_name,
                    COUNT(*) as count,
                    AVG(metric_value) as avg_value,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value
                FROM metrics 
                WHERE ts > now() - INTERVAL '{hours} hours'
                GROUP BY component, metric_name
                ORDER BY component, metric_name
            """).fetchall()
            
            # Eval scores summary
            eval_summary = conn.execute(f"""
                SELECT 
                    eval_type,
                    COUNT(*) as count,
                    AVG(score) as avg_score,
                    MIN(score) as min_score,
                    MAX(score) as max_score
                FROM eval_scores 
                WHERE ts > now() - INTERVAL '{hours} hours'
                GROUP BY eval_type
            """).fetchall()
            
            # Total counts
            total_metrics = conn.execute(f"""
                SELECT COUNT(*) FROM metrics 
                WHERE ts > now() - INTERVAL '{hours} hours'
            """).fetchone()[0]
            
            total_evals = conn.execute(f"""
                SELECT COUNT(*) FROM eval_scores 
                WHERE ts > now() - INTERVAL '{hours} hours'
            """).fetchone()[0]
            
        finally:
            conn.close()
        
        return {
            "period_hours": hours,
            "total_metrics": total_metrics,
            "total_evals": total_evals,
            "metrics_by_component": [
                {
                    "component": row[0],
                    "metric_name": row[1],
                    "count": row[2],
                    "avg": row[3],
                    "min": row[4],
                    "max": row[5],
                }
                for row in metrics_by_component
            ],
            "eval_summary": [
                {
                    "eval_type": row[0],
                    "count": row[1],
                    "avg_score": row[2],
                    "min_score": row[3],
                    "max_score": row[4],
                }
                for row in eval_summary
            ],
        }
    
    async def get_component_metrics(
        self,
        component: str,
        hours: int = 1,
    ) -> list[dict[str, Any]]:
        """Get metrics for a specific component."""
        return await self.query_metrics(f"""
            SELECT 
                metric_name,
                AVG(metric_value) as avg_value,
                COUNT(*) as count
            FROM metrics 
            WHERE component = '{component}'
              AND ts > now() - INTERVAL '{hours} hours'
            GROUP BY metric_name
        """)
    
    async def get_latency_percentiles(
        self,
        metric_name: str = "search_latency_ms",
        hours: int = 1,
    ) -> dict[str, float]:
        """Get latency percentiles."""
        import duckdb
        
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            result = conn.execute(f"""
                SELECT 
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY metric_value) as p50,
                    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY metric_value) as p90,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY metric_value) as p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY metric_value) as p99
                FROM metrics 
                WHERE metric_name = '{metric_name}'
                  AND ts > now() - INTERVAL '{hours} hours'
            """).fetchone()
        finally:
            conn.close()
        
        return {
            "p50": result[0] or 0,
            "p90": result[1] or 0,
            "p95": result[2] or 0,
            "p99": result[3] or 0,
        }
    
    async def get_eval_trend(
        self,
        eval_type: str,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Get daily eval score trend."""
        return await self.query_metrics(f"""
            SELECT 
                DATE_TRUNC('day', ts) as day,
                AVG(score) as avg_score,
                COUNT(*) as count
            FROM eval_scores 
            WHERE eval_type = '{eval_type}'
              AND ts > now() - INTERVAL '{days} days'
            GROUP BY DATE_TRUNC('day', ts)
            ORDER BY day
        """)
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        with self._cache_lock:
            self._query_cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_analytics_instance: Optional[Analytics] = None


def get_analytics() -> Analytics:
    """
    Get the singleton Analytics instance.
    
    Returns:
        The Analytics instance.
    """
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = Analytics()
    return _analytics_instance


# ── Convenience Functions ──

async def record_metric(
    component: str,
    metric_name: str,
    metric_value: float,
    dimensions: Optional[dict[str, Any]] = None,
) -> None:
    """Record a metric event."""
    await get_analytics().record_metric(
        component=component,
        metric_name=metric_name,
        metric_value=metric_value,
        dimensions=dimensions,
    )


async def record_eval(
    eval_type: str,
    score: float,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Record an evaluation score."""
    await get_analytics().record_eval(
        eval_type=eval_type,
        score=score,
        details=details,
    )


async def query_metrics(sql: str) -> list[dict[str, Any]]:
    """Execute a SQL query on metrics."""
    return await get_analytics().query_metrics(sql)


async def get_dashboard_stats(hours: int = 24) -> dict[str, Any]:
    """Get dashboard statistics."""
    return await get_analytics().get_dashboard_stats(hours)


# ── Metric Recording Helpers ──

async def record_latency(
    component: str,
    operation: str,
    latency_ms: float,
    **dimensions: Any,
) -> None:
    """Record a latency metric."""
    await record_metric(
        component=component,
        metric_name=f"{operation}_latency_ms",
        metric_value=latency_ms,
        dimensions=dimensions,
    )


async def record_count(
    component: str,
    metric_name: str,
    count: int = 1,
    **dimensions: Any,
) -> None:
    """Record a count metric."""
    await record_metric(
        component=component,
        metric_name=metric_name,
        metric_value=float(count),
        dimensions=dimensions,
    )


async def record_rate(
    component: str,
    metric_name: str,
    rate: float,
    **dimensions: Any,
) -> None:
    """Record a rate metric (0.0-1.0)."""
    await record_metric(
        component=component,
        metric_name=metric_name,
        metric_value=rate,
        dimensions=dimensions,
    )


class MetricTimer:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        component: str,
        operation: str,
        **dimensions: Any,
    ):
        self.component = component
        self.operation = operation
        self.dimensions = dimensions
        self.start_time: Optional[float] = None
    
    async def __aenter__(self) -> "MetricTimer":
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            await record_latency(
                self.component,
                self.operation,
                elapsed_ms,
                **self.dimensions,
            )

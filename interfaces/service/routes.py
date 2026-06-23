"""
Service Routes

Developer/service API endpoints for ingestion, metrics, and evals.

Ingestion delegates to helpers.ingestion.IngestionService which stages
data through MML for maintenance processing (like WM stages conversations).
"""

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from config import config
from helpers import get_ingestion_service, csv_to_text, json_to_text
from observability import audit, get_analytics, get_eval_runner
from observability.tracing import get_tracer
from connectors import get_registry


router = APIRouter(tags=["service"])

# SQL-safe identifier pattern (alphanumeric + underscore + hyphen + dot)
_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+$")


def _safe_sql_value(value: str) -> str:
    """
    Validate and escape a user-supplied value for SQL interpolation.
    
    Only allows alphanumeric + underscore + hyphen + dot + colon.
    Raises HTTPException 400 for invalid input.
    """
    if not value or not re.match(r"^[a-zA-Z0-9_.\-: ]+$", value):
        raise HTTPException(status_code=400, detail=f"Invalid parameter value: {value!r}")
    return value.replace("'", "''")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Models
# ══════════════════════════════════════════════════════════════════════════════

class MetricsSummary(BaseModel):
    """Overall metrics summary."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    memory_facts: int = 0
    memory_documents: int = 0
    active_sessions: int = 0
    uptime_hours: float = 0.0


class MemoryMetrics(BaseModel):
    """Memory usage statistics."""
    sfm_facts: int = 0
    lfm_documents: int = 0
    lfm_chunks: int = 0
    am_nodes: int = 0
    am_edges: int = 0
    wm_conversations: int = 0
    mm_procedures: int = 0
    faiss_vectors: dict[str, int] = {}


class LLMMetrics(BaseModel):
    """LLM usage and costs."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    calls_by_model: dict[str, int] = {}
    cost_by_model: dict[str, float] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Evals Models
# ══════════════════════════════════════════════════════════════════════════════

class EvalResultResponse(BaseModel):
    """Eval result response."""
    eval_type: str
    score: float
    sample_size: int
    passed: int
    failed: int
    details: dict[str, Any] = {}
    timestamp: str


class RunEvalRequest(BaseModel):
    """Request to run an eval."""
    eval_type: str
    params: dict[str, Any] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Models
# ══════════════════════════════════════════════════════════════════════════════

class TextIngestionRequest(BaseModel):
    """Request body for text ingestion."""
    text: str
    title: str = "untitled"
    domain: str = ""
    source: str = "api_upload"


class FactIngestionRequest(BaseModel):
    """Request body for fact ingestion."""
    facts: list[dict[str, Any]]
    domain: str = ""


class IngestionResponse(BaseModel):
    """Response for ingestion requests."""
    job_id: str
    status: str
    doc_id: Optional[str] = None
    chunks_created: int = 0
    facts_extracted: int = 0
    errors: list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/ingest/text", response_model=IngestionResponse)
async def ingest_text(request: TextIngestionRequest):
    """Ingest plain text. Stages through MML for maintenance processing."""
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required")

    audit.log_raw("interface", "ingest_text", "service", "started", domain=request.domain or "")

    svc = get_ingestion_service()
    result = await svc.ingest_text(
        text=request.text,
        title=request.title,
        domain=request.domain,
        source=request.source,
    )

    if result.status == "failed":
        audit.log_raw("interface", "ingest_text", "service", "failed", error=result.errors[0] if result.errors else "unknown")
        raise HTTPException(status_code=500, detail=result.errors[0] if result.errors else "ingestion failed")

    audit.log_raw("interface", "ingest_text", "service", "completed", chunks=result.chunks_created)
    return IngestionResponse(**result.to_dict())


@router.post("/ingest/file", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
    domain: str = Form(""),
):
    """Upload and ingest a file. Supported: .md, .txt, .csv, .json"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".md", ".txt", ".csv", ".json"}:
        raise HTTPException(status_code=400, detail=f"Unsupported: {ext}")

    audit.log_raw("interface", "ingest_file", "service", "started", filename=file.filename, domain=domain)

    content = await file.read()

    match ext:
        case ".md" | ".txt":
            text = content.decode("utf-8")
        case ".csv":
            text = csv_to_text(content)
        case ".json":
            text = json_to_text(content)
        case _:
            text = content.decode("utf-8")

    svc = get_ingestion_service()
    result = await svc.ingest_text(
        text=text,
        title=file.filename,
        domain=domain or config.domain_config.default_domain,
        source="file_upload",
    )

    if result.status == "failed":
        audit.log_raw("interface", "ingest_file", "service", "failed", filename=file.filename, error=result.errors[0] if result.errors else "unknown")
        raise HTTPException(status_code=500, detail=result.errors[0] if result.errors else "ingestion failed")

    audit.log_raw("interface", "ingest_file", "service", "completed", filename=file.filename, chunks=result.chunks_created)
    return IngestionResponse(**result.to_dict())


@router.post("/ingest/facts", response_model=IngestionResponse)
async def ingest_facts(request: FactIngestionRequest):
    """Stage facts for SFM insertion via maintenance."""
    if not request.facts:
        raise HTTPException(status_code=400, detail="facts array is required")

    audit.log_raw("interface", "ingest_facts", "service", "started", count=len(request.facts), domain=request.domain or "")

    svc = get_ingestion_service()
    result = await svc.ingest_facts(facts=request.facts, domain=request.domain)

    if result.status == "failed":
        audit.log_raw("interface", "ingest_facts", "service", "failed", error=result.errors[0] if result.errors else "unknown")
        raise HTTPException(status_code=500, detail=result.errors[0] if result.errors else "ingestion failed")

    audit.log_raw("interface", "ingest_facts", "service", "completed", count=len(request.facts))
    return IngestionResponse(**result.to_dict())


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/metrics/summary", response_model=MetricsSummary)
async def get_metrics_summary():
    """Get overall metrics summary from analytics."""
    analytics = get_analytics()
    stats = await analytics.get_dashboard_stats(hours=24)

    # Pull LLM-specific counts
    llm_metrics = [m for m in stats.get("metrics_by_component", []) if m["component"] == "llm"]
    total_calls = sum(m["count"] for m in llm_metrics if "call" in m["metric_name"])
    avg_latency = next((m["avg"] for m in llm_metrics if "latency" in m["metric_name"]), 0.0)

    return MetricsSummary(
        total_requests=stats.get("total_metrics", 0),
        total_tokens=0,
        total_cost_usd=0.0,
        avg_latency_ms=avg_latency,
        memory_facts=0,
        memory_documents=0,
        active_sessions=0,
        uptime_hours=0.0,
    )


@router.get("/metrics/memory", response_model=MemoryMetrics)
async def get_memory_metrics():
    """Get memory usage statistics from analytics."""
    analytics = get_analytics()
    mem_metrics = await analytics.get_component_metrics("sfm", hours=24)
    lfm_metrics = await analytics.get_component_metrics("lfm", hours=24)

    return MemoryMetrics(
        sfm_facts=int(next((m["count"] for m in mem_metrics if "fact" in m.get("metric_name", "")), 0)),
        lfm_documents=int(next((m["count"] for m in lfm_metrics if "document" in m.get("metric_name", "")), 0)),
    )


@router.get("/metrics/llm", response_model=LLMMetrics)
async def get_llm_metrics():
    """Get LLM usage and cost metrics from analytics."""
    analytics = get_analytics()
    llm_metrics = await analytics.get_component_metrics("llm", hours=24)

    total_calls = int(sum(m["count"] for m in llm_metrics if "call" in m.get("metric_name", "")))
    avg_latency = next((m["avg_value"] for m in llm_metrics if "latency" in m.get("metric_name", "")), 0.0)

    return LLMMetrics(
        total_calls=total_calls,
        avg_latency_ms=avg_latency,
    )


@router.get("/metrics/connectors")
async def get_connector_health():
    """Get connector health status from registry."""
    registry = get_registry()
    health = await registry.health_check_all()
    status = registry.get_status()

    return {
        "connectors": status["connectors"],
        "health": health,
        "total": status["total"],
        "connected": status["connected"],
    }


@router.get("/metrics/timeseries")
async def get_timeseries(
    metric: str = Query(..., description="Metric name", pattern=r"^[a-zA-Z0-9_.\-]+$"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    interval: str = Query("1h"),
):
    """Get time-series data for a metric from analytics."""
    now = datetime.now(timezone.utc)
    end_time = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else now
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00")) if start else end_time - timedelta(hours=24)

    analytics = get_analytics()
    # metric is regex-validated (alphanumeric + _ . -), safe to interpolate
    data = await analytics.query_metrics(
        f"SELECT ts, metric_value FROM metrics "
        f"WHERE metric_name = '{metric}' "
        f"AND ts >= '{start_time.isoformat()}' "
        f"AND ts <= '{end_time.isoformat()}' "
        f"ORDER BY ts"
    )

    return {
        "metric": metric,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "interval": interval,
        "data": data,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evals Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/evals", response_model=list[EvalResultResponse])
async def list_evals(
    eval_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """List recent eval results (cached 60s in Redis)."""
    analytics = get_analytics()
    
    # Use Redis-cached DuckDB query for eval results
    where = f"WHERE eval_type = '{_safe_sql_value(eval_type)}'" if eval_type else ""
    rows = await analytics.query_metrics(f"""
        SELECT ts, eval_type, score, sample_size, passed, failed, details
        FROM eval_results
        {where}
        ORDER BY ts DESC
        LIMIT {limit}
    """)
    
    return [
        EvalResultResponse(
            eval_type=r["eval_type"],
            score=r["score"],
            sample_size=r["sample_size"],
            passed=r["passed"],
            failed=r["failed"],
            details=r.get("details") or {},
            timestamp=str(r["ts"]),
        )
        for r in rows
    ]


@router.get("/evals/types")
async def list_eval_types():
    """List all registered eval types."""
    return {"eval_types": get_eval_runner().list_evals()}


@router.get("/evals/summary")
async def get_eval_summary():
    """Get summary of all eval results (cached 60s in Redis)."""
    analytics = get_analytics()
    
    rows = await analytics.query_metrics("""
        SELECT 
            eval_type,
            COUNT(*) as run_count,
            AVG(score) as avg_score,
            MIN(score) as min_score,
            MAX(score) as max_score,
            SUM(passed) as total_passed,
            SUM(failed) as total_failed,
            MAX(ts) as last_run
        FROM eval_results
        GROUP BY eval_type
        ORDER BY eval_type
    """)
    
    return {"eval_types": rows}


@router.post("/evals/run")
async def run_eval(request: RunEvalRequest):
    """Trigger an eval run."""
    runner = get_eval_runner()
    
    if request.eval_type not in runner.list_evals():
        raise HTTPException(status_code=404, detail=f"Unknown eval: {request.eval_type}")
    
    audit.log_raw("interface", "run_eval", "service", "started", target=request.eval_type)
    
    try:
        result = await runner.run_single(request.eval_type)
        audit.log_raw("interface", "run_eval", "service", "completed", target=request.eval_type)
        
        return {
            "eval_type": request.eval_type,
            "status": "completed",
            "result": result.to_dict() if result else None,
        }
    except Exception as e:
        audit.log_raw("interface", "run_eval", "service", "failed", error=str(e))
        return {"eval_type": request.eval_type, "status": "failed", "error": str(e)}


@router.post("/evals/run-all")
async def run_all_evals():
    """Trigger all registered evals."""
    runner = get_eval_runner()
    audit.log_raw("interface", "run_all_evals", "service", "started")
    
    results = await runner.run_all()
    
    audit.log_raw(
        "interface", "run_all_evals", "service", "completed",
        details={"count": len(results)},
    )
    
    return {
        "status": "completed",
        "results": [r.to_dict() for r in results],
        "total": len(results),
    }


@router.get("/evals/trend")
async def get_eval_trend(
    eval_type: str = Query(..., description="Eval type to get trend for"),
    days: int = Query(7, ge=1, le=90),
):
    """Get daily eval score trend for a specific eval type (cached 60s)."""
    analytics = get_analytics()
    
    safe_type = _safe_sql_value(eval_type)
    return await analytics.query_metrics(f"""
        SELECT 
            DATE_TRUNC('day', ts) as day,
            AVG(score) as avg_score,
            MIN(score) as min_score,
            MAX(score) as max_score,
            COUNT(*) as run_count,
            AVG(sample_size) as avg_sample_size
        FROM eval_results
        WHERE eval_type = '{safe_type}'
          AND ts > now() - INTERVAL '{days} days'
        GROUP BY DATE_TRUNC('day', ts)
        ORDER BY day
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Traces & Spans Routes (Arize-style observability)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/metrics/traces")
async def list_traces(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(50, ge=1, le=500),
    status: Optional[str] = Query(None, description="Filter by status: ok, error"),
    user_id: Optional[str] = Query(None),
):
    """
    List recent traces with summary info.
    
    Each trace represents one pipeline invocation with timing breakdown.
    Cached in Redis for 60s.
    """
    analytics = get_analytics()
    
    where_clauses = [f"start_ts > now() - INTERVAL '{hours} hours'"]
    if status:
        where_clauses.append(f"status = '{_safe_sql_value(status)}'")
    if user_id:
        where_clauses.append(f"user_id = '{_safe_sql_value(user_id)}'")
    
    where = " AND ".join(where_clauses)
    
    return await analytics.query_metrics(f"""
        SELECT 
            trace_id,
            start_ts,
            total_duration_ms,
            operation,
            user_id,
            status,
            span_count,
            metadata,
            error
        FROM traces
        WHERE {where}
        ORDER BY start_ts DESC
        LIMIT {limit}
    """)


@router.get("/metrics/traces/{trace_id}")
async def get_trace_detail(trace_id: str):
    """
    Get full trace detail with all spans.
    
    Returns the trace metadata plus every span (operation timing).
    """
    analytics = get_analytics()
    
    safe_trace_id = _safe_sql_value(trace_id)
    traces = await analytics.query_metrics(f"""
        SELECT * FROM traces WHERE trace_id = '{safe_trace_id}'
    """)
    if not traces:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    spans = await analytics.query_metrics(f"""
        SELECT 
            span_id,
            parent_span_id,
            operation,
            component,
            target,
            start_ts,
            end_ts,
            duration_ms,
            status,
            input_summary,
            output_summary,
            metadata,
            error
        FROM spans
        WHERE trace_id = '{safe_trace_id}'
        ORDER BY start_ts
    """)
    
    return {
        "trace": traces[0],
        "spans": spans,
    }


@router.get("/metrics/latency")
async def get_latency_breakdown(
    hours: int = Query(1, ge=1, le=24),
    operation: Optional[str] = Query(None, description="Filter by span operation"),
):
    """
    Get latency percentiles (p50, p90, p95, p99) by operation.
    
    Shows how long each pipeline step takes: gate, think, recall,
    decision, execution, synthesis, llm_call, tool_call.
    """
    analytics = get_analytics()
    
    where = f"start_ts > now() - INTERVAL '{hours} hours'"
    if operation:
        where += f" AND operation = '{_safe_sql_value(operation)}'"
    
    return await analytics.query_metrics(f"""
        SELECT 
            operation,
            component,
            COUNT(*) as call_count,
            AVG(duration_ms) as avg_ms,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) as p50_ms,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY duration_ms) as p90_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_ms,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_ms,
            MIN(duration_ms) as min_ms,
            MAX(duration_ms) as max_ms
        FROM spans
        WHERE {where}
        GROUP BY operation, component
        ORDER BY avg_ms DESC
    """)


@router.get("/metrics/costs")
async def get_cost_breakdown(
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get LLM cost breakdown by model.
    
    Shows total spend, call count, avg cost per call, grouped by model.
    """
    analytics = get_analytics()
    
    return await analytics.query_metrics(f"""
        SELECT 
            dimensions->>'model' as model,
            COUNT(*) as call_count,
            SUM(metric_value) as total_cost_usd,
            AVG(metric_value) as avg_cost_per_call,
            MAX(metric_value) as max_cost_single_call
        FROM metrics
        WHERE metric_name = 'estimated_cost_usd'
          AND ts > now() - INTERVAL '{hours} hours'
        GROUP BY dimensions->>'model'
        ORDER BY total_cost_usd DESC
    """)


@router.get("/metrics/throughput")
async def get_throughput(
    hours: int = Query(1, ge=1, le=24),
    interval_minutes: int = Query(5, ge=1, le=60),
):
    """
    Get request throughput over time.
    
    Returns requests-per-interval for the pipeline and LLM calls.
    """
    analytics = get_analytics()
    
    return await analytics.query_metrics(f"""
        SELECT 
            time_bucket(INTERVAL '{interval_minutes} minutes', start_ts) as bucket,
            COUNT(*) as trace_count,
            AVG(total_duration_ms) as avg_duration_ms,
            SUM(span_count) as total_spans
        FROM traces
        WHERE start_ts > now() - INTERVAL '{hours} hours'
        GROUP BY bucket
        ORDER BY bucket
    """)


@router.get("/metrics/errors")
async def get_error_breakdown(
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get error breakdown by operation/component.
    
    Shows which operations are failing and how often.
    """
    analytics = get_analytics()
    
    return await analytics.query_metrics(f"""
        SELECT 
            operation,
            component,
            target,
            COUNT(*) as error_count,
            error
        FROM spans
        WHERE status = 'error'
          AND start_ts > now() - INTERVAL '{hours} hours'
        GROUP BY operation, component, target, error
        ORDER BY error_count DESC
        LIMIT 50
    """)


@router.get("/metrics/pipeline/stats")
async def get_pipeline_stats(
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get high-level pipeline statistics.
    
    Includes: total requests, success rate, avg duration,
    strategy distribution, span-level avg timings.
    """
    analytics = get_analytics()
    
    # Trace-level stats
    trace_stats = await analytics.query_metrics(f"""
        SELECT 
            COUNT(*) as total_traces,
            COUNT(*) FILTER (WHERE status = 'ok') as successful,
            COUNT(*) FILTER (WHERE status = 'error') as failed,
            AVG(total_duration_ms) as avg_duration_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_duration_ms) as p95_duration_ms,
            AVG(span_count) as avg_spans_per_trace
        FROM traces
        WHERE start_ts > now() - INTERVAL '{hours} hours'
    """)
    
    # Span-level averages by operation
    span_avgs = await analytics.query_metrics(f"""
        SELECT 
            operation,
            AVG(duration_ms) as avg_ms,
            COUNT(*) as count
        FROM spans
        WHERE start_ts > now() - INTERVAL '{hours} hours'
        GROUP BY operation
        ORDER BY avg_ms DESC
    """)
    
    # Strategy distribution
    strategy_dist = await analytics.query_metrics(f"""
        SELECT 
            metadata->>'strategy' as strategy,
            COUNT(*) as count
        FROM traces
        WHERE start_ts > now() - INTERVAL '{hours} hours'
          AND metadata->>'strategy' IS NOT NULL
        GROUP BY metadata->>'strategy'
    """)
    
    return {
        "period_hours": hours,
        "summary": trace_stats[0] if trace_stats else {},
        "avg_by_operation": span_avgs,
        "strategy_distribution": strategy_dist,
    }

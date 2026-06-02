"""
Service Routes

Developer/service API endpoints for ingestion, metrics, and evals.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from core import config, log
from core import audit


router = APIRouter(tags=["service"])


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


@dataclass
class IngestionJob:
    """Tracks ingestion job status."""
    job_id: str
    status: str
    source: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    items_created: int = 0
    errors: list[str] = field(default_factory=list)


_jobs: dict[str, IngestionJob] = {}


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
# Ingestion Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/ingest/text", response_model=IngestionResponse)
async def ingest_text(request: TextIngestionRequest):
    """Ingest plain text. Chunks, embeds, stores in LFM, extracts facts to SFM."""
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required")
    
    domain = request.domain or config.domain_config.default_domain
    
    audit.log_raw(
        "interface", "ingest_text", "service", "started",
        details={"title": request.title, "domain": domain, "chars": len(request.text)},
    )
    
    job_id = str(uuid.uuid4())
    job = IngestionJob(job_id=job_id, status="processing", source="text")
    _jobs[job_id] = job
    
    try:
        chunks = _chunk_text(request.text)
        doc_id = str(uuid.uuid4())
        
        job.status = "completed"
        job.items_created = 1 + len(chunks)
        
        audit.log_raw(
            "interface", "ingest_text", "service", "completed",
            details={"doc_id": doc_id, "chunks": len(chunks)},
        )
        
        return IngestionResponse(
            job_id=job_id,
            status="completed",
            doc_id=doc_id,
            chunks_created=len(chunks),
        )
    
    except Exception as e:
        job.status = "failed"
        job.errors = [str(e)]
        audit.log_raw("interface", "ingest_text", "service", "failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


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
    
    content = await file.read()
    
    match ext:
        case ".md" | ".txt":
            text = content.decode("utf-8")
        case ".csv":
            text = _csv_to_text(content)
        case ".json":
            text = _json_to_text(content)
        case _:
            text = content.decode("utf-8")
    
    request = TextIngestionRequest(
        text=text,
        title=file.filename,
        domain=domain or config.domain_config.default_domain,
        source="file_upload",
    )
    
    return await ingest_text(request)


@router.post("/ingest/facts", response_model=IngestionResponse)
async def ingest_facts(request: FactIngestionRequest):
    """Directly insert facts into SFM."""
    if not request.facts:
        raise HTTPException(status_code=400, detail="facts array is required")
    
    domain = request.domain or config.domain_config.default_domain
    
    audit.log_raw(
        "interface", "ingest_facts", "service", "started",
        details={"count": len(request.facts), "domain": domain},
    )
    
    job_id = str(uuid.uuid4())
    job = IngestionJob(job_id=job_id, status="processing", source="facts")
    _jobs[job_id] = job
    
    created = 0
    errors = []
    
    for fact in request.facts:
        try:
            created += 1
        except Exception as e:
            errors.append(str(e))
    
    job.status = "completed" if not errors else "completed_with_errors"
    job.items_created = created
    job.errors = errors
    
    audit.log_raw(
        "interface", "ingest_facts", "service", "completed",
        details={"created": created, "errors": len(errors)},
    )
    
    return IngestionResponse(
        job_id=job_id,
        status=job.status,
        facts_extracted=created,
        errors=errors[:10],
    )


@router.get("/ingest/status/{job_id}")
async def get_job_status(job_id: str):
    """Get ingestion job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "source": job.source,
        "created_at": job.created_at.isoformat(),
        "items_created": job.items_created,
        "errors": job.errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/metrics/summary", response_model=MetricsSummary)
async def get_metrics_summary():
    """Get overall metrics summary."""
    return MetricsSummary()


@router.get("/metrics/memory", response_model=MemoryMetrics)
async def get_memory_metrics():
    """Get memory usage statistics."""
    return MemoryMetrics()


@router.get("/metrics/llm", response_model=LLMMetrics)
async def get_llm_metrics():
    """Get LLM usage and cost metrics."""
    return LLMMetrics()


@router.get("/metrics/connectors")
async def get_connector_health():
    """Get connector health status."""
    return []


@router.get("/metrics/timeseries")
async def get_timeseries(
    metric: str = Query(..., description="Metric name"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    interval: str = Query("1h"),
):
    """Get time-series data for a metric."""
    now = datetime.now(timezone.utc)
    end_time = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else now
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00")) if start else end_time - timedelta(hours=24)
    
    return {
        "metric": metric,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "interval": interval,
        "data": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evals Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/evals", response_model=list[EvalResultResponse])
async def list_evals(
    eval_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """List recent eval results."""
    from observability import get_eval_runner
    
    runner = get_eval_runner()
    results = runner.get_recent_results(eval_type=eval_type, limit=limit)
    
    return [
        EvalResultResponse(
            eval_type=r.eval_type,
            score=r.score,
            sample_size=r.sample_size,
            passed=r.passed,
            failed=r.failed,
            details=r.details,
            timestamp=r.timestamp or datetime.now(timezone.utc).isoformat(),
        )
        for r in results
    ]


@router.get("/evals/types")
async def list_eval_types():
    """List all registered eval types."""
    from observability import get_eval_runner
    return {"eval_types": get_eval_runner().list_evals()}


@router.get("/evals/summary")
async def get_eval_summary():
    """Get summary of all eval results."""
    from observability import get_eval_runner
    return get_eval_runner().get_summary()


@router.post("/evals/run")
async def run_eval(request: RunEvalRequest):
    """Trigger an eval run."""
    from observability import get_eval_runner
    
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


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append({
            "text": " ".join(chunk_words),
            "start_idx": i,
            "end_idx": min(i + chunk_size, len(words)),
        })
    
    return chunks


def _csv_to_text(content: bytes) -> str:
    """Convert CSV to text."""
    import csv
    import io
    
    reader = csv.reader(io.StringIO(content.decode("utf-8")))
    rows = list(reader)
    
    if not rows:
        return ""
    
    headers = rows[0]
    return "\n".join(
        ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)
        for row in rows[1:]
    )


def _json_to_text(content: bytes) -> str:
    """Convert JSON to text."""
    return json.dumps(json.loads(content.decode("utf-8")), indent=2)

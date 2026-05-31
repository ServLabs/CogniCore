"""
Ingestion API Routes

REST endpoints for data ingestion:
- POST /ingest/text - Ingest plain text
- POST /ingest/file - Upload and ingest file
- POST /ingest/facts - Directly insert facts
- GET /ingest/status/{job_id} - Check job status
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core import config, log
from audit import audit


router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# Models
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
    """Tracks the status of an ingestion job."""
    job_id: str
    status: str  # "queued" | "processing" | "completed" | "failed"
    source: str  # "text" | "file" | "facts"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    items_created: int = 0
    errors: list[str] = field(default_factory=list)


# In-memory job storage (would use Redis/DB in production)
_jobs: dict[str, IngestionJob] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/text", response_model=IngestionResponse)
async def ingest_text(request: TextIngestionRequest):
    """
    Ingest plain text.
    
    Chunks, embeds, and stores in LFM. Extracts facts to SFM.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required")
    
    domain = request.domain or config.domain_config.default_domain
    
    audit.log_raw(
        "api",
        "ingest_text",
        "ingestion",
        "started",
        details={"title": request.title, "domain": domain, "chars": len(request.text)},
    )
    log.info(f"Ingestion: text '{request.title}' ({len(request.text)} chars), domain={domain}")
    
    job_id = str(uuid.uuid4())
    job = IngestionJob(job_id=job_id, status="processing", source="text")
    _jobs[job_id] = job
    
    try:
        # Chunk text
        chunks = _chunk_text(request.text, chunk_size=512, overlap=50)
        
        # Store in LFM (would call MML in production)
        doc_id = str(uuid.uuid4())
        
        # Extract facts (would use LLM in production)
        facts = []
        
        job.status = "completed"
        job.items_created = 1 + len(chunks) + len(facts)
        
        audit.log_raw(
            "api",
            "ingest_text",
            "ingestion",
            "completed",
            details={"doc_id": doc_id, "chunks": len(chunks), "facts": len(facts)},
        )
        
        return IngestionResponse(
            job_id=job_id,
            status="completed",
            doc_id=doc_id,
            chunks_created=len(chunks),
            facts_extracted=len(facts),
        )
    
    except Exception as e:
        job.status = "failed"
        job.errors = [str(e)]
        
        audit.log_raw("api", "ingest_text", "ingestion", "failed", error=str(e))
        log.error(f"Ingestion failed: {e}", exc_info=True)
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
    domain: str = Form(""),
):
    """
    Upload and ingest a file.
    
    Supported formats: .md, .txt, .csv, .json
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    
    ext = Path(file.filename).suffix.lower()
    
    if ext not in {".md", ".txt", ".csv", ".json"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: .md, .txt, .csv, .json",
        )
    
    content = await file.read()
    
    # Parse based on file type
    match ext:
        case ".md" | ".txt":
            text = content.decode("utf-8")
        case ".csv":
            text = _csv_to_text(content)
        case ".json":
            text = _json_to_text(content)
        case _:
            text = content.decode("utf-8")
    
    domain = domain or config.domain_config.default_domain
    
    audit.log_raw(
        "api",
        "ingest_file",
        "ingestion",
        "started",
        details={"filename": file.filename, "size": len(content), "domain": domain},
    )
    log.info(f"Ingestion: file '{file.filename}' ({len(content)} bytes), domain={domain}")
    
    # Reuse text ingestion
    request = TextIngestionRequest(
        text=text,
        title=file.filename,
        domain=domain,
        source="file_upload",
    )
    
    return await ingest_text(request)


@router.post("/facts", response_model=IngestionResponse)
async def ingest_facts(request: FactIngestionRequest):
    """
    Directly insert facts into SFM.
    
    For structured data imports.
    """
    if not request.facts:
        raise HTTPException(status_code=400, detail="facts array is required")
    
    domain = request.domain or config.domain_config.default_domain
    
    audit.log_raw(
        "api",
        "ingest_facts",
        "ingestion",
        "started",
        details={"count": len(request.facts), "domain": domain},
    )
    log.info(f"Ingestion: {len(request.facts)} direct facts, domain={domain}")
    
    job_id = str(uuid.uuid4())
    job = IngestionJob(job_id=job_id, status="processing", source="facts")
    _jobs[job_id] = job
    
    created = 0
    errors = []
    
    for fact in request.facts:
        try:
            # Would call MML.sfm_write in production
            created += 1
        except Exception as e:
            errors.append(str(e))
    
    job.status = "completed" if not errors else "completed_with_errors"
    job.items_created = created
    job.errors = errors
    
    audit.log_raw(
        "api",
        "ingest_facts",
        "ingestion",
        "completed",
        details={"created": created, "errors": len(errors)},
    )
    
    return IngestionResponse(
        job_id=job_id,
        status=job.status,
        facts_extracted=created,
        errors=errors[:10],  # Limit error list
    )


@router.get("/status/{job_id}")
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
    """Convert CSV content to text."""
    import csv
    import io
    
    reader = csv.reader(io.StringIO(content.decode("utf-8")))
    rows = list(reader)
    
    if not rows:
        return ""
    
    headers = rows[0]
    text_parts = []
    
    for row in rows[1:]:
        row_text = ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)
        text_parts.append(row_text)
    
    return "\n".join(text_parts)


def _json_to_text(content: bytes) -> str:
    """Convert JSON content to text."""
    data = json.loads(content.decode("utf-8"))
    return json.dumps(data, indent=2)

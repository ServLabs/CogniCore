"""
Data Ingestion API

REST interface for pushing domain data into the agent's memory.
Admins, pipelines, or external systems POST text or files → agent reads,
chunks, embeds, and stores in LFM and SFM.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import config
from audit import audit
from logger import log


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Job
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestionJob:
    """Tracks the status of an ingestion job."""
    job_id: str
    status: str  # "queued" | "processing" | "completed" | "failed"
    source: str  # "text" | "file" | "facts"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    items_created: int = 0
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "items_created": self.items_created,
            "errors": self.errors,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion API
# ══════════════════════════════════════════════════════════════════════════════

class IngestionAPI:
    """
    REST API for data ingestion.
    
    Endpoints:
    - POST /ingest/text - Ingest plain text → LFM + SFM
    - POST /ingest/file - Upload file → parse → LFM + SFM
    - POST /ingest/facts - Directly insert facts → SFM
    - GET /ingest/status/{job_id} - Check job status
    - GET /ingest/health - Health check
    """
    
    def __init__(self):
        """Initialize ingestion API."""
        self._jobs: dict[str, IngestionJob] = {}
    
    def routes(self) -> list:
        """
        Return aiohttp route definitions.
        
        Returns:
            List of route definitions.
        """
        try:
            from aiohttp import web
            return [
                web.post("/ingest/text", self.ingest_text),
                web.post("/ingest/file", self.ingest_file),
                web.post("/ingest/facts", self.ingest_facts),
                web.get("/ingest/status/{job_id}", self.get_status),
                web.get("/ingest/health", self.health),
            ]
        except ImportError:
            return []
    
    async def ingest_text(self, request) -> Any:
        """
        Ingest plain text → chunk → embed → store in LFM + extract facts to SFM.
        
        Request body:
        {
            "text": "...",
            "title": "Document Title",
            "domain": "general",
            "source": "api_upload"
        }
        """
        from aiohttp import web
        
        body = await request.json()
        text = body.get("text", "")
        title = body.get("title", "untitled")
        domain = body.get("domain", config.domain_config.default_domain)
        source = body.get("source", "api_upload")
        
        if not text:
            return web.json_response({"error": "text is required"}, status=400)
        
        audit.log_raw(
            "ingestion",
            "ingest_text",
            "api",
            "started",
            details={"title": title, "domain": domain, "chars": len(text)},
        )
        log.info(f"Ingestion: text '{title}' ({len(text)} chars), domain={domain}")
        
        job_id = str(uuid.uuid4())
        job = IngestionJob(job_id=job_id, status="processing", source="text")
        self._jobs[job_id] = job
        
        try:
            # 1. Chunk text
            chunks = self._chunk_text(text, chunk_size=512, overlap=50)
            
            # 2. Store in LFM (would call MML in production)
            doc_id = str(uuid.uuid4())
            
            # 3. Extract facts (would use LLM in production)
            facts = self._extract_basic_facts(text, domain)
            
            job.status = "completed"
            job.items_created = 1 + len(chunks) + len(facts)
            
            audit.log_raw(
                "ingestion",
                "ingest_text",
                "api",
                "completed",
                details={
                    "doc_id": doc_id,
                    "chunks": len(chunks),
                    "facts": len(facts),
                },
            )
            
            return web.json_response({
                "job_id": job_id,
                "status": "completed",
                "doc_id": doc_id,
                "chunks_created": len(chunks),
                "facts_extracted": len(facts),
            })
        
        except Exception as e:
            job.status = "failed"
            job.errors = [str(e)]
            
            audit.log_raw(
                "ingestion",
                "ingest_text",
                "api",
                "failed",
                error=str(e),
            )
            log.error(f"Ingestion failed: {e}", exc_info=True)
            
            return web.json_response(
                {"error": str(e), "job_id": job_id},
                status=500,
            )
    
    async def ingest_file(self, request) -> Any:
        """
        Upload a file → parse → ingest to LFM + SFM.
        
        Supported formats: .md, .txt, .csv, .json
        """
        from aiohttp import web
        
        reader = await request.multipart()
        field = await reader.next()
        
        if field is None or field.name != "file":
            return web.json_response({"error": "file field required"}, status=400)
        
        filename = field.filename
        content = await field.read(decode=True)
        
        # Parse based on file type
        ext = Path(filename).suffix.lower()
        
        match ext:
            case ".md" | ".txt":
                text = content.decode("utf-8")
            case ".csv":
                text = self._csv_to_text(content)
            case ".json":
                text = self._json_to_text(content)
            case _:
                return web.json_response(
                    {"error": f"Unsupported file type: {ext}. Supported: .md, .txt, .csv, .json"},
                    status=400,
                )
        
        domain = request.query.get("domain", config.domain_config.default_domain)
        
        audit.log_raw(
            "ingestion",
            "ingest_file",
            "api",
            "started",
            details={"filename": filename, "size": len(content), "domain": domain},
        )
        log.info(f"Ingestion: file '{filename}' ({len(content)} bytes), domain={domain}")
        
        # Create a mock request for text ingestion
        class MockRequest:
            async def json(self):
                return {
                    "text": text,
                    "title": filename,
                    "domain": domain,
                    "source": "file_upload",
                }
        
        return await self.ingest_text(MockRequest())
    
    async def ingest_facts(self, request) -> Any:
        """
        Directly insert facts into SFM.
        
        Request body:
        {
            "facts": [
                {"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}
            ],
            "domain": "general"
        }
        """
        from aiohttp import web
        
        body = await request.json()
        facts = body.get("facts", [])
        domain = body.get("domain", config.domain_config.default_domain)
        
        if not facts:
            return web.json_response({"error": "facts array is required"}, status=400)
        
        audit.log_raw(
            "ingestion",
            "ingest_facts",
            "api",
            "started",
            details={"count": len(facts), "domain": domain},
        )
        log.info(f"Ingestion: {len(facts)} direct facts, domain={domain}")
        
        created = 0
        errors = []
        
        for fact in facts:
            try:
                # Would call MML.sfm_write in production
                created += 1
            except Exception as e:
                errors.append({"fact": fact, "error": str(e)})
        
        audit.log_raw(
            "ingestion",
            "ingest_facts",
            "api",
            "completed",
            details={"created": created, "errors": len(errors)},
        )
        
        return web.json_response({
            "created": created,
            "errors": len(errors),
            "error_details": errors if errors else None,
        })
    
    async def get_status(self, request) -> Any:
        """Get ingestion job status."""
        from aiohttp import web
        
        job_id = request.match_info["job_id"]
        job = self._jobs.get(job_id)
        
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        
        return web.json_response(job.to_dict())
    
    async def health(self, request) -> Any:
        """Health check endpoint."""
        from aiohttp import web
        return web.json_response({"status": "ok"})
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Split text into overlapping chunks for embedding.
        
        Args:
            text: Input text.
            chunk_size: Words per chunk.
            overlap: Overlap between chunks.
            
        Returns:
            List of chunk dicts with text and indices.
        """
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
    
    def _extract_basic_facts(
        self,
        text: str,
        domain: str,
    ) -> list[dict[str, Any]]:
        """
        Basic fact extraction (placeholder).
        
        In production, this would use LLM for extraction.
        
        Args:
            text: Input text.
            domain: Domain for the facts.
            
        Returns:
            List of extracted facts.
        """
        # Placeholder - would use LLM in production
        return []
    
    def _csv_to_text(self, content: bytes) -> str:
        """Convert CSV content to text."""
        import csv
        import io
        
        reader = csv.reader(io.StringIO(content.decode("utf-8")))
        rows = list(reader)
        
        if not rows:
            return ""
        
        # Use first row as headers
        headers = rows[0]
        text_parts = []
        
        for row in rows[1:]:
            row_text = ", ".join(
                f"{h}: {v}" for h, v in zip(headers, row) if v
            )
            text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def _json_to_text(self, content: bytes) -> str:
        """Convert JSON content to text."""
        data = json.loads(content.decode("utf-8"))
        return json.dumps(data, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_ingestion_api: Optional[IngestionAPI] = None


def get_ingestion_api() -> IngestionAPI:
    """Get the singleton IngestionAPI instance."""
    global _ingestion_api
    if _ingestion_api is None:
        _ingestion_api = IngestionAPI()
    return _ingestion_api

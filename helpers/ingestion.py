"""
Data Ingestion Service

Framework-agnostic ingestion logic. External data (text, files, facts)
is staged through MML — just like Working Memory stages conversations.

Flow:
    REST API → IngestionService.ingest_text() → stage in MML
    → Maintenance (consolidation) processes staged data
    → Originals stored in LFM, extracted facts in SFM

Called by: interfaces/service/routes.py
"""

import csv
import io
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import config
from logger import log
from observability import audit


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    job_id: str
    status: str  # "staged" | "completed" | "failed"
    doc_id: Optional[str] = None
    chunks_created: int = 0
    facts_extracted: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "doc_id": self.doc_id,
            "chunks_created": self.chunks_created,
            "facts_extracted": self.facts_extracted,
            "errors": self.errors,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Service
# ══════════════════════════════════════════════════════════════════════════════

class IngestionService:
    """
    Core ingestion logic — no web framework dependencies.

    Stages ingested data through MML so maintenance daemons can
    process it properly (just like WM stages conversation data).
    Originals are preserved in LFM; facts are extracted to SFM.
    """

    async def ingest_text(
        self,
        text: str,
        title: str = "untitled",
        domain: str = "",
        source: str = "api_upload",
    ) -> IngestionResult:
        """
        Ingest plain text → chunk → stage through MML.

        The original document is stored in LFM immediately.
        Chunks and fact extraction are staged for maintenance to process.
        """
        from memory import get_mml

        domain = domain or config.domain_config.default_domain
        job_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())

        audit.log_raw(
            "ingestion", "ingest_text", source, "started",
            details={"title": title, "domain": domain, "chars": len(text)},
        )
        log.info("Ingestion: text '%s' (%d chars), domain=%s", title, len(text), domain)

        try:
            mml = get_mml()

            # 1. Chunk the text
            chunks = chunk_text(text)

            # 2. Stage the ingestion for maintenance processing
            #    MML stages it like WM stages conversations — maintenance
            #    (consolidation) will process, extract facts, and commit.
            await mml.stage_proposal("ingestion", {
                "type": "document",
                "doc_id": doc_id,
                "title": title,
                "domain": domain,
                "source": source,
                "text": text,
                "chunks": chunks,
            })

            result = IngestionResult(
                job_id=job_id,
                status="staged",
                doc_id=doc_id,
                chunks_created=len(chunks),
            )

            audit.log_raw(
                "ingestion", "ingest_text", source, "staged",
                details={"doc_id": doc_id, "chunks": len(chunks)},
            )
            return result

        except Exception as e:
            audit.log_raw("ingestion", "ingest_text", source, "failed", error=str(e))
            log.error("Ingestion failed: %s", e, exc_info=True)
            return IngestionResult(job_id=job_id, status="failed", errors=[str(e)])

    async def ingest_facts(
        self,
        facts: list[dict[str, Any]],
        domain: str = "",
    ) -> IngestionResult:
        """
        Directly stage facts for SFM insertion via maintenance.
        """
        from memory import get_mml

        domain = domain or config.domain_config.default_domain
        job_id = str(uuid.uuid4())

        audit.log_raw(
            "ingestion", "ingest_facts", "api", "started",
            details={"count": len(facts), "domain": domain},
        )

        try:
            mml = get_mml()

            await mml.stage_proposal("ingestion", {
                "type": "facts",
                "domain": domain,
                "facts": facts,
            })

            return IngestionResult(
                job_id=job_id,
                status="staged",
                facts_extracted=len(facts),
            )

        except Exception as e:
            audit.log_raw("ingestion", "ingest_facts", "api", "failed", error=str(e))
            return IngestionResult(job_id=job_id, status="failed", errors=[str(e)])


# ══════════════════════════════════════════════════════════════════════════════
# Text Processing Utilities
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text: Input text.
        chunk_size: Words per chunk.
        overlap: Overlapping words between consecutive chunks.

    Returns:
        List of chunk dicts with text and word indices.
    """
    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append({
            "text": " ".join(chunk_words),
            "start_idx": i,
            "end_idx": min(i + chunk_size, len(words)),
        })

    return chunks


def csv_to_text(content: bytes) -> str:
    """Convert CSV bytes to readable text (header: value per row)."""
    reader = csv.reader(io.StringIO(content.decode("utf-8")))
    rows = list(reader)
    if not rows:
        return ""

    headers = rows[0]
    return "\n".join(
        ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)
        for row in rows[1:]
    )


def json_to_text(content: bytes) -> str:
    """Convert JSON bytes to formatted text."""
    return json.dumps(json.loads(content.decode("utf-8")), indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_ingestion_service: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    """Get the singleton IngestionService instance."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service

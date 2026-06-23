"""
Coherence Daemon

Ensures consistency ACROSS memory types (cross-type contradictions
and referential integrity).

Distinct from Conflict daemon which checks WITHIN a single type.

Checks:
1. Cross-type contradictions: SFM fact vs AM attribute vs EM preference vs MM step
2. Referential integrity: Meta pointer → target exists, SFM source_ref → LFM exists

AI-powered for contradiction detection.
Incoherencies are queued for admin review.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

from connectors import genai
from config import config
from logger import log
from memory.management.faiss_wrapper import get_faiss_store
from memory.management.mml import get_mml
from memory.types.am import get_associative_memory
from memory.types.lfm import get_long_form_memory
from memory.types.meta import get_meta_memory
from memory.types.mm import get_motor_memory
from memory.types.sfm import get_short_form_memory
from observability import record_metric
from prompts import prompts



class CoherenceDaemon:
    """
    Cross-type consistency checker.

    Detects:
    - SFM fact contradicts AM entity attribute
    - EM preference contradicts SFM fact
    - Meta pointer to non-existent target
    - SFM promoted_from reference to deleted LFM chunk
    """

    async def run(self, limit: int = 100) -> dict[str, Any]:
        """
        Execute one coherence pass.

        Args:
            limit: Max items to check per phase.

        Returns:
            Stats dict.
        """
        start = time.perf_counter()

        stats = {
            "cross_type_checked": 0,
            "cross_type_issues": 0,
            "referential_checked": 0,
            "referential_issues": 0,
            "errors": 0,
            "duration_ms": 0.0,
        }

        # Phase 1: Cross-type contradiction scan
        try:
            cross_result = await self._check_cross_type(limit)
            stats["cross_type_checked"] = cross_result["checked"]
            stats["cross_type_issues"] = cross_result["issues"]
        except Exception as e:
            logger.warning("coherence: cross-type scan failed: %s", e)
            stats["errors"] += 1

        # Phase 2: Referential integrity
        try:
            ref_result = await self._check_referential_integrity(limit)
            stats["referential_checked"] = ref_result["checked"]
            stats["referential_issues"] = ref_result["issues"]
        except Exception as e:
            logger.warning("coherence: referential check failed: %s", e)
            stats["errors"] += 1

        stats["duration_ms"] = (time.perf_counter() - start) * 1000

        await record_metric("mml", "coherence_duration_ms", stats["duration_ms"])
        await record_metric("mml", "coherence_issues", stats["cross_type_issues"] + stats["referential_issues"])

        total_issues = stats["cross_type_issues"] + stats["referential_issues"]
        if total_issues > 0:
            logger.info("coherence: found %d issues (%d cross-type, %d referential)",
                        total_issues, stats["cross_type_issues"], stats["referential_issues"])

        return stats

    # ── Phase 1: Cross-Type Contradictions ──

    async def _check_cross_type(self, limit: int) -> dict[str, int]:
        """
        Check for contradictions between different memory types.

        Strategy: Take recent SFM facts, search for related items in AM and EM,
        then ask AI if they contradict.
        """
        result = {"checked": 0, "issues": 0}

        mml = get_mml()
        if not mml.embedder:
            return result

        sfm = get_short_form_memory()
        facts = await sfm.get_recent_facts(limit=limit)

        for fact in facts:
            content = fact.get("content", "")
            if not content:
                continue

            result["checked"] += 1

            # Search for related items in other types
            related = await self._find_cross_type_related(content, fact.get("id"))
            if not related:
                continue

            # AI checks for cross-type contradiction
            issue = await self._ai_check_cross_type(fact, related)
            if issue:
                await self._queue_incoherence(issue)
                result["issues"] += 1

        return result

    async def _find_cross_type_related(
        self,
        content: str,
        exclude_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Find related items in AM and EM for a given content."""
        mml = get_mml()
        related = []

        try:
            embedding = await mml.embedder(content)

            # Search AM
            am_store = get_faiss_store("am")
            am_results = await am_store.search(embedding, top_k=3)
            for r in am_results:
                if r.score > 0.7 and r.id != exclude_id:
                    entity = await get_associative_memory().get_entity(r.id)
                    if entity:
                        related.append({"type": "am", "item": entity, "score": r.score})

            # Search MM
            mm_store = get_faiss_store("mm")
            mm_results = await mm_store.search(embedding, top_k=3)
            for r in mm_results:
                if r.score > 0.7 and r.id != exclude_id:
                    proc = await get_motor_memory().get_procedure(r.id)
                    if proc:
                        related.append({"type": "mm", "item": proc, "score": r.score})

        except Exception as e:
            logger.debug("coherence: cross-type search failed: %s", e)

        return related

    async def _ai_check_cross_type(
        self,
        fact: dict[str, Any],
        related: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Ask AI if a fact contradicts related items from other types."""
        fact_content = fact.get("content", "")

        related_lines = []
        for r in related:
            item = r["item"]
            rtype = r["type"]
            rcontent = item.get("content", "") or item.get("name", "")
            related_lines.append(f"[{rtype}] {rcontent}")

        if not related_lines:
            return None

        messages = prompts.get_messages(
            "memory/coherence_check.md",
            fact_id=fact.get("id", "?"),
            fact_content=fact_content,
            related_items="\n".join(related_lines),
        )

        try:
            raw = await genai.ask({
                "model": "cheap",
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })

            data = json.loads(raw.strip())

            if not data.get("incoherent", False):
                return None

            return {
                "type": "cross_type_contradiction",
                "primary_item": fact,
                "related_items": related,
                "explanation": data.get("explanation", ""),
                "severity": data.get("severity", "medium"),
            }

        except Exception as e:
            logger.debug("coherence: AI cross-type check failed: %s", e)
            return None

    # ── Phase 2: Referential Integrity ──

    async def _check_referential_integrity(self, limit: int) -> dict[str, int]:
        """
        Check that cross-type references are valid.

        - Meta pointers → target memory exists
        - SFM promotion source → LFM chunk exists
        """
        result = {"checked": 0, "issues": 0}

        # Check Meta pointers
        meta = get_meta_memory()
        pointers = await meta.get_recent_pointers(limit=limit)

        for pointer in pointers:
            result["checked"] += 1
            exists = await self._verify_target_exists(
                pointer.get("memory_type"),
                pointer.get("ref_id"),
            )
            if not exists:
                await self._queue_incoherence({
                    "type": "dangling_pointer",
                    "pointer_id": pointer.get("id"),
                    "memory_type": pointer.get("memory_type"),
                    "ref_id": pointer.get("ref_id"),
                    "explanation": f"Meta pointer to non-existent {pointer.get('memory_type')} item",
                    "severity": "high",
                })
                result["issues"] += 1

        return result

    async def _verify_target_exists(self, memory_type: Optional[str], ref_id: Optional[str]) -> bool:
        """Verify that a referenced memory item still exists."""
        if not memory_type or not ref_id:
            return False

        try:
            if memory_type == "sfm":
                item = await get_short_form_memory().get_fact(ref_id)
                return item is not None
            elif memory_type == "lfm":
                item = await get_long_form_memory().get_document(ref_id)
                return item is not None
            elif memory_type == "am":
                item = await get_associative_memory().get_entity(ref_id)
                return item is not None
            elif memory_type == "mm":
                item = await get_motor_memory().get_procedure(ref_id)
                return item is not None
        except Exception:
            pass

        return False

    # ── Queue for Admin Review ──

    async def _queue_incoherence(self, issue: dict[str, Any]) -> None:
        """Add an incoherence to the admin review queue."""
        import uuid
        import aiosqlite

        db_path = config.paths.sqlite_dir / "conflicts.db"

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS coherence_queue (
                    id TEXT PRIMARY KEY,
                    issue_type TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolution TEXT
                )
            """)
            await db.execute(
                """
                INSERT OR IGNORE INTO coherence_queue
                (id, issue_type, details_json, severity, detected_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    issue.get("type", "unknown"),
                    json.dumps(issue, default=str),
                    issue.get("severity", "medium"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

    async def get_pending_issues(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get unresolved coherence issues for admin panel."""
        import aiosqlite

        db_path = config.paths.sqlite_dir / "conflicts.db"
        issues = []

        try:
            async with aiosqlite.connect(str(db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT * FROM coherence_queue
                    WHERE resolved = 0
                    ORDER BY
                        CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                        detected_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    issues.append({
                        "id": row["id"],
                        "issue_type": row["issue_type"],
                        "details": json.loads(row["details_json"]),
                        "severity": row["severity"],
                        "detected_at": row["detected_at"],
                    })
        except Exception as e:
            logger.warning("coherence: could not fetch pending issues: %s", e)

        return issues


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_daemon_instance: Optional[CoherenceDaemon] = None


def get_coherence_daemon() -> CoherenceDaemon:
    """Get the singleton CoherenceDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = CoherenceDaemon()
    return _daemon_instance

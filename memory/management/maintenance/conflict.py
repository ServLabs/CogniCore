"""
Conflict Detection Daemon

Background periodic scan that identifies contradictions across memory types.
Detected conflicts are queued for user review via admin panel.

Flow:
1. Scan SFM facts, AM entity attributes, MM procedure steps
2. For each item, FAISS-search for semantically similar items
3. LLM judges if similar items truly contradict
4. Confirmed conflicts → admin review queue
5. User resolves: superseded fact gets valid_until timestamp

AI-powered: FAISS finds candidates, LLM confirms contradictions.
Scope: SFM + AM + MM
"""

import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Optional

from connectors import genai
from config import config
from logger import log
from memory.management.faiss_wrapper import get_faiss_store
from memory.management.mml import get_mml
from memory.types.am import get_associative_memory
from memory.types.meta import get_meta_memory
from memory.types.mm import get_motor_memory
from memory.types.sfm import get_short_form_memory
from observability import record_metric
from prompts import prompts


# Similarity threshold for conflict candidates
CONFLICT_SIMILARITY_THRESHOLD = 0.75


@dataclass
class ConflictRecord:
    """A detected conflict awaiting user resolution."""
    id: str
    item_a: dict[str, Any]  # First conflicting item
    item_b: dict[str, Any]  # Second conflicting item
    memory_type: str  # sfm, am, mm
    conflict_type: str  # contradiction, outdated, ambiguous
    ai_explanation: str  # Why AI thinks these conflict
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution: Optional[str] = None  # superseded, kept_both, deleted

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "item_a": self.item_a,
            "item_b": self.item_b,
            "memory_type": self.memory_type,
            "conflict_type": self.conflict_type,
            "ai_explanation": self.ai_explanation,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


class ConflictDaemon:
    """
    Detects contradictions in stored knowledge.

    Scans SFM facts, AM entity attributes, and MM procedure steps
    for semantic similarity. LLM confirms true contradictions.
    Conflicts are queued for user review.
    """

    async def run(self, limit_per_type: int = 50) -> dict[str, Any]:
        """
        Execute one conflict detection pass.

        Args:
            limit_per_type: Max items to check per memory type.

        Returns:
            Stats dict.
        """
        start = time.perf_counter()

        stats = {
            "scanned": 0,
            "candidates_found": 0,
            "conflicts_confirmed": 0,
            "errors": 0,
            "duration_ms": 0.0,
        }

        # Scan each memory type
        for memory_type in ("sfm", "am", "mm"):
            try:
                result = await self._scan_type(memory_type, limit_per_type)
                stats["scanned"] += result["scanned"]
                stats["candidates_found"] += result["candidates"]
                stats["conflicts_confirmed"] += result["confirmed"]
            except Exception as e:
                logger.warning("conflict: error scanning %s: %s", memory_type, e)
                stats["errors"] += 1

        stats["duration_ms"] = (time.perf_counter() - start) * 1000

        await record_metric("mml", "conflict_duration_ms", stats["duration_ms"])
        await record_metric("mml", "conflict_detected", stats["conflicts_confirmed"])

        if stats["conflicts_confirmed"] > 0:
            logger.info("conflict: detected %d new conflicts", stats["conflicts_confirmed"])

        return stats

    async def _scan_type(self, memory_type: str, limit: int) -> dict[str, int]:
        """Scan a memory type for internal contradictions."""
        items = await self._get_items(memory_type, limit)
        result = {"scanned": len(items), "candidates": 0, "confirmed": 0}

        mml = get_mml()
        if not mml.embedder:
            logger.debug("conflict: no embedder, skipping %s", memory_type)
            return result

        store = get_faiss_store(memory_type)

        for item in items:
            content = item.get("content", "") or item.get("name", "")
            if not content:
                continue

            # Embed and find similar items
            try:
                embedding = await mml.embedder(content)
                similar = await store.search(embedding, top_k=5)
            except Exception:
                continue

            # Filter to candidates above threshold (exclude self)
            candidates = [
                s for s in similar
                if s.score > CONFLICT_SIMILARITY_THRESHOLD
                and s.id != item.get("id")
            ]

            if not candidates:
                continue

            result["candidates"] += len(candidates)

            # LLM confirms if these truly conflict
            for candidate in candidates:
                candidate_item = await self._fetch_item(memory_type, candidate.id)
                if not candidate_item:
                    continue

                conflict = await self._ai_check_conflict(item, candidate_item, memory_type)
                if conflict:
                    await self._queue_conflict(conflict)
                    result["confirmed"] += 1

        return result

    async def _get_items(self, memory_type: str, limit: int) -> list[dict[str, Any]]:
        """Get items from a memory type for scanning."""
        if memory_type == "sfm":
            return await get_short_form_memory().get_recent_facts(limit=limit)
        elif memory_type == "am":
            return await get_associative_memory().get_recent_entities(limit=limit)
        elif memory_type == "mm":
            return await get_motor_memory().get_recent_procedures(limit=limit)
        return []

    async def _fetch_item(self, memory_type: str, item_id: str) -> Optional[dict[str, Any]]:
        """Fetch a specific item by ID."""
        try:
            if memory_type == "sfm":
                return await get_short_form_memory().get_fact(item_id)
            elif memory_type == "am":
                return await get_associative_memory().get_entity(item_id)
            elif memory_type == "mm":
                return await get_motor_memory().get_procedure(item_id)
        except Exception:
            pass
        return None

    async def _ai_check_conflict(
        self,
        item_a: dict[str, Any],
        item_b: dict[str, Any],
        memory_type: str,
    ) -> Optional[ConflictRecord]:
        """
        Ask AI to confirm whether two items truly conflict.

        Returns ConflictRecord if confirmed, None otherwise.
        """
        import uuid

        content_a = item_a.get("content", "") or item_a.get("name", "")
        content_b = item_b.get("content", "") or item_b.get("name", "")

        messages = prompts.get_messages(
            "memory/conflict_detection.md",
            memory_type=memory_type,
            item_a_id=item_a.get("id", "?"),
            content_a=content_a,
            item_b_id=item_b.get("id", "?"),
            content_b=content_b,
        )

        try:
            raw = await genai.ask({
                "model": "cheap",
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })

            data = json.loads(raw.strip())

            if not data.get("conflicts", False):
                return None

            return ConflictRecord(
                id=str(uuid.uuid4()),
                item_a=item_a,
                item_b=item_b,
                memory_type=memory_type,
                conflict_type=data.get("conflict_type", "ambiguous"),
                ai_explanation=data.get("explanation", ""),
            )

        except Exception as e:
            logger.debug("conflict: AI check failed: %s", e)
            return None

    async def _queue_conflict(self, conflict: ConflictRecord) -> None:
        """
        Add a confirmed conflict to the admin review queue.

        Stored in SQLite for persistence across restarts.
        """
        import aiosqlite

        db_path = config.paths.sqlite_dir / "conflicts.db"

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conflict_queue (
                    id TEXT PRIMARY KEY,
                    item_a_json TEXT NOT NULL,
                    item_b_json TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    ai_explanation TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolution TEXT
                )
            """)
            await db.execute(
                """
                INSERT OR IGNORE INTO conflict_queue
                (id, item_a_json, item_b_json, memory_type, conflict_type, ai_explanation, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conflict.id,
                    json.dumps(conflict.item_a),
                    json.dumps(conflict.item_b),
                    conflict.memory_type,
                    conflict.conflict_type,
                    conflict.ai_explanation,
                    conflict.detected_at.isoformat(),
                ),
            )
            await db.commit()

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        winner_id: Optional[str] = None,
    ) -> bool:
        """
        Resolve a conflict from the admin panel.

        Args:
            conflict_id: ID of the conflict record.
            resolution: One of: superseded, kept_both, deleted.
            winner_id: ID of the winning item (for superseded resolution).

        Returns:
            True if resolved successfully.
        """
        import aiosqlite

        db_path = config.paths.sqlite_dir / "conflicts.db"

        async with aiosqlite.connect(str(db_path)) as db:
            # Fetch the conflict
            cursor = await db.execute(
                "SELECT * FROM conflict_queue WHERE id = ? AND resolved = 0",
                (conflict_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return False

            # Apply resolution
            if resolution == "superseded" and winner_id:
                await self._apply_superseded(row, winner_id)
            elif resolution == "deleted":
                loser_id = row[1] if winner_id != row[1] else row[2]  # item_a or item_b
                await self._apply_deletion(json.loads(loser_id), row[3])

            # Mark as resolved
            await db.execute(
                "UPDATE conflict_queue SET resolved = 1, resolution = ? WHERE id = ?",
                (resolution, conflict_id),
            )
            await db.commit()

        return True

    async def _apply_superseded(self, row: Any, winner_id: str) -> None:
        """Mark the loser as superseded with valid_until timestamp."""
        memory_type = row[3]
        item_a = json.loads(row[1])
        item_b = json.loads(row[2])

        loser = item_a if item_a.get("id") != winner_id else item_b

        if memory_type == "sfm":
            await get_short_form_memory().mark_superseded(
                fact_id=loser["id"],
                superseded_by=winner_id,
                valid_until=datetime.now(timezone.utc),
            )
        elif memory_type == "am":
            await get_associative_memory().mark_superseded(
                entity_id=loser["id"],
                superseded_by=winner_id,
            )
        elif memory_type == "mm":
            await get_motor_memory().mark_superseded(
                procedure_id=loser["id"],
                superseded_by=winner_id,
            )

    async def _apply_deletion(self, item: dict[str, Any], memory_type: str) -> None:
        """Hard-delete wrong data from memory."""
        item_id = item.get("id")
        if not item_id:
            return

        if memory_type == "sfm":
            await get_short_form_memory().remove_fact(item_id)
        elif memory_type == "am":
            await get_associative_memory().remove_entity(item_id)
        elif memory_type == "mm":
            await get_motor_memory().remove_procedure(item_id)

        # Clean Meta pointer
        meta = get_meta_memory()
        await meta.remove_pointer_by_ref(item_id)

    async def get_pending_conflicts(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get unresolved conflicts for admin panel display.

        Returns list of conflict records.
        """
        import aiosqlite

        db_path = config.paths.sqlite_dir / "conflicts.db"
        conflicts = []

        try:
            async with aiosqlite.connect(str(db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT * FROM conflict_queue
                    WHERE resolved = 0
                    ORDER BY detected_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    conflicts.append({
                        "id": row["id"],
                        "item_a": json.loads(row["item_a_json"]),
                        "item_b": json.loads(row["item_b_json"]),
                        "memory_type": row["memory_type"],
                        "conflict_type": row["conflict_type"],
                        "ai_explanation": row["ai_explanation"],
                        "detected_at": row["detected_at"],
                    })
        except Exception as e:
            logger.warning("conflict: could not fetch pending conflicts: %s", e)

        return conflicts


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_daemon_instance: Optional[ConflictDaemon] = None


def get_conflict_daemon() -> ConflictDaemon:
    """Get the singleton ConflictDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = ConflictDaemon()
    return _daemon_instance

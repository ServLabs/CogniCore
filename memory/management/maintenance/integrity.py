"""
Integrity Daemon

Self-healing checks and auto-repair for cross-system consistency.
Purely algorithmic — no AI required.

Checks:
1. FAISS ID → no SQLite record (orphan vector) → remove from FAISS
2. SQLite record → no FAISS vector (missing embedding) → re-embed and insert
3. Meta pointer → deleted memory target → remove dangling pointer
4. SFM promotion reference → deleted LFM chunk → clear reference
5. AM entity in relationship → entity doesn't exist → remove orphan edge
"""

import time
from typing import Any, Optional

from logger import log
from memory.management.faiss_wrapper import get_faiss_store
from memory.management.mml import get_mml
from memory.types.am import get_associative_memory
from memory.types.lfm import get_long_form_memory
from memory.types.meta import get_meta_memory
from memory.types.mm import get_motor_memory
from memory.types.sfm import get_short_form_memory
from observability import record_metric


class IntegrityDaemon:
    """
    Self-healing integrity checker.

    Detects and auto-repairs broken cross-system references.
    All algorithmic — no AI calls.
    """

    async def run(self) -> dict[str, Any]:
        """
        Execute one integrity pass.

        Returns:
            Stats dict with repairs per check type.
        """
        start = time.perf_counter()

        stats = {
            "orphan_vectors_removed": 0,
            "missing_embeddings_fixed": 0,
            "dangling_pointers_removed": 0,
            "broken_promotions_cleared": 0,
            "orphan_graph_edges_removed": 0,
            "total_checked": 0,
            "total_repaired": 0,
            "errors": 0,
            "duration_ms": 0.0,
        }

        # Check 1: Orphan FAISS vectors
        try:
            result = await self._check_orphan_vectors()
            stats["orphan_vectors_removed"] = result
            stats["total_repaired"] += result
        except Exception as e:
            logger.warning("integrity: orphan vector check failed: %s", e)
            stats["errors"] += 1

        # Check 2: Missing embeddings
        try:
            result = await self._check_missing_embeddings()
            stats["missing_embeddings_fixed"] = result
            stats["total_repaired"] += result
        except Exception as e:
            logger.warning("integrity: missing embedding check failed: %s", e)
            stats["errors"] += 1

        # Check 3: Dangling Meta pointers
        try:
            result = await self._check_dangling_pointers()
            stats["dangling_pointers_removed"] = result
            stats["total_repaired"] += result
        except Exception as e:
            logger.warning("integrity: dangling pointer check failed: %s", e)
            stats["errors"] += 1

        # Check 4: Broken SFM promotion references
        try:
            result = await self._check_broken_promotions()
            stats["broken_promotions_cleared"] = result
            stats["total_repaired"] += result
        except Exception as e:
            logger.warning("integrity: broken promotion check failed: %s", e)
            stats["errors"] += 1

        # Check 5: Orphan graph edges
        try:
            result = await self._check_orphan_graph_edges()
            stats["orphan_graph_edges_removed"] = result
            stats["total_repaired"] += result
        except Exception as e:
            logger.warning("integrity: orphan graph edge check failed: %s", e)
            stats["errors"] += 1

        stats["duration_ms"] = (time.perf_counter() - start) * 1000

        await record_metric("mml", "integrity_duration_ms", stats["duration_ms"])
        await record_metric("mml", "integrity_repairs", stats["total_repaired"])

        if stats["total_repaired"] > 0:
            logger.info("integrity: repaired %d issues in %.1fms", stats["total_repaired"], stats["duration_ms"])

        return stats

    # ── Check 1: Orphan FAISS vectors ──

    async def _check_orphan_vectors(self) -> int:
        """
        Find FAISS vectors with no corresponding SQLite record.
        Remove orphan vectors from FAISS index.
        """
        repaired = 0
        memory_types = ("sfm", "lfm", "am", "mm", "meta")

        for mtype in memory_types:
            try:
                store = get_faiss_store(mtype)
                orphans = await store.find_orphan_ids()

                for orphan_id in orphans:
                    await store.remove(orphan_id)
                    repaired += 1

                if orphans:
                    logger.debug("integrity: removed %d orphan vectors from %s FAISS", len(orphans), mtype)

            except Exception as e:
                logger.debug("integrity: orphan vector check for %s failed: %s", mtype, e)

        return repaired

    # ── Check 2: Missing embeddings ──

    async def _check_missing_embeddings(self) -> int:
        """
        Find SQLite records with no corresponding FAISS vector.
        Re-embed content and insert into FAISS.
        """
        mml = get_mml()
        if not mml.embedder:
            return 0

        repaired = 0

        # Check SFM
        try:
            sfm = get_short_form_memory()
            store = get_faiss_store("sfm")

            missing = await sfm.find_unembedded_facts()
            for fact in missing:
                content = fact.get("content", "")
                if not content:
                    continue
                embedding = await mml.embedder(content)
                await store.add(fact["id"], embedding)
                repaired += 1

            if missing:
                logger.debug("integrity: re-embedded %d SFM facts", len(missing))
        except Exception as e:
            logger.debug("integrity: SFM embedding check failed: %s", e)

        # Check AM
        try:
            am = get_associative_memory()
            store = get_faiss_store("am")

            missing = await am.find_unembedded_entities()
            for entity in missing:
                name = entity.get("name", "")
                if not name:
                    continue
                embedding = await mml.embedder(name)
                await store.add(entity["id"], embedding)
                repaired += 1

            if missing:
                logger.debug("integrity: re-embedded %d AM entities", len(missing))
        except Exception as e:
            logger.debug("integrity: AM embedding check failed: %s", e)

        # Check MM
        try:
            mm = get_motor_memory()
            store = get_faiss_store("mm")

            missing = await mm.find_unembedded_procedures()
            for proc in missing:
                title = proc.get("title", "")
                if not title:
                    continue
                embedding = await mml.embedder(title)
                await store.add(proc["id"], embedding)
                repaired += 1

            if missing:
                logger.debug("integrity: re-embedded %d MM procedures", len(missing))
        except Exception as e:
            logger.debug("integrity: MM embedding check failed: %s", e)

        return repaired

    # ── Check 3: Dangling Meta pointers ──

    async def _check_dangling_pointers(self) -> int:
        """
        Find Meta pointers that reference non-existent memory items.
        Remove the dangling pointers.
        """
        meta = get_meta_memory()
        repaired = 0

        try:
            pointers = await meta.get_all_pointers()

            for pointer in pointers:
                exists = await self._target_exists(
                    pointer.get("memory_type"),
                    pointer.get("ref_id"),
                )
                if not exists:
                    await meta.remove_pointer(pointer["id"])
                    repaired += 1

            if repaired > 0:
                logger.debug("integrity: removed %d dangling Meta pointers", repaired)
        except Exception as e:
            logger.debug("integrity: Meta pointer check failed: %s", e)

        return repaired

    # ── Check 4: Broken promotion references ──

    async def _check_broken_promotions(self) -> int:
        """
        Find SFM facts with promotion source refs to deleted LFM chunks.
        Clear the broken reference.
        """
        sfm = get_short_form_memory()
        lfm = get_long_form_memory()
        repaired = 0

        try:
            promoted_facts = await sfm.get_promoted_facts()

            for fact in promoted_facts:
                source_chunk_id = fact.get("promoted_from")
                if not source_chunk_id:
                    continue

                chunk_exists = await lfm.chunk_exists(source_chunk_id)
                if not chunk_exists:
                    await sfm.clear_promotion_ref(fact["id"])
                    repaired += 1

            if repaired > 0:
                logger.debug("integrity: cleared %d broken promotion references", repaired)
        except Exception as e:
            logger.debug("integrity: promotion ref check failed: %s", e)

        return repaired

    # ── Check 5: Orphan graph edges ──

    async def _check_orphan_graph_edges(self) -> int:
        """
        Find AM relationship edges where source or target entity doesn't exist.
        Remove orphan edges.
        """
        am = get_associative_memory()
        repaired = 0

        try:
            orphan_edges = await am.find_orphan_edges()
            for edge_id in orphan_edges:
                await am.remove_relationship(edge_id)
                repaired += 1

            if repaired > 0:
                logger.debug("integrity: removed %d orphan graph edges", repaired)
        except Exception as e:
            logger.debug("integrity: graph edge check failed: %s", e)

        return repaired

    # ── Helpers ──

    async def _target_exists(self, memory_type: Optional[str], ref_id: Optional[str]) -> bool:
        """Check if a memory item exists in its respective store."""
        if not memory_type or not ref_id:
            return False

        try:
            if memory_type == "sfm":
                return await get_short_form_memory().fact_exists(ref_id)
            elif memory_type == "lfm":
                return await get_long_form_memory().document_exists(ref_id)
            elif memory_type == "am":
                return await get_associative_memory().entity_exists(ref_id)
            elif memory_type == "mm":
                return await get_motor_memory().procedure_exists(ref_id)
        except Exception:
            pass

        return False


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_daemon_instance: Optional[IntegrityDaemon] = None


def get_integrity_daemon() -> IntegrityDaemon:
    """Get the singleton IntegrityDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = IntegrityDaemon()
    return _daemon_instance

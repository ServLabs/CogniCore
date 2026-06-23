"""
Optimization Daemon

Index and database maintenance — purely algorithmic, no AI.

Tasks:
1. FAISS: Adaptive — incremental cleanup normally, full rebuild if fragmented
2. SQLite: PRAGMA optimize + VACUUM + integrity_check
3. Kuzu: Remove orphan nodes/edges + re-index frequently traversed paths
"""

import time
from typing import Any, Optional

from config import config
from logger import log
from memory.management.faiss_wrapper import get_faiss_store
from memory.types.am import get_associative_memory
from observability import record_metric

# FAISS fragmentation threshold (ratio of deleted to total vectors)
FAISS_FRAGMENTATION_THRESHOLD = 0.25


class OptimizationDaemon:
    """
    Index and database maintenance.

    All algorithmic — no AI calls required.
    """

    async def run(self) -> dict[str, Any]:
        """
        Execute one optimization pass.

        Returns:
            Stats dict with results per subsystem.
        """
        start = time.perf_counter()

        stats = {
            "faiss": {},
            "sqlite": {},
            "kuzu": {},
            "errors": 0,
            "duration_ms": 0.0,
        }

        # FAISS optimization
        try:
            stats["faiss"] = await self._optimize_faiss()
        except Exception as e:
            logger.warning("optimization: FAISS failed: %s", e)
            stats["errors"] += 1

        # SQLite maintenance
        try:
            stats["sqlite"] = await self._optimize_sqlite()
        except Exception as e:
            logger.warning("optimization: SQLite failed: %s", e)
            stats["errors"] += 1

        # Kuzu graph maintenance
        try:
            stats["kuzu"] = await self._optimize_kuzu()
        except Exception as e:
            logger.warning("optimization: Kuzu failed: %s", e)
            stats["errors"] += 1

        stats["duration_ms"] = (time.perf_counter() - start) * 1000

        await record_metric("mml", "optimization_duration_ms", stats["duration_ms"])

        logger.info("optimization: completed in %.1fms", stats["duration_ms"])
        return stats

    # ── FAISS ──

    async def _optimize_faiss(self) -> dict[str, Any]:
        """
        Adaptive FAISS optimization.

        - Incremental: Remove tombstoned vectors
        - Full rebuild: If fragmentation > threshold
        """
        result = {
            "indexes_processed": 0,
            "vectors_cleaned": 0,
            "full_rebuilds": 0,
        }

        memory_types = ("sfm", "lfm", "am", "mm", "meta")

        for mtype in memory_types:
            try:
                store = get_faiss_store(mtype)
                index_stats = await store.get_stats()

                total = index_stats.get("total_vectors", 0)
                deleted = index_stats.get("deleted_vectors", 0)

                if total == 0:
                    continue

                result["indexes_processed"] += 1
                fragmentation = deleted / total if total > 0 else 0

                if fragmentation > FAISS_FRAGMENTATION_THRESHOLD:
                    # Full rebuild — remove deleted, re-cluster
                    cleaned = await store.rebuild()
                    result["vectors_cleaned"] += cleaned
                    result["full_rebuilds"] += 1
                    logger.debug(
                        "optimization: FAISS %s full rebuild — removed %d vectors (%.1f%% fragmented)",
                        mtype, cleaned, fragmentation * 100,
                    )
                elif deleted > 0:
                    # Incremental — just compact deleted entries
                    cleaned = await store.compact()
                    result["vectors_cleaned"] += cleaned
                    logger.debug(
                        "optimization: FAISS %s incremental — compacted %d vectors",
                        mtype, cleaned,
                    )

            except Exception as e:
                logger.debug("optimization: FAISS %s skipped: %s", mtype, e)

        return result

    # ── SQLite ──

    async def _optimize_sqlite(self) -> dict[str, Any]:
        """
        Standard SQLite maintenance.

        PRAGMA optimize + VACUUM + integrity_check for all databases.
        """
        import aiosqlite

        result = {
            "databases_processed": 0,
            "vacuum_performed": 0,
            "integrity_ok": 0,
            "integrity_failed": 0,
        }

        # Find all SQLite databases
        sqlite_dir = config.paths.sqlite_dir
        if not sqlite_dir.exists():
            return result

        db_files = list(sqlite_dir.glob("*.db"))

        for db_path in db_files:
            try:
                async with aiosqlite.connect(str(db_path)) as db:
                    result["databases_processed"] += 1

                    # Integrity check first
                    cursor = await db.execute("PRAGMA integrity_check")
                    check = await cursor.fetchone()
                    if check and check[0] == "ok":
                        result["integrity_ok"] += 1
                    else:
                        result["integrity_failed"] += 1
                        logger.warning("optimization: SQLite integrity FAILED for %s", db_path.name)
                        continue  # Don't vacuum a corrupt database

                    # Optimize query planner stats
                    await db.execute("PRAGMA optimize")

                    # VACUUM to reclaim space
                    await db.execute("VACUUM")
                    result["vacuum_performed"] += 1

                    logger.debug("optimization: SQLite %s — optimized + vacuumed", db_path.name)

            except Exception as e:
                logger.debug("optimization: SQLite %s skipped: %s", db_path.name, e)

        return result

    # ── Kuzu ──

    async def _optimize_kuzu(self) -> dict[str, Any]:
        """
        Kuzu graph maintenance.

        - Remove orphan nodes (entities with no edges)
        - Remove orphan relationships (edges to non-existent nodes)
        - Log frequently traversed paths for potential index optimization
        """
        result = {
            "orphan_nodes_removed": 0,
            "orphan_edges_removed": 0,
            "hot_paths_identified": 0,
        }

        am = get_associative_memory()

        # Remove orphan nodes (no relationships and no access in 90 days)
        try:
            orphan_nodes = await am.find_orphan_nodes()
            for node_id in orphan_nodes:
                await am.remove_entity(node_id)
                result["orphan_nodes_removed"] += 1
        except Exception as e:
            logger.debug("optimization: Kuzu orphan node cleanup failed: %s", e)

        # Remove orphan edges (pointing to non-existent nodes)
        try:
            orphan_edges = await am.find_orphan_edges()
            for edge_id in orphan_edges:
                await am.remove_relationship(edge_id)
                result["orphan_edges_removed"] += 1
        except Exception as e:
            logger.debug("optimization: Kuzu orphan edge cleanup failed: %s", e)

        # Identify hot paths for future optimization
        try:
            hot_paths = await am.get_frequently_traversed_paths(min_traversals=10, limit=20)
            result["hot_paths_identified"] = len(hot_paths)
        except Exception as e:
            logger.debug("optimization: Kuzu hot path analysis failed: %s", e)

        if result["orphan_nodes_removed"] > 0 or result["orphan_edges_removed"] > 0:
            logger.info(
                "optimization: Kuzu — removed %d orphan nodes, %d orphan edges",
                result["orphan_nodes_removed"],
                result["orphan_edges_removed"],
            )

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_daemon_instance: Optional[OptimizationDaemon] = None


def get_optimization_daemon() -> OptimizationDaemon:
    """Get the singleton OptimizationDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = OptimizationDaemon()
    return _daemon_instance

"""
Forgetting Daemon

Demotes stale SFM facts to LFM (cold storage).
LFM and AM are permanent — only wrong data gets deleted (via Conflict daemon).

Algorithm:
1. Scan SFM for unpinned facts with zero access in last 30 days
2. For each stale fact, create an archived chunk in LFM
3. Remove from SFM (SQLite + FAISS)
4. Update Meta pointers to reflect new location

No AI required — purely algorithmic.
"""

import time
from typing import Any, Optional

from logger import log
from memory.types.lfm import get_long_form_memory
from memory.types.meta import get_meta_memory, MemoryType
from memory.types.sfm import get_short_form_memory
from observability import record_metric

# Staleness thresholds (days without access)
SFM_STALE_DAYS = 30


class ForgettingDaemon:
    """
    Demotes stale memories from hot to cold storage.

    Only affects SFM → LFM demotion. LFM and AM are permanent.
    Wrong data deletion is handled by the Conflict daemon.
    """

    async def run(self, limit: int = 100) -> dict[str, Any]:
        """
        Execute one forgetting pass.

        Args:
            limit: Max items to process per run.

        Returns:
            Stats dict.
        """
        start = time.perf_counter()

        stats = {
            "scanned": 0,
            "demoted": 0,
            "errors": 0,
            "duration_ms": 0.0,
        }

        try:
            candidates = await self._find_stale_sfm(limit)
            stats["scanned"] = len(candidates)

            for fact in candidates:
                try:
                    await self._demote_to_lfm(fact)
                    stats["demoted"] += 1
                except Exception as e:
                    logger.warning("forgetting: failed to demote fact %s: %s", fact.get("id"), e)
                    stats["errors"] += 1

        except Exception as e:
            logger.warning("forgetting: scan failed: %s", e)
            stats["errors"] += 1

        stats["duration_ms"] = (time.perf_counter() - start) * 1000

        await record_metric("mml", "forgetting_duration_ms", stats["duration_ms"])
        await record_metric("mml", "forgetting_demoted", stats["demoted"])

        if stats["demoted"] > 0:
            logger.info("forgetting: demoted %d stale facts to LFM", stats["demoted"])

        return stats

    async def _find_stale_sfm(self, limit: int) -> list[dict[str, Any]]:
        """
        Find SFM facts that are stale and unpinned.

        Stale = not accessed in SFM_STALE_DAYS and not pinned.
        """
        sfm = get_short_form_memory()
        return await sfm.get_stale_facts(days=SFM_STALE_DAYS, limit=limit)

    async def _demote_to_lfm(self, fact: dict[str, Any]) -> None:
        """
        Demote a single SFM fact to LFM.

        1. Create archived chunk in LFM
        2. Remove from SFM
        3. Update Meta pointer
        """
        sfm = get_short_form_memory()
        lfm = get_long_form_memory()
        meta = get_meta_memory()

        fact_id = fact["id"]
        content = fact["content"]
        domain = fact.get("domain")

        # Step 1: Archive to LFM
        lfm_id = await lfm.ingest_text(
            title=f"Archived SFM fact: {content[:50]}",
            content=content,
            source=f"forgetting:sfm:{fact_id}",
            domain=domain,
        )

        # Step 2: Remove from SFM
        await sfm.remove_fact(fact_id)

        # Step 3: Update Meta pointer (point to new LFM location)
        await meta.update_pointer_location(
            old_type=MemoryType.SFM,
            old_ref_id=fact_id,
            new_type=MemoryType.LFM,
            new_ref_id=lfm_id,
        )

        logger.debug("forgetting: demoted fact %s → LFM %s", fact_id, lfm_id)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_daemon_instance: Optional[ForgettingDaemon] = None


def get_forgetting_daemon() -> ForgettingDaemon:
    """Get the singleton ForgettingDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = ForgettingDaemon()
    return _daemon_instance

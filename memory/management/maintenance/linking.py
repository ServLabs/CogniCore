"""
Linking Daemon

Auto-creates cross-memory associations by discovering semantic
relationships between items across all memory types.

Flow:
1. Take recently added/modified items from each memory type
2. FAISS search across OTHER types for semantically related items
3. AI confirms relatedness and determines relationship type
4. Create edges in AM graph linking the items

AI-powered: FAISS finds candidates, LLM confirms and labels relationships.
"""

import json
import time
from typing import Any, Optional

from connectors import genai
from logger import log
from memory.management.faiss_wrapper import get_faiss_store
from memory.management.mml import get_mml
from memory.types.am import get_associative_memory
from memory.types.lfm import get_long_form_memory
from memory.types.mm import get_motor_memory
from memory.types.sfm import get_short_form_memory
from observability import record_metric
from prompts import prompts

# Minimum cosine similarity to consider as link candidate
LINK_SIMILARITY_THRESHOLD = 0.70


class LinkingDaemon:
    """
    Auto-creates cross-memory associations.

    Discovers relationships between:
    - SFM facts ↔ AM entities
    - SFM facts ↔ MM procedures
    - LFM chunks ↔ AM entities
    - MM procedures ↔ AM entities
    - Any memory item ↔ any other type
    """

    async def run(self, limit_per_type: int = 30) -> dict[str, Any]:
        """
        Execute one linking pass.

        Args:
            limit_per_type: Max recent items to process per type.

        Returns:
            Stats dict.
        """
        start = time.perf_counter()

        stats = {
            "items_scanned": 0,
            "candidates_found": 0,
            "links_created": 0,
            "links_existed": 0,
            "errors": 0,
            "duration_ms": 0.0,
        }

        # Process recently modified items from each type
        source_types = ("sfm", "lfm", "mm")
        target_types = ("sfm", "lfm", "am", "mm")

        for source_type in source_types:
            try:
                items = await self._get_recent_items(source_type, limit_per_type)
                stats["items_scanned"] += len(items)

                for item in items:
                    result = await self._find_and_create_links(
                        item, source_type, target_types
                    )
                    stats["candidates_found"] += result["candidates"]
                    stats["links_created"] += result["created"]
                    stats["links_existed"] += result["existed"]

            except Exception as e:
                logger.warning("linking: error processing %s: %s", source_type, e)
                stats["errors"] += 1

        stats["duration_ms"] = (time.perf_counter() - start) * 1000

        await record_metric("mml", "linking_duration_ms", stats["duration_ms"])
        await record_metric("mml", "linking_created", stats["links_created"])

        if stats["links_created"] > 0:
            logger.info("linking: created %d new cross-memory links", stats["links_created"])

        return stats

    async def _get_recent_items(self, memory_type: str, limit: int) -> list[dict[str, Any]]:
        """Get recently added/modified items from a memory type."""
        if memory_type == "sfm":
            return await get_short_form_memory().get_recent_facts(limit=limit)
        elif memory_type == "lfm":
            return await get_long_form_memory().get_recent_chunks(limit=limit)
        elif memory_type == "mm":
            return await get_motor_memory().get_recent_procedures(limit=limit)
        return []

    async def _find_and_create_links(
        self,
        item: dict[str, Any],
        source_type: str,
        target_types: tuple[str, ...],
    ) -> dict[str, int]:
        """Find related items across types and create links."""
        result = {"candidates": 0, "created": 0, "existed": 0}

        mml = get_mml()
        if not mml.embedder:
            return result

        content = self._get_item_content(item)
        if not content:
            return result

        try:
            embedding = await mml.embedder(content)
        except Exception:
            return result

        item_id = item.get("id", "")

        # Search across other types
        for target_type in target_types:
            if target_type == source_type:
                continue  # Skip same-type (handled by Conflict daemon)

            try:
                store = get_faiss_store(target_type)
                candidates = await store.search(embedding, top_k=5)

                for candidate in candidates:
                    if candidate.score < LINK_SIMILARITY_THRESHOLD:
                        continue
                    if candidate.id == item_id:
                        continue

                    result["candidates"] += 1

                    # Check if link already exists
                    am = get_associative_memory()

                    link_exists = await am.relationship_exists(
                        source_id=item_id,
                        target_id=candidate.id,
                    )
                    if link_exists:
                        result["existed"] += 1
                        continue

                    # AI confirms and labels the relationship
                    target_item = await self._fetch_item(target_type, candidate.id)
                    if not target_item:
                        continue

                    relationship = await self._ai_determine_relationship(
                        source_item=item,
                        source_type=source_type,
                        target_item=target_item,
                        target_type=target_type,
                    )

                    if relationship:
                        await am.add_cross_link(
                            source_id=item_id,
                            source_type=source_type,
                            target_id=candidate.id,
                            target_type=target_type,
                            relationship_type=relationship["type"],
                            confidence=relationship.get("confidence", 0.7),
                        )
                        result["created"] += 1

            except Exception as e:
                logger.debug("linking: search in %s failed: %s", target_type, e)

        return result

    async def _ai_determine_relationship(
        self,
        source_item: dict[str, Any],
        source_type: str,
        target_item: dict[str, Any],
        target_type: str,
    ) -> Optional[dict[str, Any]]:
        """
        Ask AI to confirm and label the relationship between two items.

        Returns relationship dict or None if not related.
        """
        source_content = self._get_item_content(source_item)
        target_content = self._get_item_content(target_item)

        messages = prompts.get_messages(
            "memory/linking.md",
            source_type=source_type,
            source_id=source_item.get("id", "?"),
            source_content=source_content,
            target_type=target_type,
            target_id=target_item.get("id", "?"),
            target_content=target_content,
        )

        try:
            raw = await genai.ask({
                "model": "cheap",
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })

            data = json.loads(raw.strip())

            if not data.get("related", False):
                return None

            return {
                "type": data.get("type", "related_to"),
                "confidence": data.get("confidence", 0.7),
                "direction": data.get("direction", "bidirectional"),
            }

        except Exception as e:
            logger.debug("linking: AI relationship check failed: %s", e)
            return None

    async def _fetch_item(self, memory_type: str, item_id: str) -> Optional[dict[str, Any]]:
        """Fetch a specific item by ID."""
        try:
            if memory_type == "sfm":
                return await get_short_form_memory().get_fact(item_id)
            elif memory_type == "lfm":
                return await get_long_form_memory().get_chunk(item_id)
            elif memory_type == "am":
                return await get_associative_memory().get_entity(item_id)
            elif memory_type == "mm":
                return await get_motor_memory().get_procedure(item_id)
        except Exception:
            pass
        return None

    def _get_item_content(self, item: dict[str, Any]) -> str:
        """Extract the main text content from an item dict."""
        return (
            item.get("content", "")
            or item.get("name", "")
            or item.get("title", "")
            or ""
        )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_daemon_instance: Optional[LinkingDaemon] = None


def get_linking_daemon() -> LinkingDaemon:
    """Get the singleton LinkingDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = LinkingDaemon()
    return _daemon_instance

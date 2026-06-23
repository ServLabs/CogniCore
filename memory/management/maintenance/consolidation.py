"""
Consolidation Daemon

Background process that drains staged insights from Redis,
enriches them with conversation context, and routes categorized
knowledge to the appropriate memory types.

Runs on schedule (1 AM daily via PM cron) or on-demand.
"""

import json
import time
from typing import Any, Optional

from connectors import genai
from logger import log
from memory.management.faiss_wrapper import get_faiss_store
from memory.management.learning.meta import get_meta_learner
from memory.management.mml import get_mml
from memory.types.am import get_associative_memory
from memory.types.em import get_emotional_memory
from memory.types.lfm import get_long_form_memory
from memory.types.mm import get_motor_memory
from memory.types.sfm import get_short_form_memory
from memory.types.wm import get_working_memory
from observability import record_metric
from prompts import prompts


class ConsolidationDaemon:
    """
    Processes staged insights into long-term memory.

    Flow:
    1. Drain insights from Redis staging
    2. Group by conversation
    3. Fetch conversation context from WM
    4. Single genai call per conversation → categorized knowledge
    5. FAISS dedup check per item
    6. Route to target memory types (SFM, AM, EM, MM, LFM)
    7. Register pointers in Meta
    """

    async def run(self, batch_size: int = 50) -> dict[str, Any]:
        """
        Execute one consolidation pass.

        Args:
            batch_size: Max insights to process per run.

        Returns:
            Stats dict with counts per category.
        """
        start = time.perf_counter()

        mml = get_mml()

        stats = {
            "conversations_processed": 0,
            "facts_created": 0,
            "entities_created": 0,
            "preferences_created": 0,
            "procedures_created": 0,
            "documents_created": 0,
            "proposals_committed": 0,
            "proposals_skipped": 0,
            "duplicates_skipped": 0,
            "errors": 0,
        }

        # Part 1: Drain and process real-time insights (from thinking prompt)
        insights = await mml.drain_insights(batch_size)

        if insights:
            by_convo: dict[str, list[dict[str, Any]]] = {}
            for ins in insights:
                cid = ins.get("convo_id", "unknown")
                by_convo.setdefault(cid, []).append(ins)

            for convo_id, convo_insights in by_convo.items():
                try:
                    result = await self._process_conversation(convo_id, convo_insights)
                    stats["conversations_processed"] += 1
                    stats["facts_created"] += result.get("facts", 0)
                    stats["entities_created"] += result.get("entities", 0)
                    stats["preferences_created"] += result.get("preferences", 0)
                    stats["procedures_created"] += result.get("procedures", 0)
                    stats["documents_created"] += result.get("documents", 0)
                    stats["duplicates_skipped"] += result.get("duplicates_skipped", 0)
                except Exception as e:
                    logger.warning("consolidation: error processing convo %s: %s", convo_id, e)
                    stats["errors"] += 1

        # Part 2: Drain and commit learning proposals (from sleep-mode learners)
        proposals = await mml.drain_proposals(batch_size)

        if proposals:
            for proposal in proposals:
                try:
                    committed = await self._commit_proposal(proposal)
                    if committed:
                        stats["proposals_committed"] += 1
                    else:
                        stats["proposals_skipped"] += 1
                except Exception as e:
                    logger.warning("consolidation: error committing proposal: %s", e)
                    stats["errors"] += 1

        if not insights and not proposals:
            logger.debug("consolidation: nothing to process")
            return {"status": "empty", "duration_ms": 0}

        stats["duration_ms"] = (time.perf_counter() - start) * 1000

        await record_metric("mml", "consolidation_duration_ms", stats["duration_ms"])
        await record_metric("mml", "consolidation_facts", stats["facts_created"])
        await record_metric("mml", "consolidation_entities", stats["entities_created"])

        logger.info("consolidation: completed — %s", stats)
        return stats

    async def _process_conversation(
        self,
        convo_id: str,
        insights: list[dict[str, Any]],
    ) -> dict[str, int]:
        """
        Process all insights for a single conversation.

        Makes one genai call to categorize all insights together.
        """
        # Build insights block for prompt
        insights_lines = []
        for i, ins in enumerate(insights, 1):
            insights_lines.append(f"{i}. [{ins.get('convo_id', '?')}] {ins['insight']}")
        insights_block = "\n".join(insights_lines)

        # Fetch conversation context from WM
        conversation_block = await self._get_conversation_context(convo_id)

        # Single LLM call for categorization
        messages = prompts.get_messages(
            "memory/consolidation.md",
            insights_block=insights_block,
            conversation_block=conversation_block,
        )

        raw = await genai.ask({
            "model": "cheap",
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        })

        categorized = self._parse_response(raw)
        if not categorized:
            return {"duplicates_skipped": 0}

        # Route each category to the right memory type
        user_id = insights[0].get("user_id", "unknown")
        result = await self._route_to_memory(categorized, user_id, convo_id)
        return result

    async def _get_conversation_context(self, convo_id: str) -> str:
        """Fetch recent messages from WM for context."""
        try:
            wm = get_working_memory()
            # Try to get messages from the conversation
            messages = await wm.get_messages(convo_id=convo_id, limit=20)
            if not messages:
                return "(no conversation context available)"

            lines = []
            for msg in messages:
                role = msg.role.value if hasattr(msg.role, "value") else msg.role
                lines.append(f"[{role}] {msg.content[:200]}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("consolidation: could not fetch convo context: %s", e)
            return "(conversation context unavailable)"

    def _parse_response(self, raw: str) -> Optional[dict[str, list]]:
        """Parse LLM JSON response into categorized knowledge."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0].strip()

            data = json.loads(text)

            # Validate structure
            expected_keys = {"facts", "entities", "preferences", "procedures", "documents"}
            if not any(k in data for k in expected_keys):
                logger.warning("consolidation: LLM response missing expected keys")
                return None

            return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("consolidation: failed to parse LLM response: %s", e)
            return None

    async def _route_to_memory(
        self,
        categorized: dict[str, list],
        user_id: str,
        convo_id: str,
    ) -> dict[str, int]:
        """Route categorized knowledge to target memory types."""
        mml = get_mml()
        counts = {
            "facts": 0,
            "entities": 0,
            "preferences": 0,
            "procedures": 0,
            "documents": 0,
            "duplicates_skipped": 0,
        }

        # Route facts → SFM
        for fact in categorized.get("facts", []):
            if await self._is_duplicate(fact.get("content", ""), "sfm"):
                counts["duplicates_skipped"] += 1
                continue
            try:
                fact_id = await get_short_form_memory().add_fact(
                    content=fact["content"],
                    source=f"consolidation:{convo_id}",
                    confidence=fact.get("confidence", 0.8),
                )
                await mml.auto_register(fact["content"], "sfm", fact_id)
                counts["facts"] += 1
            except Exception as e:
                logger.warning("consolidation: failed to store fact: %s", e)

        # Route entities → AM
        for entity in categorized.get("entities", []):
            try:
                am = get_associative_memory()
                entity_id = await am.add_entity(
                    name=entity["name"],
                    entity_type=entity.get("type", "concept"),
                    attributes=entity.get("attributes", {}),
                )
                # Add relationships
                for rel in entity.get("relationships", []):
                    await am.add_relationship(
                        source=entity["name"],
                        target=rel["target"],
                        relationship_type=rel["type"],
                    )
                await mml.auto_register(entity["name"], "am", entity_id)
                counts["entities"] += 1
            except Exception as e:
                logger.warning("consolidation: failed to store entity: %s", e)

        # Route preferences → EM
        for pref in categorized.get("preferences", []):
            if await self._is_duplicate(pref.get("content", ""), "em"):
                counts["duplicates_skipped"] += 1
                continue
            try:
                await get_emotional_memory().update_preference(
                    user_id=user_id,
                    preference=pref["content"],
                    strength=pref.get("strength", 0.7),
                )
                counts["preferences"] += 1
            except Exception as e:
                logger.warning("consolidation: failed to store preference: %s", e)

        # Route procedures → MM
        for proc in categorized.get("procedures", []):
            if await self._is_duplicate(proc.get("title", ""), "mm"):
                counts["duplicates_skipped"] += 1
                continue
            try:
                proc_id = await get_motor_memory().add_procedure(
                    title=proc["title"],
                    steps=proc.get("steps", []),
                    source=f"consolidation:{convo_id}",
                )
                await mml.auto_register(proc["title"], "mm", proc_id)
                counts["procedures"] += 1
            except Exception as e:
                logger.warning("consolidation: failed to store procedure: %s", e)

        # Route documents → LFM
        for doc in categorized.get("documents", []):
            if await self._is_duplicate(doc.get("content", "")[:100], "lfm"):
                counts["duplicates_skipped"] += 1
                continue
            try:
                doc_id = await get_long_form_memory().ingest_text(
                    title=doc["title"],
                    content=doc["content"],
                    source=f"consolidation:{convo_id}",
                )
                await mml.auto_register(doc["title"], "lfm", doc_id)
                counts["documents"] += 1
            except Exception as e:
                logger.warning("consolidation: failed to store document: %s", e)

        return counts

    async def _commit_proposal(self, proposal_envelope: dict[str, Any]) -> bool:
        """
        Commit a learning proposal to permanent memory.

        Proposals are pre-categorized by learning type.
        Dedup check before each write. No LLM call needed.

        Args:
            proposal_envelope: {learning_type, proposal, timestamp}

        Returns:
            True if committed, False if skipped (duplicate or unrecognized).
        """
        mml = get_mml()
        learning_type = proposal_envelope.get("learning_type", "")
        proposal = proposal_envelope.get("proposal", {})
        ptype = proposal.get("type", "")

        # Route based on learning type + proposal type
        if ptype == "procedure" or ptype == "transferred_procedure":
            title = proposal.get("title", "")
            if await self._is_duplicate(title, "mm"):
                return False
            proc_id = await get_motor_memory().add_procedure(
                title=title,
                steps=proposal.get("steps", []),
                source=f"learning:{learning_type}",
            )
            await mml.auto_register(title, "mm", proc_id)
            return True

        elif ptype == "hierarchy":
            summary = proposal.get("summary", "")
            if await self._is_duplicate(summary, "sfm"):
                return False
            fact_id = await get_short_form_memory().add_fact(
                content=f"[L{proposal.get('level', 0)} abstraction] {summary}",
                source=f"learning:abstraction",
                confidence=0.7,
            )
            await mml.auto_register(summary, "sfm", fact_id)
            return True

        elif ptype == "analogy":
            source_domain = proposal.get("source_domain", "")
            target_domain = proposal.get("target_domain", "")
            content = f"Analogy: {source_domain} → {target_domain}"
            if await self._is_duplicate(content, "am"):
                return False
            am = get_associative_memory()
            for mapping in proposal.get("mappings", []):
                await am.add_relationship(
                    source=mapping.get("source", ""),
                    target=mapping.get("target", ""),
                    relationship_type="analogous_to",
                )
            return True

        elif ptype == "contrastive_pair":
            positive = proposal.get("positive", "")
            negative = proposal.get("negative", "")
            content = f"{positive} ≠ {negative}"
            if await self._is_duplicate(content, "am"):
                return False
            await get_associative_memory().add_relationship(
                source=positive,
                target=negative,
                relationship_type="NOT_SAME_AS",
            )
            return True

        elif ptype == "budget_adjustment":
            # Meta-learning budget adjustments are applied directly to meta-learner
            meta = get_meta_learner()
            meta.apply_adjustments(proposal.get("adjustments", {}))
            return True

        else:
            logger.debug("consolidation: unrecognized proposal type '%s' from %s", ptype, learning_type)
            return False

    async def _is_duplicate(self, content: str, memory_type: str) -> bool:
        """
        Check if content already exists in memory via FAISS similarity.

        Returns True if a near-duplicate (>0.92 cosine similarity) exists.
        """
        if not content:
            return False

        mml = get_mml()
        if not mml.embedder:
            return False  # No embedder available, skip dedup

        try:
            embedding = await mml.embedder(content)

            # Check appropriate FAISS index based on memory type
            store = get_faiss_store(memory_type)
            results = await store.search(embedding, top_k=1)

            if results and results[0].score > 0.92:
                logger.debug("consolidation: duplicate detected (%.3f): %s...", results[0].score, content[:50])
                return True
        except Exception as e:
            logger.debug("consolidation: dedup check failed: %s", e)

        return False


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_daemon_instance: Optional[ConsolidationDaemon] = None


def get_consolidation_daemon() -> ConsolidationDaemon:
    """Get the singleton ConsolidationDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = ConsolidationDaemon()
    return _daemon_instance

"""
Memory Management Layer (MML)

The agent's subconscious — autonomous processes that maintain, optimize,
and evolve the memory system.

This module provides:
- Real-time operations (recall, auto-registration, light learning)
- Background operations (consolidation, forgetting, optimization)
- Self-scheduling via Prospective Memory
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from connectors import genai
from config import config
from logger import log
from memory.management.recall import RecallEngine, RecallResult, get_recall_engine
from memory.management.budget import BudgetManager, ModelTier, TaskPriority, get_budget_manager
from prompts import prompts


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsolidationResult:
    """Result of memory consolidation."""
    conversations_processed: int = 0
    entities_extracted: int = 0
    facts_created: int = 0
    pointers_registered: int = 0
    duration_ms: float = 0.0
    llm_calls: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "conversations_processed": self.conversations_processed,
            "entities_extracted": self.entities_extracted,
            "facts_created": self.facts_created,
            "pointers_registered": self.pointers_registered,
            "duration_ms": self.duration_ms,
            "llm_calls": self.llm_calls,
        }


@dataclass
class MaintenanceResult:
    """Result of maintenance operation."""
    task_type: str
    items_processed: int = 0
    items_affected: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "items_processed": self.items_processed,
            "items_affected": self.items_affected,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Memory Management Layer
# ══════════════════════════════════════════════════════════════════════════════

class MemoryManagementLayer:
    """
    The agent's subconscious — autonomous memory management.
    
    Provides:
    - Real-time: recall, auto-registration, light learning
    - Background: consolidation, forgetting, optimization
    - Self-scheduling via PM cron jobs
    
    Attributes:
        recall_engine: Engine for memory recall.
        budget_manager: LLM budget manager.
        llm: LLM callable for background tasks.
    """
    
    def __init__(
        self,
        recall_engine: Optional[RecallEngine] = None,
        budget_manager: Optional[BudgetManager] = None,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
        redis_client: Optional[Any] = None,
    ):
        """
        Initialize MML.
        
        Args:
            recall_engine: Engine for memory recall.
            budget_manager: LLM budget manager.
            llm: LLM callable for background tasks.
            embedder: Embedding function.
            redis_client: Redis async client for insight staging.
        """
        self.recall_engine = recall_engine or get_recall_engine()
        self.budget_manager = budget_manager or get_budget_manager()
        self.llm = llm
        self.embedder = embedder
        self.redis = redis_client
    
    # ══════════════════════════════════════════════════════════════════════════
    # Real-Time Operations
    # ══════════════════════════════════════════════════════════════════════════
    
    async def recall(
        self,
        query: str,
        user_id: Optional[str] = None,
        convo_id: Optional[str] = None,
        sources: Optional[list[str]] = None,
        top_k: int = 20,
    ) -> RecallResult:
        """
        Recall relevant memories for a query.
        
        Real-time operation - must be fast.
        
        Args:
            query: User query.
            user_id: User ID for context.
            convo_id: Conversation ID for context.
            sources: Specific sources to search.
            top_k: Number of results.
            
        Returns:
            RecallResult with ranked items.
        """
        result = await self.recall_engine.recall(
            query=query,
            sources=sources,
            top_k=top_k,
            include_wm=True,
            user_id=user_id,
            convo_id=convo_id,
        )
        
        # Record metrics
        from observability import record_latency, record_count
        await record_latency("mml", "recall", result.search_time_ms)
        await record_count("mml", "recall_results", len(result.items))
        
        return result
    
    async def auto_register(
        self,
        content: str,
        memory_type: str,
        ref_id: str,
        domain: Optional[str] = None,
    ) -> Optional[str]:
        """
        Auto-register a memory write in Meta Memory.
        
        Called after any memory write (SFM, LFM, AM, MM).
        
        Args:
            content: Content that was written.
            memory_type: Type of memory (sfm, lfm, am, mm).
            ref_id: Reference ID in target memory.
            domain: Domain from config.
            
        Returns:
            Pointer ID if registered, None if skipped.
        """
        # Extract topic label (first 50 chars or first sentence)
        topic = content[:50].split(".")[0].strip()
        if len(topic) < 5:
            return None  # Too short to be useful
        
        from memory.types.meta import get_meta_memory, MemoryType
        
        meta = get_meta_memory()
        
        # Map string to MemoryType enum
        type_map = {
            "sfm": MemoryType.SFM,
            "lfm": MemoryType.LFM,
            "am": MemoryType.AM,
            "mm": MemoryType.MM,
        }
        
        mem_type = type_map.get(memory_type)
        if not mem_type:
            return None
        
        pointer = await meta.register_pointer(
            topic=topic,
            memory_type=mem_type,
            ref_id=ref_id,
            domain=domain,
        )
        
        return pointer.id
    
    async def light_learn(
        self,
        user_message: str,
        assistant_response: str,
        user_id: str,
    ) -> bool:
        """
        Extract obvious facts from conversation inline.
        
        Called during conversation flow for quick learning.
        
        Args:
            user_message: User's message.
            assistant_response: Assistant's response.
            user_id: User ID.
            
        Returns:
            True if something was learned.
        """
        # Simple pattern matching for corrections
        correction_patterns = [
            "actually it's",
            "no, it's",
            "that's wrong",
            "the correct",
            "i meant",
        ]
        
        msg_lower = user_message.lower()
        if not any(p in msg_lower for p in correction_patterns):
            return False
        
        try:
            messages = prompts.get_messages(
                "memory/correction.md",
                assistant_response=assistant_response[:500],
                user_message=user_message[:500],
            )

            raw = await genai.ask({
                "model": "cheap",
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })

            data = json.loads(raw.strip())

            if not data.get("is_correction", False):
                return False

            fact_summary = data.get("fact_summary", "")
            confidence = data.get("confidence", 0.8)

            if not fact_summary:
                return False

            from memory.types.sfm import get_short_form_memory

            sfm = get_short_form_memory()
            fact_id = await sfm.add_fact(
                content=fact_summary,
                source=f"correction:{user_id}",
                confidence=confidence,
            )

            await self.auto_register(fact_summary, "sfm", fact_id)
            logger.info("light_learn: correction applied — %s", fact_summary[:80])
            return True

        except Exception as e:
            logger.debug("light_learn: correction extraction failed: %s", e)
            return False
    
    # ── Insight Staging ──
    
    INSIGHT_KEY_PREFIX = "cognicore:insights:"
    INSIGHT_TTL_SECONDS = 36 * 60 * 60  # 36 hours
    
    async def stage_insight(
        self,
        insight: str,
        user_id: str,
        convo_id: str,
    ) -> Optional[str]:
        """
        Stage a real-time insight for background consolidation.
        
        Called from pipeline after Thinker extracts a non-null new_insight.
        Stores as individual Redis key with 36hr TTL.
        
        Args:
            insight: The extracted insight text.
            user_id: User ID.
            convo_id: Conversation ID.
            
        Returns:
            Insight ID if staged, None if Redis unavailable.
        """
        if not self.redis:
            logger.debug("mml.stage_insight: no redis client, skipping")
            return None
        
        insight_id = str(uuid.uuid4())
        key = f"{self.INSIGHT_KEY_PREFIX}{insight_id}"
        
        payload = json.dumps({
            "user_id": user_id,
            "convo_id": convo_id,
            "insight": insight,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            await self.redis.set(key, payload, ex=self.INSIGHT_TTL_SECONDS)
            logger.debug("mml.stage_insight: staged %s", insight_id)
            return insight_id
        except Exception as e:
            logger.warning("mml.stage_insight: redis error: %s", e)
            return None
    
    async def drain_insights(self, batch_size: int = 50) -> list[dict[str, Any]]:
        """
        Drain staged insights from Redis for consolidation.
        
        Reads and deletes insight keys atomically.
        
        Args:
            batch_size: Maximum insights to drain per call.
            
        Returns:
            List of insight payloads.
        """
        if not self.redis:
            return []
        
        insights: list[dict[str, Any]] = []
        
        try:
            # SCAN for insight keys
            cursor = 0
            keys_to_drain: list[str] = []
            
            while len(keys_to_drain) < batch_size:
                cursor, found = await self.redis.scan(
                    cursor=cursor,
                    match=f"{self.INSIGHT_KEY_PREFIX}*",
                    count=batch_size,
                )
                keys_to_drain.extend(found)
                if cursor == 0:
                    break
            
            keys_to_drain = keys_to_drain[:batch_size]
            
            if not keys_to_drain:
                return []
            
            # GET all values then DELETE
            pipe = self.redis.pipeline()
            for key in keys_to_drain:
                pipe.get(key)
            values = await pipe.execute()
            
            # Delete drained keys
            await self.redis.delete(*keys_to_drain)
            
            for val in values:
                if val:
                    insights.append(json.loads(val))
            
            logger.debug("mml.drain_insights: drained %d insights", len(insights))
        except Exception as e:
            logger.warning("mml.drain_insights: redis error: %s", e)
        
        return insights
    
    # ── Learning Proposal Staging ──
    
    PROPOSAL_KEY_PREFIX = "cognicore:proposals:"
    PROPOSAL_TTL_SECONDS = 36 * 60 * 60  # 36 hours
    
    async def stage_proposal(
        self,
        learning_type: str,
        proposal: dict[str, Any],
    ) -> Optional[str]:
        """
        Stage a learning proposal for maintenance to commit.
        
        Sleep-mode learners produce proposals; maintenance daemons
        validate and commit them to permanent memory.
        
        Args:
            learning_type: One of the 10 learning types (e.g., "generalization").
            proposal: Structured proposal data (type-specific).
            
        Returns:
            Proposal ID if staged, None if Redis unavailable.
        """
        if not self.redis:
            logger.debug("mml.stage_proposal: no redis client, skipping")
            return None
        
        proposal_id = str(uuid.uuid4())
        key = f"{self.PROPOSAL_KEY_PREFIX}{proposal_id}"
        
        payload = json.dumps({
            "learning_type": learning_type,
            "proposal": proposal,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        try:
            await self.redis.set(key, payload, ex=self.PROPOSAL_TTL_SECONDS)
            logger.debug("mml.stage_proposal: staged %s (%s)", proposal_id, learning_type)
            return proposal_id
        except Exception as e:
            logger.warning("mml.stage_proposal: redis error: %s", e)
            return None
    
    async def drain_proposals(self, batch_size: int = 100) -> list[dict[str, Any]]:
        """
        Drain staged learning proposals for maintenance to process.
        
        Args:
            batch_size: Maximum proposals to drain per call.
            
        Returns:
            List of proposal payloads.
        """
        if not self.redis:
            return []
        
        proposals: list[dict[str, Any]] = []
        
        try:
            cursor = 0
            keys_to_drain: list[str] = []
            
            while len(keys_to_drain) < batch_size:
                cursor, found = await self.redis.scan(
                    cursor=cursor,
                    match=f"{self.PROPOSAL_KEY_PREFIX}*",
                    count=batch_size,
                )
                keys_to_drain.extend(found)
                if cursor == 0:
                    break
            
            keys_to_drain = keys_to_drain[:batch_size]
            
            if not keys_to_drain:
                return []
            
            pipe = self.redis.pipeline()
            for key in keys_to_drain:
                pipe.get(key)
            values = await pipe.execute()
            
            await self.redis.delete(*keys_to_drain)
            
            for val in values:
                if val:
                    proposals.append(json.loads(val))
            
            logger.debug("mml.drain_proposals: drained %d proposals", len(proposals))
        except Exception as e:
            logger.warning("mml.drain_proposals: redis error: %s", e)
        
        return proposals
    
    async def detect_conflict(
        self,
        items: list[Any],
    ) -> list[tuple[Any, Any]]:
        """
        Detect conflicts in retrieved items (inline, real-time check).
        
        Args:
            items: Retrieved memory items.
            
        Returns:
            List of conflicting pairs.
        """
        # Real-time conflict detection for recall results
        # Full background scan handled by ConflictDaemon
        return []
    
    async def run_conflict_scan(self) -> MaintenanceResult:
        """
        Run background conflict detection across all memory types.
        
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="conflict")
        
        from memory.management.maintenance.conflict import get_conflict_daemon
        
        daemon = get_conflict_daemon()
        stats = await daemon.run()
        
        result.items_processed = stats.get("scanned", 0)
        result.items_affected = stats.get("conflicts_confirmed", 0)
        result.errors = [f"errors: {stats['errors']}"] if stats.get("errors") else []
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def check_coherence(self) -> MaintenanceResult:
        """
        Check cross-type consistency and referential integrity.
        
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="coherence")
        
        from memory.management.maintenance.coherence import get_coherence_daemon
        
        daemon = get_coherence_daemon()
        stats = await daemon.run()
        
        result.items_processed = stats.get("cross_type_checked", 0) + stats.get("referential_checked", 0)
        result.items_affected = stats.get("cross_type_issues", 0) + stats.get("referential_issues", 0)
        result.errors = [f"errors: {stats['errors']}"] if stats.get("errors") else []
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def run_linking(self) -> MaintenanceResult:
        """
        Auto-create cross-memory associations.
        
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="linking")
        
        from memory.management.maintenance.linking import get_linking_daemon
        
        daemon = get_linking_daemon()
        stats = await daemon.run()
        
        result.items_processed = stats.get("items_scanned", 0)
        result.items_affected = stats.get("links_created", 0)
        result.errors = [f"errors: {stats['errors']}"] if stats.get("errors") else []
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    # ══════════════════════════════════════════════════════════════════════════
    # Background Operations
    # ══════════════════════════════════════════════════════════════════════════
    
    async def consolidate(
        self,
        days_back: int = 1,
    ) -> ConsolidationResult:
        """
        Consolidate recent conversations into long-term memory.
        
        Background operation - runs during off-peak hours.
        Drains staged insights, categorizes via LLM, routes to memory types.
        
        Args:
            days_back: Number of days to process (used for batch sizing).
            
        Returns:
            ConsolidationResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = ConsolidationResult()
        
        # Check budget
        cost = self.budget_manager.estimate_task_cost("consolidation", 1)
        if not self.budget_manager.can_afford(cost.estimated_calls):
            logger.info("mml.consolidate: budget exceeded, skipping")
            return result
        
        # Delegate to consolidation daemon
        from memory.management.maintenance.consolidation import get_consolidation_daemon
        
        daemon = get_consolidation_daemon()
        stats = await daemon.run(batch_size=50)
        
        # Map daemon stats to ConsolidationResult
        result.conversations_processed = stats.get("conversations_processed", 0)
        result.entities_extracted = stats.get("entities_created", 0)
        result.facts_created = stats.get("facts_created", 0)
        result.llm_calls = result.conversations_processed  # 1 call per conversation
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Register pointer count = all items stored
        result.pointers_registered = (
            result.facts_created
            + result.entities_extracted
            + stats.get("procedures_created", 0)
            + stats.get("documents_created", 0)
        )
        
        # Log to analytics
        from observability import record_metric
        await record_metric("mml", "consolidation_duration_ms", result.duration_ms)
        await record_metric("mml", "consolidation_entities", result.entities_extracted)
        
        return result
    
    async def promote_to_sfm(
        self,
        threshold: int = 10,
        limit: int = 50,
    ) -> MaintenanceResult:
        """
        Promote frequently accessed LFM chunks to SFM.
        
        Args:
            threshold: Minimum access count.
            limit: Maximum promotions.
            
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="sfm_promotion")
        
        from memory.types.lfm import get_long_form_memory
        from memory.types.sfm import get_short_form_memory
        
        lfm = get_long_form_memory()
        sfm = get_short_form_memory()
        
        # Get promotion candidates
        candidates = await lfm.get_promotion_candidates(threshold, limit)
        result.items_processed = len(candidates)
        
        for chunk in candidates:
            # Check budget
            if not self.budget_manager.can_afford(1, model_tier=ModelTier.CHEAP):
                break
            
            # Summarize chunk (would use LLM)
            if self.llm:
                # compressed = await self.llm(f"Summarize in one line: {chunk.content}")
                compressed = chunk.content[:100]  # Placeholder
            else:
                compressed = chunk.content[:100]
            
            # Get document for domain
            doc = await lfm.get_document(chunk.document_id)
            domain = doc.domain if doc else config.domain_config.default_domain
            
            # Promote to SFM
            await sfm.promote_from_lfm(
                lfm_chunk_id=chunk.id,
                compressed_fact=compressed,
                domain=domain,
            )
            
            # Mark as promoted
            await lfm.mark_promoted(chunk.id)
            result.items_affected += 1
        
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def forget_stale(
        self,
        sfm_days: int = 30,
        lfm_days: int = 90,
    ) -> MaintenanceResult:
        """
        Demote stale SFM facts to LFM.
        
        LFM and AM are permanent — only wrong data gets deleted (Conflict daemon).
        
        Args:
            sfm_days: Days without access before SFM demotion.
            lfm_days: Unused (LFM never forgets).
            
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="forgetting")
        
        from memory.management.maintenance.forgetting import get_forgetting_daemon
        
        daemon = get_forgetting_daemon()
        stats = await daemon.run(limit=100)
        
        result.items_processed = stats.get("scanned", 0)
        result.items_affected = stats.get("demoted", 0)
        result.errors = [f"errors: {stats['errors']}"] if stats.get("errors") else []
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def optimize_indexes(self) -> MaintenanceResult:
        """
        Optimize FAISS indexes, SQLite databases, and Kuzu graph.
        
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="optimization")
        
        from memory.management.maintenance.optimization import get_optimization_daemon
        
        daemon = get_optimization_daemon()
        stats = await daemon.run()
        
        faiss_stats = stats.get("faiss", {})
        sqlite_stats = stats.get("sqlite", {})
        kuzu_stats = stats.get("kuzu", {})
        
        result.items_processed = (
            faiss_stats.get("indexes_processed", 0)
            + sqlite_stats.get("databases_processed", 0)
        )
        result.items_affected = (
            faiss_stats.get("vectors_cleaned", 0)
            + sqlite_stats.get("vacuum_performed", 0)
            + kuzu_stats.get("orphan_nodes_removed", 0)
        )
        result.errors = [f"errors: {stats['errors']}"] if stats.get("errors") else []
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def check_integrity(self) -> MaintenanceResult:
        """
        Check and auto-repair memory integrity.
        
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="integrity")
        
        from memory.management.maintenance.integrity import get_integrity_daemon
        
        daemon = get_integrity_daemon()
        stats = await daemon.run()
        
        result.items_processed = stats.get("total_checked", 0)
        result.items_affected = stats.get("total_repaired", 0)
        result.errors = [f"errors: {stats['errors']}"] if stats.get("errors") else []
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    # ══════════════════════════════════════════════════════════════════════════
    # Task Execution (called by DMN)
    # ══════════════════════════════════════════════════════════════════════════
    
    async def execute_task(self, task_name: str) -> Optional[MaintenanceResult]:
        """
        Execute a maintenance task by name.
        
        Called by DMN during sleep cycle. No self-scheduling —
        DMN owns the pipeline order and parallelism.
        
        Args:
            task_name: One of: consolidation, forgetting, conflict,
                       coherence, optimization, integrity, linking.
                       
        Returns:
            MaintenanceResult or None if unknown task.
        """
        task_map: dict[str, Callable[..., Any]] = {
            "consolidation": self.consolidate,
            "forgetting": self.forget_stale,
            "conflict": self.run_conflict_scan,
            "coherence": self.check_coherence,
            "optimization": self.optimize_indexes,
            "integrity": self.check_integrity,
            "linking": self.run_linking,
        }
        
        handler = task_map.get(task_name)
        if handler:
            return await handler()
        
        logger.warning("mml.execute_task: unknown task '%s'", task_name)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_mml_instance: Optional[MemoryManagementLayer] = None


def get_mml() -> MemoryManagementLayer:
    """Get the singleton MML instance with Redis on DB for MML staging."""
    global _mml_instance
    if _mml_instance is None:
        redis_client = _connect_redis_mml()
        _mml_instance = MemoryManagementLayer(redis_client=redis_client)
    return _mml_instance


def _connect_redis_mml():
    """Connect to Redis DB dedicated to MML staging. Returns None on failure."""
    try:
        import redis.asyncio as aioredis
        return aioredis.from_url(config.redis.url(config.redis.db_mml_staging))
    except Exception as e:
        log.warning("MML: Redis unavailable (%s) — staging disabled", e)
        return None


# ── Convenience Functions ──

async def recall(
    query: str,
    user_id: Optional[str] = None,
    convo_id: Optional[str] = None,
    top_k: int = 20,
) -> RecallResult:
    """Recall relevant memories for a query."""
    return await get_mml().recall(query, user_id, convo_id, top_k=top_k)


async def consolidate(days_back: int = 1) -> ConsolidationResult:
    """Consolidate recent conversations."""
    return await get_mml().consolidate(days_back)

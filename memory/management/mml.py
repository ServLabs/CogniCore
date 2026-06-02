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
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from core import config
from memory.management.recall import RecallEngine, RecallResult, get_recall_engine
from memory.management.budget import BudgetManager, ModelTier, TaskPriority, get_budget_manager


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
    
    # Background task schedule (cron expressions)
    BACKGROUND_SCHEDULE = {
        "consolidation": "0 1 * * *",      # 1 AM daily
        "generalization": "0 2 * * *",     # 2 AM daily
        "forgetting": "0 3 * * *",         # 3 AM daily
        "coherence": "0 4 * * *",          # 4 AM daily
        "optimization": "0 5 * * *",       # 5 AM daily
        "auto_registration": "*/30 * * * *",  # Every 30 min
    }
    
    def __init__(
        self,
        recall_engine: Optional[RecallEngine] = None,
        budget_manager: Optional[BudgetManager] = None,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
    ):
        """
        Initialize MML.
        
        Args:
            recall_engine: Engine for memory recall.
            budget_manager: LLM budget manager.
            llm: LLM callable for background tasks.
            embedder: Embedding function.
        """
        self.recall_engine = recall_engine or get_recall_engine()
        self.budget_manager = budget_manager or get_budget_manager()
        self.llm = llm
        self.embedder = embedder
        
        self._background_running = False
        self._background_tasks: list[asyncio.Task] = []
    
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
        
        # TODO: Use LLM to extract the correction and update SFM
        # For now, just flag for background processing
        return False
    
    async def detect_conflict(
        self,
        items: list[Any],
    ) -> list[tuple[Any, Any]]:
        """
        Detect conflicts in retrieved items.
        
        Args:
            items: Retrieved memory items.
            
        Returns:
            List of conflicting pairs.
        """
        # TODO: Implement conflict detection
        # Compare facts from different sources
        return []
    
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
        
        Args:
            days_back: Number of days to process.
            
        Returns:
            ConsolidationResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = ConsolidationResult()
        
        # Check budget
        cost = self.budget_manager.estimate_task_cost("consolidation", 1)
        if not self.budget_manager.can_afford(cost.estimated_calls):
            return result
        
        # Get recent conversations from WM/JSONL
        from memory.types.wm import get_working_memory
        
        wm = get_working_memory()
        # TODO: Implement conversation retrieval for date range
        
        # For each conversation:
        # 1. Extract entities and relationships (LLM)
        # 2. Insert into AM graph
        # 3. Compress key facts into SFM
        # 4. Register in Meta
        
        # Placeholder - actual implementation needs LLM
        if self.llm:
            # Would call LLM here for extraction
            pass
        
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        
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
        Prune stale, never-accessed memories.
        
        Args:
            sfm_days: Days without access before SFM eviction.
            lfm_days: Days without access before LFM flagging.
            
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="forgetting")
        
        # TODO: Implement forgetting logic
        # - SFM: Evict unpinned facts with zero access in last N days
        # - LFM: Flag chunks for review
        # - AM: Remove edges with zero traversals
        # - Meta: Remove pointers to deleted memories
        
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def optimize_indexes(self) -> MaintenanceResult:
        """
        Optimize FAISS indexes and databases.
        
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="optimization")
        
        # TODO: Implement optimization
        # - FAISS: Remove deleted vectors, re-cluster
        # - SQLite: PRAGMA optimize, VACUUM
        # - Kuzu: Remove orphan nodes
        
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def check_integrity(self) -> MaintenanceResult:
        """
        Check and repair memory integrity.
        
        Returns:
            MaintenanceResult with stats.
        """
        import time
        start_time = time.perf_counter()
        
        result = MaintenanceResult(task_type="integrity")
        
        # TODO: Implement integrity checks
        # - SFM fact → deleted LFM chunk
        # - Meta pointer → deleted memory
        # - FAISS ID → no SQLite record
        # - SQLite record → no FAISS vector
        
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    # ══════════════════════════════════════════════════════════════════════════
    # Self-Scheduling
    # ══════════════════════════════════════════════════════════════════════════
    
    async def schedule_background_tasks(self) -> None:
        """
        Schedule background tasks via Prospective Memory.
        
        Uses PM cron jobs for self-scheduling.
        """
        from memory.types.pm import get_prospective_memory
        
        pm = get_prospective_memory()
        
        for task_name, cron_expr in self.BACKGROUND_SCHEDULE.items():
            await pm.schedule_task(
                task_type="mml_background",
                payload={"task": task_name},
                cron_expression=cron_expr,
            )
    
    async def start_background_loop(self) -> None:
        """Start background task processing loop."""
        if self._background_running:
            return
        
        self._background_running = True
        
        # Schedule tasks in PM
        await self.schedule_background_tasks()
        
        # Start processing loop
        task = asyncio.create_task(self._background_loop())
        self._background_tasks.append(task)
    
    async def stop_background_loop(self) -> None:
        """Stop background task processing."""
        self._background_running = False
        
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._background_tasks.clear()
    
    async def _background_loop(self) -> None:
        """Background task processing loop."""
        while self._background_running:
            # Check for due tasks from PM
            from memory.types.pm import get_prospective_memory
            
            pm = get_prospective_memory()
            due_tasks = await pm.get_due_tasks()
            
            for task in due_tasks:
                if task.payload.get("task") in self.BACKGROUND_SCHEDULE:
                    await self._execute_background_task(task.payload["task"])
                    await pm.mark_executed(task.id, success=True)
            
            # Sleep before next check
            await asyncio.sleep(60)  # Check every minute
    
    async def _execute_background_task(self, task_name: str) -> None:
        """Execute a background task by name."""
        task_map = {
            "consolidation": self.consolidate,
            "generalization": lambda: None,  # TODO
            "forgetting": self.forget_stale,
            "coherence": lambda: None,  # TODO
            "optimization": self.optimize_indexes,
            "auto_registration": lambda: None,  # TODO
        }
        
        handler = task_map.get(task_name)
        if handler:
            await handler()


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_mml_instance: Optional[MemoryManagementLayer] = None


def get_mml() -> MemoryManagementLayer:
    """Get the singleton MML instance."""
    global _mml_instance
    if _mml_instance is None:
        _mml_instance = MemoryManagementLayer()
    return _mml_instance


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


async def start_background_tasks() -> None:
    """Start background task processing."""
    await get_mml().start_background_loop()


async def stop_background_tasks() -> None:
    """Stop background task processing."""
    await get_mml().stop_background_loop()

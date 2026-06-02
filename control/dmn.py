"""
Default Mode Network (DMN)

What the agent does when no one is asking — self-reflection,
spontaneous planning, and proactive maintenance.

Activated by CEN when Salience Network detects idle mode.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from core import config


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReflectionResult:
    """Result of self-reflection."""
    timestamp: datetime
    performance_summary: dict[str, Any] = field(default_factory=dict)
    weak_areas: list[str] = field(default_factory=list)
    total_interactions: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "performance_summary": self.performance_summary,
            "weak_areas": self.weak_areas,
            "total_interactions": self.total_interactions,
        }


@dataclass
class PlanningResult:
    """Result of spontaneous planning."""
    timestamp: datetime
    patterns_found: int = 0
    tasks_scheduled: int = 0
    stale_docs_found: int = 0
    slow_domains: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "patterns_found": self.patterns_found,
            "tasks_scheduled": self.tasks_scheduled,
            "stale_docs_found": self.stale_docs_found,
            "slow_domains": self.slow_domains,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Default Mode Network
# ══════════════════════════════════════════════════════════════════════════════

class DefaultModeNetwork:
    """
    Idle-time processing network.
    
    Provides:
    - Self-reflection: Review recent performance
    - Spontaneous planning: Anticipate user needs
    - Maintenance: Delegate to MML sleep tasks
    
    Three phases:
    1. Self-Reflection: "How did I perform recently?"
    2. Spontaneous Planning: "What should I prepare for?"
    3. Maintenance: MML sleep tasks (consolidation, etc.)
    """
    
    def __init__(self):
        """Initialize DMN."""
        self.active = False
        self._cancel_event = asyncio.Event()
        self._last_reflection: Optional[ReflectionResult] = None
        self._last_planning: Optional[PlanningResult] = None
    
    async def activate(self) -> None:
        """
        Start DMN processing.
        
        Runs until deactivated by CEN.
        """
        self.active = True
        self._cancel_event.clear()
        
        try:
            # Phase 1: Self-reflection
            await self._reflect()
            if not self.active:
                return
            
            # Phase 2: Spontaneous planning
            await self._plan()
            if not self.active:
                return
            
            # Phase 3: Maintenance (delegate to MML)
            await self._run_maintenance()
            
        except asyncio.CancelledError:
            pass
        finally:
            self.active = False
    
    def deactivate(self) -> None:
        """
        Deactivate DMN.
        
        Called by CEN when switching back to active mode.
        """
        self.active = False
        self._cancel_event.set()
    
    async def _reflect(self) -> ReflectionResult:
        """
        Self-reflection: How did I perform?
        
        Queries analytics for recent performance and identifies weak areas.
        """
        from observability import query_metrics
        
        result = ReflectionResult(timestamp=datetime.now(timezone.utc))
        weak_areas = []
        
        try:
            # Query performance metrics
            performance = await query_metrics("""
                SELECT
                    metric_name,
                    AVG(metric_value) as avg_val,
                    MIN(metric_value) as min_val,
                    MAX(metric_value) as max_val,
                    COUNT(*) as count
                FROM metrics
                WHERE ts > now() - INTERVAL '24 hours'
                AND component IN ('response', 'mml', 'recall')
                GROUP BY metric_name
            """)
            
            result.performance_summary = {
                row["metric_name"]: {
                    "avg": row["avg_val"],
                    "min": row["min_val"],
                    "max": row["max_val"],
                    "count": row["count"],
                }
                for row in performance
            }
            
            # Check for high frustration rate
            frustration_data = await query_metrics("""
                SELECT COUNT(*) as frustrated
                FROM metrics
                WHERE ts > now() - INTERVAL '24 hours'
                AND metric_name = 'sentiment_frustrated'
            """)
            
            if frustration_data and frustration_data[0].get("frustrated", 0) > 5:
                weak_areas.append("High frustration rate — review failed interactions")
            
            # Check recall miss rate
            miss_data = await query_metrics("""
                SELECT AVG(metric_value) as miss_rate
                FROM metrics
                WHERE metric_name = 'recall_miss_rate'
                AND ts > now() - INTERVAL '24 hours'
            """)
            
            if miss_data and miss_data[0].get("miss_rate", 0) > 0.3:
                weak_areas.append("High recall miss rate (>30%) — knowledge gaps or index issues")
            
            # Check knowledge gaps
            from memory.types.meta import get_meta_memory
            meta = get_meta_memory()
            gaps = await meta.get_top_gaps(limit=5)
            
            if gaps:
                gap_topics = [g.topic[:50] for g in gaps]
                weak_areas.append(f"Top unresolved knowledge gaps: {gap_topics}")
            
        except Exception as e:
            weak_areas.append(f"Error during reflection: {str(e)}")
        
        result.weak_areas = weak_areas
        self._last_reflection = result
        
        # Log reflection
        from observability import record_metric
        await record_metric("dmn", "self_reflection", 1, result.to_dict())
        
        # Create remediation tasks for weak areas
        for area in weak_areas:
            await self._create_remediation_task(area)
        
        return result
    
    async def _plan(self) -> PlanningResult:
        """
        Spontaneous planning: What should I prepare for?
        
        Analyzes patterns and schedules proactive tasks.
        """
        if not self.active:
            return PlanningResult(timestamp=datetime.now(timezone.utc))
        
        from observability import query_metrics
        
        result = PlanningResult(timestamp=datetime.now(timezone.utc))
        
        try:
            # Find slow domains
            slow_domains = await query_metrics("""
                SELECT dimensions->>'domain' as domain,
                       AVG(metric_value) as avg_latency
                FROM metrics
                WHERE metric_name = 'recall_latency_ms'
                AND ts > now() - INTERVAL '7 days'
                GROUP BY dimensions->>'domain'
                HAVING AVG(metric_value) > 100
            """)
            
            for domain_data in slow_domains:
                domain = domain_data.get("domain")
                if domain:
                    result.slow_domains.append(domain)
                    await self._schedule_index_optimization(domain)
                    result.tasks_scheduled += 1
            
        except Exception:
            pass  # Planning is best-effort
        
        self._last_planning = result
        
        # Log planning
        from observability import record_metric
        await record_metric("dmn", "spontaneous_planning", 1, result.to_dict())
        
        return result
    
    async def _run_maintenance(self) -> None:
        """
        Run maintenance tasks via MML.
        
        Prioritizes based on self-reflection results.
        """
        if not self.active:
            return
        
        from memory.management import get_mml
        mml = get_mml()
        
        # Prioritize based on reflection
        priority_tasks = self._prioritize_maintenance()
        
        # Run MML background tasks
        # The MML will check self._cancel_event periodically
        try:
            # Consolidation
            if "consolidation" in priority_tasks and self.active:
                await mml.consolidate()
            
            # SFM promotion
            if "promotion" in priority_tasks and self.active:
                await mml.promote_to_sfm()
            
            # Optimization
            if "optimization" in priority_tasks and self.active:
                await mml.optimize_indexes()
            
            # Integrity check
            if "integrity" in priority_tasks and self.active:
                await mml.check_integrity()
            
        except asyncio.CancelledError:
            pass
    
    def _prioritize_maintenance(self) -> list[str]:
        """
        Prioritize maintenance tasks based on reflection.
        
        Returns:
            Ordered list of task types to run.
        """
        # Default order
        tasks = ["consolidation", "promotion", "optimization", "integrity"]
        
        if not self._last_reflection:
            return tasks
        
        # Boost based on weak areas
        priority_map = {
            "recall_miss": ["optimization", "consolidation"],
            "frustration": ["consolidation", "integrity"],
            "knowledge_gaps": ["consolidation"],
            "slow": ["optimization"],
        }
        
        boosted = []
        for area in self._last_reflection.weak_areas:
            area_lower = area.lower()
            for keyword, boost_tasks in priority_map.items():
                if keyword in area_lower:
                    for t in boost_tasks:
                        if t not in boosted:
                            boosted.append(t)
        
        # Boosted tasks first, then remaining
        remaining = [t for t in tasks if t not in boosted]
        return boosted + remaining
    
    async def _create_remediation_task(self, weak_area: str) -> None:
        """Create a PM task to address an identified weakness."""
        from memory.types.pm import get_prospective_memory
        
        pm = get_prospective_memory()
        
        # Schedule for next sleep window (e.g., 2 AM)
        now = datetime.now(timezone.utc)
        next_sleep = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if next_sleep <= now:
            next_sleep += timedelta(days=1)
        
        await pm.schedule_task(
            task_type="remediation",
            payload={
                "issue": weak_area,
                "action": "investigate_and_fix",
            },
            scheduled_at=next_sleep,
        )
    
    async def _schedule_index_optimization(self, domain: str) -> None:
        """Schedule FAISS index optimization for a slow domain."""
        from memory.types.pm import get_prospective_memory
        
        pm = get_prospective_memory()
        
        # Schedule for next sleep window
        now = datetime.now(timezone.utc)
        next_sleep = now.replace(hour=5, minute=0, second=0, microsecond=0)
        if next_sleep <= now:
            next_sleep += timedelta(days=1)
        
        await pm.schedule_task(
            task_type="maintenance",
            payload={
                "domain": domain,
                "action": "reindex_faiss",
            },
            scheduled_at=next_sleep,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_dmn_instance: Optional[DefaultModeNetwork] = None


def get_dmn() -> DefaultModeNetwork:
    """Get the singleton DMN instance."""
    global _dmn_instance
    if _dmn_instance is None:
        _dmn_instance = DefaultModeNetwork()
    return _dmn_instance

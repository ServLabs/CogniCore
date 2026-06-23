"""
Meta-Learning (Learning Type 7)

Learning HOW to learn — track which learning strategies are most effective
and allocate resources accordingly.

Algorithm: Learning effectiveness tracking + budget optimization
Trigger: Sleep mode (weekly review)
LLM: No (pure analytics)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from collections.abc import Callable

from logger import log
from observability import audit


@dataclass
class LearningEvent:
    """A single learning event."""
    event_id: str
    learning_type: str  # "reinforced", "generalization", etc.
    source: str  # What triggered it
    output_memory: str  # Where result was stored (sfm, mm, am, etc.)
    output_id: Optional[str]  # ID of created memory
    llm_cost: float  # Cost in USD
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    downstream_utility: Optional[float] = None  # Filled later
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "learning_type": self.learning_type,
            "source": self.source,
            "output_memory": self.output_memory,
            "output_id": self.output_id,
            "llm_cost": self.llm_cost,
            "timestamp": self.timestamp.isoformat(),
            "downstream_utility": self.downstream_utility,
        }


@dataclass
class LearningTypeStats:
    """Statistics for a learning type."""
    learning_type: str
    total_events: int
    avg_utility: float
    avg_cost: float
    roi: float  # utility / cost
    budget_allocation: float  # 0-1, percentage of budget
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_type": self.learning_type,
            "total_events": self.total_events,
            "avg_utility": self.avg_utility,
            "avg_cost": self.avg_cost,
            "roi": self.roi,
            "budget_allocation": self.budget_allocation,
        }


@dataclass
class MetaLearningResult:
    """Result of meta-learning analysis."""
    stats_by_type: dict[str, LearningTypeStats]
    budget_adjustments: dict[str, float]  # learning_type -> new allocation
    recommendations: list[str]
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "stats_by_type": {k: v.to_dict() for k, v in self.stats_by_type.items()},
            "budget_adjustments": self.budget_adjustments,
            "recommendations": self.recommendations,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
        }


# Learning types and their default costs
LEARNING_TYPES = {
    "reinforced": {"cost": 0.0, "default_budget": 0.15},
    "generalization": {"cost": 0.05, "default_budget": 0.10},
    "abstraction": {"cost": 0.01, "default_budget": 0.10},
    "analogical": {"cost": 0.05, "default_budget": 0.10},
    "corrective": {"cost": 0.01, "default_budget": 0.15},
    "transfer": {"cost": 0.05, "default_budget": 0.10},
    "meta": {"cost": 0.0, "default_budget": 0.05},
    "incremental": {"cost": 0.0, "default_budget": 0.10},
    "observational": {"cost": 0.01, "default_budget": 0.10},
    "contrastive": {"cost": 0.01, "default_budget": 0.05},
}


class MetaLearner:
    """
    Tracks learning effectiveness and optimizes budget allocation.
    
    Monitors which learning types produce useful knowledge
    and adjusts resource allocation accordingly.
    
    Attributes:
        events: List of learning events.
        budget_allocations: Current budget per learning type.
    """
    
    def __init__(
        self,
        get_access_count: Optional[Callable[[str, str], int]] = None,
        get_sentiment_after: Optional[Callable[[datetime], float]] = None,
    ):
        """
        Initialize meta-learner.
        
        Args:
            get_access_count: Function to get memory access count.
            get_sentiment_after: Function to get avg sentiment after timestamp.
        """
        self.get_access_count = get_access_count
        self.get_sentiment_after = get_sentiment_after
        self.events: list[LearningEvent] = []
        self.budget_allocations = {
            lt: info["default_budget"]
            for lt, info in LEARNING_TYPES.items()
        }
    
    def log_event(
        self,
        learning_type: str,
        source: str,
        output_memory: str,
        output_id: Optional[str] = None,
        llm_cost: float = 0.0,
    ) -> LearningEvent:
        """
        Log a learning event.
        
        Args:
            learning_type: Type of learning.
            source: What triggered it.
            output_memory: Where result was stored.
            output_id: ID of created memory.
            llm_cost: Cost in USD.
            
        Returns:
            Created LearningEvent.
        """
        import uuid
        
        event = LearningEvent(
            event_id=str(uuid.uuid4()),
            learning_type=learning_type,
            source=source,
            output_memory=output_memory,
            output_id=output_id,
            llm_cost=llm_cost,
        )
        
        self.events.append(event)
        
        # Keep only recent events (last 10000)
        if len(self.events) > 10000:
            self.events = self.events[-10000:]
        
        return event
    
    async def measure_utility(self, lookback_days: int = 7) -> int:
        """
        Measure utility of past learning events.
        
        Args:
            lookback_days: How far back to look.
            
        Returns:
            Number of events updated.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        updated = 0
        
        for event in self.events:
            # Skip if already measured or too recent
            if event.downstream_utility is not None:
                continue
            if event.timestamp > cutoff:
                continue
            
            utility = await self._compute_utility(event)
            event.downstream_utility = utility
            updated += 1
        
        return updated
    
    async def _compute_utility(self, event: LearningEvent) -> float:
        """Compute utility of a learning event."""
        utility = 0.0
        
        # Access count component (0-0.5)
        if self.get_access_count and event.output_id:
            access_count = self.get_access_count(event.output_memory, event.output_id)
            # Normalize: 10+ accesses = max utility
            utility += min(0.5, access_count / 20.0)
        
        # Sentiment component (0-0.5)
        if self.get_sentiment_after:
            sentiment = self.get_sentiment_after(event.timestamp)
            # Normalize: sentiment is typically -1 to 1
            utility += (sentiment + 1) / 4.0  # Maps to 0-0.5
        
        return utility
    
    def analyze(self) -> MetaLearningResult:
        """
        Analyze learning effectiveness and compute new budget allocations.
        
        Returns:
            MetaLearningResult with stats and recommendations.
        """
        # Compute stats by learning type
        stats_by_type: dict[str, LearningTypeStats] = {}
        
        for learning_type in LEARNING_TYPES:
            type_events = [e for e in self.events if e.learning_type == learning_type]
            measured_events = [e for e in type_events if e.downstream_utility is not None]
            
            if not type_events:
                stats_by_type[learning_type] = LearningTypeStats(
                    learning_type=learning_type,
                    total_events=0,
                    avg_utility=0.0,
                    avg_cost=LEARNING_TYPES[learning_type]["cost"],
                    roi=0.0,
                    budget_allocation=self.budget_allocations[learning_type],
                )
                continue
            
            avg_utility = (
                sum(e.downstream_utility for e in measured_events) / len(measured_events)
                if measured_events else 0.0
            )
            avg_cost = (
                sum(e.llm_cost for e in type_events) / len(type_events)
                if type_events else LEARNING_TYPES[learning_type]["cost"]
            )
            
            # ROI = utility / cost (avoid division by zero)
            roi = avg_utility / max(avg_cost, 0.001)
            
            stats_by_type[learning_type] = LearningTypeStats(
                learning_type=learning_type,
                total_events=len(type_events),
                avg_utility=avg_utility,
                avg_cost=avg_cost,
                roi=roi,
                budget_allocation=self.budget_allocations[learning_type],
            )
        
        # Compute new budget allocations based on ROI
        budget_adjustments = self._compute_budget_adjustments(stats_by_type)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stats_by_type)
        
        result = MetaLearningResult(
            stats_by_type=stats_by_type,
            budget_adjustments=budget_adjustments,
            recommendations=recommendations,
        )
        
        audit.log_raw(
            "learning",
            "meta_analysis",
            "mml",
            "completed",
            types_analyzed=len(stats_by_type),
        )
        
        return result
    
    def _compute_budget_adjustments(
        self,
        stats: dict[str, LearningTypeStats],
    ) -> dict[str, float]:
        """Compute new budget allocations based on ROI."""
        # Get ROIs
        rois = {lt: s.roi for lt, s in stats.items()}
        total_roi = sum(rois.values())
        
        if total_roi == 0:
            # No data, keep defaults
            return {lt: LEARNING_TYPES[lt]["default_budget"] for lt in LEARNING_TYPES}
        
        # Allocate proportionally to ROI
        # But keep minimum 5% for each type to allow exploration
        min_allocation = 0.05
        remaining = 1.0 - (min_allocation * len(LEARNING_TYPES))
        
        adjustments = {}
        for lt in LEARNING_TYPES:
            roi_share = rois[lt] / total_roi if total_roi > 0 else 0
            adjustments[lt] = min_allocation + (remaining * roi_share)
        
        return adjustments
    
    def _generate_recommendations(
        self,
        stats: dict[str, LearningTypeStats],
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Find high and low performers
        sorted_by_roi = sorted(stats.values(), key=lambda s: s.roi, reverse=True)
        
        if sorted_by_roi:
            top = sorted_by_roi[0]
            if top.roi > 0:
                recommendations.append(
                    f"Increase budget for '{top.learning_type}' (ROI: {top.roi:.2f})"
                )
            
            bottom = sorted_by_roi[-1]
            if bottom.roi < 0.1 and bottom.total_events > 10:
                recommendations.append(
                    f"Consider reducing '{bottom.learning_type}' (ROI: {bottom.roi:.2f})"
                )
        
        # Check for underutilized types
        for lt, s in stats.items():
            if s.total_events < 5:
                recommendations.append(
                    f"'{lt}' has few events ({s.total_events}), consider triggering more"
                )
        
        return recommendations
    
    def apply_adjustments(self, adjustments: dict[str, float]) -> None:
        """Apply budget adjustments."""
        for lt, allocation in adjustments.items():
            if lt in self.budget_allocations:
                self.budget_allocations[lt] = allocation
        
        log.info(f"Meta-learning: Applied budget adjustments: {adjustments}")
    
    def get_budget(self, learning_type: str) -> float:
        """Get current budget allocation for a learning type."""
        return self.budget_allocations.get(learning_type, 0.1)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_meta_learner: Optional[MetaLearner] = None


def get_meta_learner() -> MetaLearner:
    """Get the singleton MetaLearner instance."""
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = MetaLearner()
    return _meta_learner

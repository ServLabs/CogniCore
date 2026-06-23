"""
LLM Budget Manager

Controls LLM costs for background tasks with:
- Per-cycle budget caps
- Priority queue execution
- Model tier selection
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Enums (System values - universal)
# ══════════════════════════════════════════════════════════════════════════════

class ModelTier(Enum):
    """Model tier for cost optimization."""
    EXPENSIVE = "expensive"  # GPT-4o, Claude Sonnet - reasoning, complex tasks
    CHEAP = "cheap"          # GPT-4o-mini, local - summarization, simple tasks
    FREE = "free"            # No LLM - embedding, FAISS, graph, SQL


class TaskPriority(Enum):
    """Task priority levels."""
    P0_CRITICAL = 0   # Always runs: conflict resolution, critical consolidation
    P1_IMPORTANT = 1  # Runs if budget: SFM promotion, experience generalization
    P2_ROUTINE = 2    # Runs if budget remains: linking, coherence scan


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskCost:
    """Estimated and actual cost of a task."""
    estimated_calls: int = 1
    estimated_tokens: int = 1000
    estimated_cost_usd: float = 0.01
    actual_calls: int = 0
    actual_tokens: int = 0
    actual_cost_usd: float = 0.0


@dataclass
class BudgetState:
    """Current budget state."""
    max_calls: int
    max_cost_usd: float
    used_calls: int = 0
    used_cost_usd: float = 0.0
    cycle_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.used_calls)
    
    @property
    def remaining_cost_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.used_cost_usd)
    
    @property
    def utilization_pct(self) -> float:
        if self.max_calls == 0:
            return 0.0
        return (self.used_calls / self.max_calls) * 100
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "max_cost_usd": self.max_cost_usd,
            "used_calls": self.used_calls,
            "used_cost_usd": self.used_cost_usd,
            "remaining_calls": self.remaining_calls,
            "remaining_cost_usd": self.remaining_cost_usd,
            "utilization_pct": self.utilization_pct,
            "cycle_start": self.cycle_start.isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Budget Manager
# ══════════════════════════════════════════════════════════════════════════════

class BudgetManager:
    """
    Manages LLM budget for background tasks.
    
    Provides:
    - Per-cycle budget caps (calls and cost)
    - Affordability checks
    - Cost tracking and deduction
    - Model tier recommendations
    
    Attributes:
        max_calls: Maximum LLM calls per cycle.
        max_cost_usd: Maximum cost per cycle.
    """
    
    # Default costs per model tier (rough estimates)
    COST_PER_CALL = {
        ModelTier.EXPENSIVE: 0.03,  # ~$0.03 per call (GPT-4o)
        ModelTier.CHEAP: 0.001,     # ~$0.001 per call (GPT-4o-mini)
        ModelTier.FREE: 0.0,        # No cost
    }
    
    def __init__(
        self,
        max_calls: int = 500,
        max_cost_usd: float = 5.0,
    ):
        """
        Initialize Budget Manager.
        
        Args:
            max_calls: Maximum LLM calls per cycle.
            max_cost_usd: Maximum cost per cycle in USD.
        """
        self._state = BudgetState(
            max_calls=max_calls,
            max_cost_usd=max_cost_usd,
        )
    
    def can_afford(
        self,
        estimated_calls: int = 1,
        estimated_cost_usd: Optional[float] = None,
        model_tier: ModelTier = ModelTier.CHEAP,
    ) -> bool:
        """
        Check if budget can afford a task.
        
        Args:
            estimated_calls: Estimated LLM calls.
            estimated_cost_usd: Estimated cost (auto-calculated if None).
            model_tier: Model tier for cost estimation.
            
        Returns:
            True if affordable.
        """
        if model_tier == ModelTier.FREE:
            return True
        
        if estimated_cost_usd is None:
            estimated_cost_usd = estimated_calls * self.COST_PER_CALL[model_tier]
        
        return (
            self._state.remaining_calls >= estimated_calls
            and self._state.remaining_cost_usd >= estimated_cost_usd
        )
    
    def deduct(
        self,
        actual_calls: int,
        actual_cost_usd: float,
    ) -> None:
        """
        Deduct from budget after task execution.
        
        Args:
            actual_calls: Actual LLM calls made.
            actual_cost_usd: Actual cost incurred.
        """
        self._state.used_calls += actual_calls
        self._state.used_cost_usd += actual_cost_usd
    
    def reset_cycle(self) -> BudgetState:
        """
        Reset budget for new cycle.
        
        Returns:
            Previous cycle's state for logging.
        """
        previous = self._state
        self._state = BudgetState(
            max_calls=previous.max_calls,
            max_cost_usd=previous.max_cost_usd,
        )
        return previous
    
    def get_state(self) -> BudgetState:
        """Get current budget state."""
        return self._state
    
    def recommend_tier(
        self,
        priority: TaskPriority,
        task_type: str,
    ) -> ModelTier:
        """
        Recommend model tier based on task priority and type.
        
        Args:
            priority: Task priority.
            task_type: Type of task.
            
        Returns:
            Recommended ModelTier.
        """
        # P0 critical tasks get expensive model
        if priority == TaskPriority.P0_CRITICAL:
            if self.can_afford(5, model_tier=ModelTier.EXPENSIVE):
                return ModelTier.EXPENSIVE
            return ModelTier.CHEAP
        
        # P1 important tasks get cheap model
        if priority == TaskPriority.P1_IMPORTANT:
            return ModelTier.CHEAP
        
        # P2 routine tasks - prefer free, fallback to cheap
        if task_type in ("linking", "embedding", "indexing"):
            return ModelTier.FREE
        
        return ModelTier.CHEAP
    
    def estimate_task_cost(
        self,
        task_type: str,
        item_count: int = 1,
    ) -> TaskCost:
        """
        Estimate cost for a task type.
        
        Args:
            task_type: Type of task.
            item_count: Number of items to process.
            
        Returns:
            Estimated TaskCost.
        """
        # Cost estimates per task type
        estimates = {
            "consolidation": (10, ModelTier.EXPENSIVE),
            "conflict_resolution": (5, ModelTier.EXPENSIVE),
            "experience_generalization": (5, ModelTier.EXPENSIVE),
            "sfm_promotion": (1, ModelTier.CHEAP),
            "coherence_scan": (1, ModelTier.CHEAP),
            "gap_analysis": (1, ModelTier.CHEAP),
            "linking": (0, ModelTier.FREE),
            "embedding": (0, ModelTier.FREE),
            "indexing": (0, ModelTier.FREE),
        }
        
        calls_per_item, tier = estimates.get(task_type, (1, ModelTier.CHEAP))
        total_calls = calls_per_item * item_count
        cost_per_call = self.COST_PER_CALL[tier]
        
        return TaskCost(
            estimated_calls=total_calls,
            estimated_tokens=total_calls * 1000,  # Rough estimate
            estimated_cost_usd=total_calls * cost_per_call,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_budget_instance: Optional[BudgetManager] = None


def get_budget_manager() -> BudgetManager:
    """Get the singleton BudgetManager instance."""
    global _budget_instance
    if _budget_instance is None:
        _budget_instance = BudgetManager()
    return _budget_instance

"""
Memory Management Layer (MML)

The agent's subconscious — autonomous processes that maintain, optimize,
and evolve the memory system.

This package provides:
- MML orchestrator (real-time + background operations)
- Recall engine (tiered search, ranking, fusion)
- Budget manager (LLM cost control)
- Learning types (10 learning algorithms)
- Maintenance daemons (7 background tasks)
"""

from memory.management.mml import (
    MemoryManagementLayer,
    get_mml,
    recall,
    consolidate,
    start_background_tasks,
    stop_background_tasks,
)
from memory.management.recall import (
    RecallEngine,
    RecallResult,
    RecallConfig,
    get_recall_engine,
)
from memory.management.budget import (
    BudgetManager,
    ModelTier,
    TaskPriority,
    get_budget_manager,
)

__all__ = [
    # MML
    "MemoryManagementLayer",
    "get_mml",
    "recall",
    "consolidate",
    "start_background_tasks",
    "stop_background_tasks",
    # Recall
    "RecallEngine",
    "RecallResult",
    "RecallConfig",
    "get_recall_engine",
    # Budget
    "BudgetManager",
    "ModelTier",
    "TaskPriority",
    "get_budget_manager",
]

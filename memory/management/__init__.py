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
)
from memory.management.recall import RecallResult

__all__ = [
    "MemoryManagementLayer",
    "get_mml",
    "recall",
    "consolidate",
    "RecallResult",
]

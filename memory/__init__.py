"""
CogniCore Memory Layer

The memory system is modeled after human cognition, with distinct memory types
serving different purposes.

This is the **sole public API** for the memory package.
All external modules must import from here — never from sub-packages directly.

Usage:
    from memory import get_mml, recall, add_user_message, get_meta_memory
"""

# ── Types ──
from memory.types import (
    get_identity,
    get_full_system_prompt,
    get_compact_system_prompt,
    get_prospective_memory,
    get_working_memory,
    add_user_message,
    add_assistant_message,
    get_emotional_memory,
    get_short_form_memory,
    get_meta_memory,
)

# ── Management ──
from memory.management import (
    MemoryManagementLayer,
    get_mml,
    recall,
    consolidate,
    RecallResult,
)

# ── Learning (used by control/dmn.py, interfaces/scheduled/tasks.py) ──
from memory.management.learning import (
    get_generalizer,
    get_abstraction_learner,
    get_analogical_learner,
    get_transfer_learner,
    get_contrastive_learner,
    get_meta_learner,
)

__all__ = [
    # Types — Identity
    "get_identity",
    "get_full_system_prompt",
    "get_compact_system_prompt",
    # Types — Getters & Convenience
    "get_prospective_memory",
    "get_working_memory",
    "add_user_message",
    "add_assistant_message",
    "get_emotional_memory",
    "get_short_form_memory",
    "get_meta_memory",
    # Management
    "MemoryManagementLayer",
    "get_mml",
    "recall",
    "consolidate",
    "RecallResult",
    # Learning
    "get_generalizer",
    "get_abstraction_learner",
    "get_analogical_learner",
    "get_transfer_learner",
    "get_contrastive_learner",
    "get_meta_learner",
]

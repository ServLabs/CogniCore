"""
CogniCore Memory Layer

The memory system is modeled after human cognition, with distinct memory types
serving different purposes.

Import chain:
    memory/types/abm.py  →  memory/types/__init__.py  →  memory/__init__.py
    (implementation)         (types package)              (memory package)

Usage:
    from memory import get_identity, get_full_system_prompt
"""

from memory.types import (
    AutobiographicalMemory,
    get_identity,
    get_full_system_prompt,
    get_compact_system_prompt,
)

__all__ = [
    # ABM - Agent Identity
    "AutobiographicalMemory",
    "get_identity",
    "get_full_system_prompt",
    "get_compact_system_prompt",
]

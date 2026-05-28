"""
CogniCore Memory Layer

The memory system is modeled after human cognition, with distinct memory types
serving different purposes.
"""

from memory.types.abm import (
    AutobiographicalMemory,
    get_identity,
    get_system_prompt,
    get_compact_prompt,
)

__all__ = [
    "AutobiographicalMemory",
    "get_identity",
    "get_system_prompt",
    "get_compact_prompt",
]

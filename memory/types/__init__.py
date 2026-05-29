"""
Memory Types

Each memory type serves a distinct cognitive purpose:
- ABM: Autobiographical Memory - Agent identity
- PM: Prospective Memory - Scheduled tasks
- WM: Working Memory - Active conversation context
- EM: Emotional Memory - User sentiment and preferences
- AM: Associative Memory - Knowledge graph
- MM: Motor Memory - Learned procedures
- SFM: Short Form Memory - Compressed facts
- LFM: Long Form Memory - Full documents
- Meta: Meta Memory - Memory about memories
"""

from memory.types.abm import (
    AutobiographicalMemory,
    get_identity,
    get_full_system_prompt,
    get_compact_system_prompt,
)

__all__ = [
    # ABM - Agent Identity
    "AutobiographicalMemory",  # Class (for type hints and testing)
    "get_identity",            # Singleton accessor
    "get_full_system_prompt",  # Full identity prompt
    "get_compact_system_prompt",  # Compact identity prompt
]

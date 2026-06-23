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

Only exports consumed by modules outside the memory package.
Internal management modules import directly from submodules.
"""

from memory.types.abm import get_identity, get_full_system_prompt, get_compact_system_prompt
from memory.types.pm import get_prospective_memory
from memory.types.wm import add_user_message, add_assistant_message, get_working_memory
from memory.types.em import get_emotional_memory
from memory.types.sfm import get_short_form_memory
from memory.types.meta import get_meta_memory

__all__ = [
    # ABM - Agent Identity
    "get_identity",
    "get_full_system_prompt",
    "get_compact_system_prompt",
    # PM - Scheduled Tasks
    "get_prospective_memory",
    # WM - Conversation Context
    "add_user_message",
    "add_assistant_message",
    "get_working_memory",
    # EM - Sentiment & Preferences
    "get_emotional_memory",
    # SFM - Facts
    "get_short_form_memory",
    # Meta - Routing
    "get_meta_memory",
]

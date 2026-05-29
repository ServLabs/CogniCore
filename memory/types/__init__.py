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
from memory.types.pm import (
    Schedule,
    ScheduleStatus,
    ExecutionLog,
    ExecutionStatus,
    ProspectiveMemory,
    get_prospective_memory,
    schedule_task,
    cancel_task,
)
from memory.types.wm import (
    Message,
    MessageRole,
    ConversationSummary,
    Conversation,
    WorkingMemory,
    get_working_memory,
    add_user_message,
    add_assistant_message,
    get_context,
)

__all__ = [
    # ABM - Agent Identity
    "AutobiographicalMemory",
    "get_identity",
    "get_full_system_prompt",
    "get_compact_system_prompt",
    # PM - Scheduled Tasks
    "Schedule",
    "ScheduleStatus",
    "ExecutionLog",
    "ExecutionStatus",
    "ProspectiveMemory",
    "get_prospective_memory",
    "schedule_task",
    "cancel_task",
    # WM - Conversation Context
    "Message",
    "MessageRole",
    "ConversationSummary",
    "Conversation",
    "WorkingMemory",
    "get_working_memory",
    "add_user_message",
    "add_assistant_message",
    "get_context",
]

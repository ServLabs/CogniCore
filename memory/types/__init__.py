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
from memory.types.em import (
    SentimentEvent,
    SessionSentiment,
    UserPreferences,
    EmotionalMemory,
    get_emotional_memory,
    record_sentiment,
    get_preferences,
    get_calibration_hints,
)
from memory.types.am import (
    Entity,
    Relationship,
    SearchResult,
    AssociativeMemory,
    get_associative_memory,
    add_entity,
    link_entities,
    search_knowledge,
)
from memory.types.mm import (
    Procedure,
    ProcedureResult,
    MotorMemory,
    get_motor_memory,
    find_procedure,
    execute_procedure,
    learn_procedure,
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
    # EM - Sentiment & Preferences
    "SentimentEvent",
    "SessionSentiment",
    "UserPreferences",
    "EmotionalMemory",
    "get_emotional_memory",
    "record_sentiment",
    "get_preferences",
    "get_calibration_hints",
    # AM - Knowledge Graph
    "Entity",
    "Relationship",
    "SearchResult",
    "AssociativeMemory",
    "get_associative_memory",
    "add_entity",
    "link_entities",
    "search_knowledge",
    # MM - Procedures
    "Procedure",
    "ProcedureResult",
    "MotorMemory",
    "get_motor_memory",
    "find_procedure",
    "execute_procedure",
    "learn_procedure",
]

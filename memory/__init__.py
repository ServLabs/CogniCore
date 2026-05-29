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
    # ABM
    AutobiographicalMemory,
    get_identity,
    get_full_system_prompt,
    get_compact_system_prompt,
    # PM
    Schedule,
    ScheduleStatus,
    ExecutionLog,
    ExecutionStatus,
    ProspectiveMemory,
    get_prospective_memory,
    schedule_task,
    cancel_task,
    # WM
    Message,
    MessageRole,
    ConversationSummary,
    Conversation,
    WorkingMemory,
    get_working_memory,
    add_user_message,
    add_assistant_message,
    get_context,
    # EM
    SentimentEvent,
    SessionSentiment,
    UserPreferences,
    EmotionalMemory,
    get_emotional_memory,
    record_sentiment,
    get_preferences,
    get_calibration_hints,
    # AM
    Entity,
    Relationship,
    SearchResult,
    AssociativeMemory,
    get_associative_memory,
    add_entity,
    link_entities,
    search_knowledge,
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
]

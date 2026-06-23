"""
Control Events

Typed event hierarchy for the control layer.
Every event flowing through Salience → CEN is a concrete subclass
with validated fields — no dict[str, Any] payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Agent Mode
# ══════════════════════════════════════════════════════════════════════════════

class AgentMode(Enum):
    """Operating mode of the agent."""
    ACTIVE = "active"
    IDLE = "idle"


# ══════════════════════════════════════════════════════════════════════════════
# Base Event
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BaseEvent:
    """
    Base class for all control events.

    Every event has a type tag, creation timestamp, optional deadline,
    and a source identifier.
    """
    type: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    source: str = "system"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "source": self.source,
        }


# ══════════════════════════════════════════════════════════════════════════════
# User Events
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserMessageEvent(BaseEvent):
    """A user sent a chat message."""
    type: str = field(default="user_message", init=False)
    message: str = ""
    user_id: str = ""
    convo_id: str = ""
    channel: Any = None  # PipelineChannel — Any to avoid circular import

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"user_id": self.user_id, "convo_id": self.convo_id})
        return d


@dataclass
class UserFrustratedEvent(BaseEvent):
    """User frustration detected (repeated questions, complaints)."""
    type: str = field(default="user_frustrated", init=False)
    message: str = ""
    user_id: str = ""
    convo_id: str = ""
    channel: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"user_id": self.user_id, "convo_id": self.convo_id})
        return d


# ══════════════════════════════════════════════════════════════════════════════
# Scheduled / Background Events
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScheduledTaskEvent(BaseEvent):
    """A scheduled task is due for execution."""
    type: str = field(default="scheduled_task_due", init=False)
    task_id: str = ""
    task_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"task_id": self.task_id, "task_type": self.task_type})
        return d


@dataclass
class SubAgentCompleteEvent(BaseEvent):
    """A background sub-agent finished its work."""
    type: str = field(default="sub_agent_complete", init=False)
    task_name: str = ""
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"task_name": self.task_name})
        return d


# ══════════════════════════════════════════════════════════════════════════════
# System Events
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthAlertEvent(BaseEvent):
    """System health issue detected."""
    type: str = field(default="health_alert", init=False)
    alert_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"alert_type": self.alert_type})
        return d


@dataclass
class ResourcePressureEvent(BaseEvent):
    """Resource pressure detected (LLM budget, memory, CPU)."""
    type: str = field(default="resource_pressure", init=False)
    resource: str = ""  # "llm_budget" | "memory" | "cpu"
    utilization: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"resource": self.resource, "utilization": self.utilization})
        return d


@dataclass
class ConflictDetectedEvent(BaseEvent):
    """Memory conflict detected."""
    type: str = field(default="conflict_detected", init=False)
    fact_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"fact_ids": self.fact_ids})
        return d


# ══════════════════════════════════════════════════════════════════════════════
# Mode Events (internal)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModeSwitchEvent(BaseEvent):
    """CEN should switch agent mode."""
    type: str = field(default="mode_switch", init=False)
    target_mode: AgentMode = AgentMode.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"target_mode": self.target_mode.value})
        return d

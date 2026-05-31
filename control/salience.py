"""
Salience Network

The attention/priority arbiter — decides what deserves the agent's
resources RIGHT NOW. Always on, lightweight, event-driven.

Analogous to the human salience network that detects what's important
and triggers mode switches.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable

from core import config


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class EventType(Enum):
    """Types of events the salience network handles."""
    USER_MESSAGE = "user_message"
    USER_FRUSTRATED = "user_frustrated"
    SCHEDULED_TASK_DUE = "scheduled_task_due"
    SUB_AGENT_COMPLETE = "sub_agent_complete"
    CONFLICT_DETECTED = "conflict_detected"
    HEALTH_ALERT = "health_alert"
    RESOURCE_PRESSURE = "resource_pressure"
    ANOMALY = "anomaly"
    MAINTENANCE = "maintenance"


class AgentMode(Enum):
    """Operating mode of the agent."""
    ACTIVE = "active"  # Serving users
    IDLE = "idle"      # DMN takes over


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentEvent:
    """An event for the salience network to process."""
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    source: str = "system"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "source": self.source,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Mode Detector
# ══════════════════════════════════════════════════════════════════════════════

class ModeDetector:
    """
    Detects mode switches between active and idle.
    
    Attributes:
        idle_threshold_seconds: Seconds without user interaction before idle.
    """
    
    def __init__(self, idle_threshold_seconds: int = 300):
        """
        Initialize ModeDetector.
        
        Args:
            idle_threshold_seconds: Seconds before switching to idle (default 5 min).
        """
        self.idle_threshold_seconds = idle_threshold_seconds
        self.last_user_interaction = datetime.now(timezone.utc)
        self.current_mode = AgentMode.ACTIVE
    
    def on_user_message(self) -> Optional[str]:
        """
        Record user interaction.
        
        Returns:
            "switch_to_active" if mode changed, None otherwise.
        """
        self.last_user_interaction = datetime.now(timezone.utc)
        
        if self.current_mode == AgentMode.IDLE:
            self.current_mode = AgentMode.ACTIVE
            return "switch_to_active"
        
        return None
    
    def check_idle(self) -> Optional[str]:
        """
        Check if agent should switch to idle.
        
        Returns:
            "switch_to_idle" if mode should change, None otherwise.
        """
        if self.current_mode == AgentMode.ACTIVE:
            now = datetime.now(timezone.utc)
            idle_time = (now - self.last_user_interaction).total_seconds()
            
            if idle_time > self.idle_threshold_seconds:
                self.current_mode = AgentMode.IDLE
                return "switch_to_idle"
        
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Salience Network
# ══════════════════════════════════════════════════════════════════════════════

class SalienceNetwork:
    """
    Priority arbiter for agent events.
    
    Provides:
    - Event scoring and prioritization
    - Priority queue management
    - Mode detection (active/idle)
    - High-priority event notification
    
    Attributes:
        mode_detector: Detector for active/idle mode.
    """
    
    # Priority tiers (higher = more important)
    TYPE_WEIGHTS = {
        "user_message": 100,
        "user_frustrated": 95,
        "health_alert": 80,
        "resource_pressure": 75,
        "scheduled_task_due": 70,
        "sub_agent_complete": 60,
        "conflict_detected": 50,
        "anomaly": 40,
        "maintenance": 10,
    }
    
    # Weights for scoring formula
    W_TYPE = 0.4
    W_URGENCY = 0.25
    W_IMPACT = 0.25
    W_DECAY = 0.1
    
    def __init__(self, idle_threshold_seconds: int = 300):
        """
        Initialize SalienceNetwork.
        
        Args:
            idle_threshold_seconds: Seconds before idle mode.
        """
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.mode_detector = ModeDetector(idle_threshold_seconds)
        self.listeners: list[Callable[[AgentEvent, float], None]] = []
        self._high_priority_threshold = 80
    
    async def ingest_event(self, event: AgentEvent) -> float:
        """
        Score and queue an incoming event.
        
        Args:
            event: The event to process.
            
        Returns:
            The priority score assigned.
        """
        # Update mode detector for user messages
        if event.type == "user_message":
            mode_signal = self.mode_detector.on_user_message()
            if mode_signal:
                # Notify listeners of mode change
                await self._notify_mode_change(mode_signal)
        
        # Score the event
        priority = self._score(event)
        
        # Add to queue (negative for max-priority-first)
        await self.event_queue.put((-priority, id(event), event))
        
        # Notify CEN immediately for high-priority events
        if priority >= self._high_priority_threshold:
            await self._notify_high_priority(event, priority)
        
        return priority
    
    def _score(self, event: AgentEvent) -> float:
        """
        Calculate priority score for an event.
        
        Formula: P = w_type * T + w_urgency * U + w_impact * I + w_decay * D
        """
        T = self.TYPE_WEIGHTS.get(event.type, 10)
        U = self._urgency_score(event)
        I = self._impact_score(event)
        D = self._decay_boost(event.created_at)
        
        return self.W_TYPE * T + self.W_URGENCY * U + self.W_IMPACT * I + self.W_DECAY * D
    
    def _urgency_score(self, event: AgentEvent) -> float:
        """Calculate urgency based on deadline."""
        if event.deadline is None:
            return 50  # No deadline = medium urgency
        
        now = datetime.now(timezone.utc)
        seconds_until = (event.deadline - now).total_seconds()
        
        if seconds_until <= 0:
            return 100  # Overdue
        if seconds_until <= 60:
            return 90   # Due within 1 min
        if seconds_until <= 300:
            return 70   # Due within 5 min
        if seconds_until <= 3600:
            return 40   # Due within 1 hour
        
        return 20
    
    def _impact_score(self, event: AgentEvent) -> float:
        """Calculate impact score based on event type."""
        # User-facing events have highest impact
        if event.type in ("user_message", "user_frustrated"):
            return 100
        
        # System health events
        if event.type in ("health_alert", "resource_pressure"):
            return 80
        
        # Task completion
        if event.type in ("scheduled_task_due", "sub_agent_complete"):
            return 60
        
        # Background tasks
        return 30
    
    def _decay_boost(self, created_at: datetime) -> float:
        """Calculate decay boost for waiting events."""
        now = datetime.now(timezone.utc)
        wait_seconds = (now - created_at).total_seconds()
        
        # +10 per minute waiting, cap at 100
        return min(100, wait_seconds / 60 * 10)
    
    async def get_next(self) -> AgentEvent:
        """
        Get the next highest-priority event.
        
        Called by CEN to get work.
        
        Returns:
            The highest priority event.
        """
        _, _, event = await self.event_queue.get()
        return event
    
    def has_events(self) -> bool:
        """Check if there are events waiting."""
        return not self.event_queue.empty()
    
    def register_listener(self, callback: Callable[[AgentEvent, float], None]) -> None:
        """Register a listener for high-priority events."""
        self.listeners.append(callback)
    
    async def _notify_high_priority(self, event: AgentEvent, priority: float) -> None:
        """Notify listeners of high-priority event."""
        for listener in self.listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event, priority)
                else:
                    listener(event, priority)
            except Exception:
                pass  # Don't let listener errors break the network
    
    async def _notify_mode_change(self, mode_signal: str) -> None:
        """Notify of mode change."""
        event = AgentEvent(
            type=f"mode_{mode_signal}",
            payload={"signal": mode_signal},
        )
        for listener in self.listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event, 100)
                else:
                    listener(event, 100)
            except Exception:
                pass
    
    def check_idle(self) -> Optional[str]:
        """Check if agent should switch to idle mode."""
        return self.mode_detector.check_idle()


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_salience_instance: Optional[SalienceNetwork] = None


def get_salience_network() -> SalienceNetwork:
    """Get the singleton SalienceNetwork instance."""
    global _salience_instance
    if _salience_instance is None:
        _salience_instance = SalienceNetwork()
    return _salience_instance

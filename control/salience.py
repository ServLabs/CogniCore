"""
Salience Network

The attention/priority arbiter — decides what deserves the agent's
resources RIGHT NOW. Always on, lightweight, event-driven.

Scores incoming events, manages a priority queue, and detects
active/idle mode switches. Workers in CEN pull from this queue.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from config import config
from control.events import (
    BaseEvent,
    AgentMode,
    UserMessageEvent,
    UserFrustratedEvent,
    ModeSwitchEvent,
)


# ══════════════════════════════════════════════════════════════════════════════
# Mode Detector
# ══════════════════════════════════════════════════════════════════════════════

class ModeDetector:
    """
    Detects mode switches between active and idle.

    Tracks the last user interaction timestamp and fires a mode-switch
    signal when the idle threshold is exceeded.
    """

    def __init__(self, idle_threshold_seconds: int = 300):
        self.idle_threshold_seconds = idle_threshold_seconds
        self.last_user_interaction = datetime.now(timezone.utc)
        self.current_mode = AgentMode.ACTIVE

    def on_user_message(self) -> Optional[AgentMode]:
        """
        Record user interaction.

        Returns:
            AgentMode.ACTIVE if mode changed from idle, None otherwise.
        """
        self.last_user_interaction = datetime.now(timezone.utc)

        if self.current_mode == AgentMode.IDLE:
            self.current_mode = AgentMode.ACTIVE
            return AgentMode.ACTIVE

        return None

    def check_idle(self) -> Optional[AgentMode]:
        """
        Check if agent should switch to idle.

        Returns:
            AgentMode.IDLE if mode changed, None otherwise.
        """
        if self.current_mode == AgentMode.ACTIVE:
            now = datetime.now(timezone.utc)
            idle_time = (now - self.last_user_interaction).total_seconds()

            if idle_time > self.idle_threshold_seconds:
                self.current_mode = AgentMode.IDLE
                return AgentMode.IDLE

        return None


# ══════════════════════════════════════════════════════════════════════════════
# Salience Network
# ══════════════════════════════════════════════════════════════════════════════

class SalienceNetwork:
    """
    Priority arbiter for agent events.

    - Scores incoming events using a weighted formula.
    - Manages an asyncio.PriorityQueue (multiple workers can pull concurrently).
    - Detects active/idle mode transitions.
    """

    # Priority tiers (higher = more important)
    TYPE_WEIGHTS: dict[str, float] = {
        "scheduled_task_due": 110,
        "health_alert": 95,
        "user_frustrated": 90,
        "user_message": 80,
        "resource_pressure": 75,
        "sub_agent_complete": 60,
        "conflict_detected": 50,
        "mode_switch": 40,
    }

    # Weights for scoring formula
    W_TYPE = 0.4
    W_URGENCY = 0.25
    W_IMPACT = 0.25
    W_DECAY = 0.1

    def __init__(self, idle_threshold_seconds: Optional[int] = None):
        threshold = idle_threshold_seconds or config.control.idle_threshold_seconds
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.mode_detector = ModeDetector(threshold)
        self._event_counter = 0  # tie-breaker for same-priority events

    # ── Public API ──

    async def ingest(self, event: BaseEvent) -> float:
        """
        Score and enqueue an event.

        Returns:
            The computed priority score.
        """
        # Track user interactions for mode detection
        if isinstance(event, (UserMessageEvent, UserFrustratedEvent)):
            new_mode = self.mode_detector.on_user_message()
            if new_mode == AgentMode.ACTIVE:
                await self._inject_mode_switch(AgentMode.ACTIVE)

        priority = self._score(event)

        # Negative priority → max-priority-first in min-heap
        self._event_counter += 1
        await self.event_queue.put((-priority, self._event_counter, event))

        return priority

    async def get_next(self) -> BaseEvent:
        """
        Block until the next highest-priority event is available.

        Safe to call from multiple workers concurrently —
        asyncio.PriorityQueue handles the synchronization.
        """
        _, _, event = await self.event_queue.get()
        return event

    def has_events(self) -> bool:
        return not self.event_queue.empty()

    def check_idle(self) -> Optional[AgentMode]:
        return self.mode_detector.check_idle()

    # ── Scoring ──

    def _score(self, event: BaseEvent) -> float:
        """
        P = w_type * T + w_urgency * U + w_impact * I + w_decay * D
        """
        T = self.TYPE_WEIGHTS.get(event.type, 10)
        U = self._urgency_score(event)
        I = self._impact_score(event)
        D = self._decay_boost(event.created_at)

        return self.W_TYPE * T + self.W_URGENCY * U + self.W_IMPACT * I + self.W_DECAY * D

    def _urgency_score(self, event: BaseEvent) -> float:
        if event.deadline is None:
            return 50

        now = datetime.now(timezone.utc)
        seconds_until = (event.deadline - now).total_seconds()

        if seconds_until <= 0:
            return 100
        if seconds_until <= 60:
            return 90
        if seconds_until <= 300:
            return 70
        if seconds_until <= 3600:
            return 40

        return 20

    def _impact_score(self, event: BaseEvent) -> float:
        if event.type in ("user_message", "user_frustrated"):
            return 100
        if event.type in ("health_alert", "resource_pressure"):
            return 80
        if event.type in ("scheduled_task_due", "sub_agent_complete"):
            return 60
        return 30

    def _decay_boost(self, created_at: datetime) -> float:
        now = datetime.now(timezone.utc)
        wait_seconds = (now - created_at).total_seconds()
        return min(100, wait_seconds / 60 * 10)

    # ── Internal ──

    async def _inject_mode_switch(self, target: AgentMode) -> None:
        """Inject a mode-switch event at high priority."""
        event = ModeSwitchEvent(target_mode=target)
        self._event_counter += 1
        await self.event_queue.put((-200, self._event_counter, event))


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_salience_instance: Optional[SalienceNetwork] = None


def get_salience_network() -> SalienceNetwork:
    global _salience_instance
    if _salience_instance is None:
        _salience_instance = SalienceNetwork()
    return _salience_instance

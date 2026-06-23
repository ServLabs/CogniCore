"""
Control Systems

Three brain-inspired control networks that coordinate the entire agent:
- Salience Network: Attention/priority arbiter
- Central Executive Network (CEN): Global orchestrator with worker pool
- Default Mode Network (DMN): Idle-time processing
- Governor: Resource limits and safety
- Events: Typed event hierarchy for type-safe dispatch

Without these, Memory, Response, Analytics, and Scheduling would
compete for resources blindly.
"""

from control.events import (
    AgentMode,
    BaseEvent,
    UserMessageEvent,
    UserFrustratedEvent,
    ScheduledTaskEvent,
    SubAgentCompleteEvent,
    HealthAlertEvent,
    ResourcePressureEvent,
    ConflictDetectedEvent,
    ModeSwitchEvent,
)
from control.salience import (
    SalienceNetwork,
    get_salience_network,
)
from control.cen import (
    CentralExecutiveNetwork,
    get_cen,
)
from control.dmn import (
    DefaultModeNetwork,
    get_dmn,
)
from control.governor import (
    Governor,
    get_governor,
)

__all__ = [
    # Events
    "AgentMode",
    "BaseEvent",
    "UserMessageEvent",
    "UserFrustratedEvent",
    "ScheduledTaskEvent",
    "SubAgentCompleteEvent",
    "HealthAlertEvent",
    "ResourcePressureEvent",
    "ConflictDetectedEvent",
    "ModeSwitchEvent",
    # Salience
    "SalienceNetwork",
    "get_salience_network",
    # CEN
    "CentralExecutiveNetwork",
    "get_cen",
    # DMN
    "DefaultModeNetwork",
    "get_dmn",
    # Governor
    "Governor",
    "get_governor",
]

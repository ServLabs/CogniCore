"""
Control Systems

Three brain-inspired control networks that coordinate the entire agent:
- Salience Network: Attention/priority arbiter
- Central Executive Network (CEN): Global orchestrator
- Default Mode Network (DMN): Idle-time processing
- Governor: Resource limits and safety

Without these, Memory, Response, Analytics, and Scheduling would
compete for resources blindly.
"""

from control.salience import (
    SalienceNetwork,
    AgentEvent,
    ModeDetector,
    get_salience_network,
)
from control.cen import (
    CentralExecutiveNetwork,
    ResourceManager,
    get_cen,
)
from control.dmn import (
    DefaultModeNetwork,
    get_dmn,
)
from control.governor import (
    Governor,
    ResourceLimits,
    get_governor,
)

__all__ = [
    # Salience
    "SalienceNetwork",
    "AgentEvent",
    "ModeDetector",
    "get_salience_network",
    # CEN
    "CentralExecutiveNetwork",
    "ResourceManager",
    "get_cen",
    # DMN
    "DefaultModeNetwork",
    "get_dmn",
    # Governor
    "Governor",
    "ResourceLimits",
    "get_governor",
]

"""
Agent Framework

The glue that turns all the layers into a running system:
- Bootstrap: Wires all systems together
- API: WebSocket server for streaming
- Sessions: Multi-user session management
- Streaming: Real-time event streaming to clients
"""

from agent.bootstrap import bootstrap, shutdown
from agent.sessions import SessionManager, Session
from agent.streaming import EventStream, StreamEvent
from agent.api import WebSocketServer

__all__ = [
    # Bootstrap
    "bootstrap",
    "shutdown",
    # Sessions
    "SessionManager",
    "Session",
    # Streaming
    "EventStream",
    "StreamEvent",
    # API
    "WebSocketServer",
]

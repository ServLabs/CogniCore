"""
Agent Framework

The glue that turns all the layers into a running system:
- Bootstrap: Wires all systems together
- API: WebSocket server for streaming
- Sessions: Multi-user session management
- Streaming: Real-time event streaming to clients
- Ingestion: REST API for data ingestion
- Errors: Retry policies and fallback chains
"""

from agent.bootstrap import bootstrap, shutdown
from agent.sessions import SessionManager, Session
from agent.streaming import EventStream, StreamEvent
from agent.api import WebSocketServer
from agent.ingestion import IngestionAPI, IngestionJob, get_ingestion_api
from agent.errors import (
    RetryPolicy,
    RETRY_POLICIES,
    FALLBACK_CHAINS,
    AgentError,
    ErrorCategory,
    FallbackExecutor,
    with_retry,
)

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
    # Ingestion
    "IngestionAPI",
    "IngestionJob",
    "get_ingestion_api",
    # Errors
    "RetryPolicy",
    "RETRY_POLICIES",
    "FALLBACK_CHAINS",
    "AgentError",
    "ErrorCategory",
    "FallbackExecutor",
    "with_retry",
]

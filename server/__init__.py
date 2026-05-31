"""
Server Module

FastAPI WebSocket server for real-time chat/agent conversations.
Handles user messages, streams pipeline events, manages sessions.

This is the primary interface for end-user interactions.
"""

from server.app import create_app, get_app
from server.websocket import ConnectionManager
from server.streaming import EventStream, StreamEvent

__all__ = [
    "create_app",
    "get_app",
    "ConnectionManager",
    "EventStream",
    "StreamEvent",
]

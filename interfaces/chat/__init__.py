"""
Chat Interface

WebSocket-based real-time chat conversations.
"""

from interfaces.chat.app import get_chat_app, create_chat_app
from interfaces.chat.websocket import ConnectionManager, Session
from interfaces.chat.streaming import EventStream, StreamEvent

__all__ = [
    "get_chat_app",
    "create_chat_app",
    "ConnectionManager",
    "Session",
    "EventStream",
    "StreamEvent",
]

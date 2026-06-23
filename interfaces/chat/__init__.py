"""
Chat Interface

WebSocket-based real-time chat conversations.

Downstream dependencies (what chat calls):
- core.log, core.audit       — logging and audit trail
- response.get_pipeline      — enters the reasoning/response engine
- memory.types.wm            — persists user and assistant messages
"""

from interfaces.chat.app import get_chat_app

__all__ = [
    "get_chat_app",
]

"""
Interfaces

All external-facing APIs and services:
- admin/    - System administration (health, config, maintenance)
- service/  - Developer APIs (ingestion, metrics, evals)
- chat/     - WebSocket chat conversations
- scheduled/- Background tasks (manually triggerable)
"""

from interfaces.admin import admin_router
from interfaces.service import service_router
from interfaces.chat import get_chat_app
from interfaces.scheduled import scheduled_router

__all__ = [
    "admin_router",
    "service_router",
    "get_chat_app",
    "scheduled_router",
]

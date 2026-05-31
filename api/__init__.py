"""
API Module

FastAPI REST API for administrative and data operations:
- Ingestion: POST text, files, facts → LFM + SFM
- Metrics: GET analytics, performance data
- Evals: GET/POST evaluation results
- Admin: System health, config, management

This is separate from the server/ module which handles
real-time WebSocket chat conversations.
"""

from api.app import create_app, get_app

__all__ = [
    "create_app",
    "get_app",
]

"""
Scheduled Interface

Background/scheduled tasks exposed as REST API.
All MML background tasks can be triggered manually here.

Tasks:
- POST /scheduled/consolidation  - Memory consolidation
- POST /scheduled/cleanup        - Stale data cleanup
- POST /scheduled/reindex        - Rebuild indexes
- POST /scheduled/learning/*     - Learning algorithms
- POST /scheduled/maintenance/*  - Maintenance daemons
- GET  /scheduled/status         - Task status
"""

from interfaces.scheduled.routes import router as scheduled_router
from interfaces.scheduled.tasks import trigger_task, get_task_status

__all__ = [
    "scheduled_router",
    "trigger_task",
    "get_task_status",
]

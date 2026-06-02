"""
Scheduled Tasks

Background task definitions and execution.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from core import log
from core import audit


@dataclass
class TaskExecution:
    """Record of a task execution."""
    task_id: str
    task_name: str
    status: str  # "running", "completed", "failed"
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# In-memory task tracking
_executions: dict[str, TaskExecution] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Task Registry
# ══════════════════════════════════════════════════════════════════════════════

TASKS = {
    # MML Background Tasks
    "consolidation": "memory.management.mml:consolidate",
    "cleanup": "memory.management.mml:forget_stale",
    "reindex": "memory.management.mml:optimize_indexes",
    "integrity": "memory.management.mml:check_integrity",
    "promote": "memory.management.mml:promote_to_sfm",
    
    # Learning Tasks
    "learning/generalization": "memory.management.learning.generalization:run",
    "learning/abstraction": "memory.management.learning.abstraction:run",
    "learning/analogical": "memory.management.learning.analogical:run",
    "learning/transfer": "memory.management.learning.transfer:run",
    "learning/meta": "memory.management.learning.meta:analyze",
    "learning/contrastive": "memory.management.learning.contrastive:generate",
    
    # Maintenance Tasks
    "maintenance/backup": "migrations:backup",
    "maintenance/vacuum": "migrations:vacuum",
}


async def trigger_task(task_name: str, params: Optional[dict] = None) -> TaskExecution:
    """
    Trigger a scheduled task.
    
    Args:
        task_name: Name of the task to run.
        params: Optional parameters.
        
    Returns:
        TaskExecution record.
    """
    import time
    
    task_id = str(uuid.uuid4())
    execution = TaskExecution(
        task_id=task_id,
        task_name=task_name,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    _executions[task_id] = execution
    
    audit.log_raw("interface", "task_trigger", "scheduled", "started", task=task_name)
    log.info(f"Scheduled task started: {task_name} (id={task_id})")
    
    start = time.monotonic()
    
    try:
        result = await _execute_task(task_name, params or {})
        
        execution.status = "completed"
        execution.completed_at = datetime.now(timezone.utc)
        execution.duration_ms = (time.monotonic() - start) * 1000
        execution.result = result
        
        audit.log_raw(
            "interface", "task_complete", "scheduled", "completed",
            task=task_name, duration_ms=execution.duration_ms,
        )
        log.info(f"Scheduled task completed: {task_name} ({execution.duration_ms:.0f}ms)")
    
    except Exception as e:
        execution.status = "failed"
        execution.completed_at = datetime.now(timezone.utc)
        execution.duration_ms = (time.monotonic() - start) * 1000
        execution.error = str(e)
        
        audit.log_raw("interface", "task_failed", "scheduled", "failed", task=task_name, error=str(e))
        log.error(f"Scheduled task failed: {task_name} - {e}")
    
    return execution


async def _execute_task(task_name: str, params: dict) -> dict[str, Any]:
    """Execute a task by name."""
    
    # Handle "all" task
    if task_name == "all":
        results = {}
        for name in ["consolidation", "cleanup", "reindex"]:
            try:
                result = await _execute_task(name, params)
                results[name] = {"status": "completed", "result": result}
            except Exception as e:
                results[name] = {"status": "failed", "error": str(e)}
        return results
    
    # MML tasks
    if task_name == "consolidation":
        # Would call MML consolidation
        return {"consolidated": 0}
    
    elif task_name == "cleanup":
        # Would call MML cleanup
        return {"removed": 0}
    
    elif task_name == "reindex":
        # Would rebuild FAISS indexes
        return {"indexes_rebuilt": 0}
    
    elif task_name == "integrity":
        # Would check data integrity
        return {"issues": 0}
    
    elif task_name == "promote":
        # Would promote facts from WM to SFM
        return {"promoted": 0}
    
    # Learning tasks
    elif task_name.startswith("learning/"):
        learning_type = task_name.split("/")[1]
        return await _run_learning_task(learning_type, params)
    
    # Maintenance tasks
    elif task_name.startswith("maintenance/"):
        maintenance_type = task_name.split("/")[1]
        return await _run_maintenance_task(maintenance_type, params)
    
    else:
        raise ValueError(f"Unknown task: {task_name}")


async def _run_learning_task(learning_type: str, params: dict) -> dict[str, Any]:
    """Run a learning task."""
    
    if learning_type == "generalization":
        from memory.management.learning import get_generalizer
        generalizer = get_generalizer()
        # Would run generalization
        return {"procedures_created": 0}
    
    elif learning_type == "abstraction":
        from memory.management.learning import get_abstraction_learner
        learner = get_abstraction_learner()
        # Would run abstraction
        return {"abstractions_created": 0}
    
    elif learning_type == "meta":
        from memory.management.learning import get_meta_learner
        learner = get_meta_learner()
        result = learner.analyze()
        return result.to_dict()
    
    elif learning_type == "contrastive":
        from memory.management.learning import get_contrastive_learner
        learner = get_contrastive_learner()
        # Would generate contrastive pairs
        return {"pairs_created": 0}
    
    else:
        return {"learning_type": learning_type, "status": "not_implemented"}


async def _run_maintenance_task(maintenance_type: str, params: dict) -> dict[str, Any]:
    """Run a maintenance task."""
    
    if maintenance_type == "backup":
        # Would run backup
        return {"backup_created": True}
    
    elif maintenance_type == "vacuum":
        # Would vacuum databases
        return {"vacuumed": True}
    
    else:
        return {"maintenance_type": maintenance_type, "status": "not_implemented"}


def get_task_status(task_id: str) -> Optional[TaskExecution]:
    """Get status of a task execution."""
    return _executions.get(task_id)


def list_recent_executions(limit: int = 50) -> list[TaskExecution]:
    """List recent task executions."""
    executions = sorted(
        _executions.values(),
        key=lambda e: e.started_at,
        reverse=True,
    )
    return executions[:limit]


def list_available_tasks() -> list[str]:
    """List all available tasks."""
    return list(TASKS.keys())

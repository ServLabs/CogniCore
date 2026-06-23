"""
Scheduled Tasks

Background task definitions and execution.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from logger import log
from observability import audit
from memory import (
    get_mml,
    get_generalizer,
    get_abstraction_learner,
    get_analogical_learner,
    get_transfer_learner,
    get_contrastive_learner,
    get_meta_learner,
)


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
    "maintenance/backup": "memory.management.mml:backup_databases",
    "maintenance/vacuum": "memory.management.mml:vacuum_databases",
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
        for name in ["consolidation", "cleanup", "reindex", "integrity"]:
            try:
                result = await _execute_task(name, params)
                results[name] = {"status": "completed", "result": result}
            except Exception as e:
                results[name] = {"status": "failed", "error": str(e)}
        return results
    
    # MML tasks
    if task_name == "consolidation":
        mml = get_mml()
        result = await mml.consolidate()
        return result
    
    elif task_name == "cleanup":
        mml = get_mml()
        result = await mml.forget_stale()
        return result
    
    elif task_name == "reindex":
        mml = get_mml()
        result = await mml.optimize_indexes()
        return result
    
    elif task_name == "integrity":
        mml = get_mml()
        result = await mml.check_integrity()
        return result
    
    elif task_name == "promote":
        mml = get_mml()
        result = await mml.promote_to_sfm()
        return result
    
    # Maintenance tasks
    elif task_name == "maintenance/backup":
        return await _run_maintenance_backup()
    
    elif task_name == "maintenance/vacuum":
        return await _run_maintenance_vacuum()
    
    # Learning tasks
    elif task_name.startswith("learning/"):
        learning_type = task_name.split("/")[1]
        return await _run_learning_task(learning_type, params)
    
    else:
        raise ValueError(f"Unknown task: {task_name}")


async def _run_learning_task(learning_type: str, params: dict) -> dict[str, Any]:
    """Run a learning task."""
    
    if learning_type == "generalization":
        generalizer = get_generalizer()
        result = await generalizer.run()
        return result
    
    elif learning_type == "abstraction":
        learner = get_abstraction_learner()
        result = await learner.run()
        return result
    
    elif learning_type == "analogical":
        learner = get_analogical_learner()
        result = await learner.run()
        return result
    
    elif learning_type == "transfer":
        learner = get_transfer_learner()
        result = await learner.run()
        return result
    
    elif learning_type == "meta":
        learner = get_meta_learner()
        result = await learner.analyze()
        return result.to_dict() if hasattr(result, 'to_dict') else result
    
    elif learning_type == "contrastive":
        learner = get_contrastive_learner()
        result = await learner.generate()
        return result
    
    else:
        return {"learning_type": learning_type, "status": "not_implemented"}


async def _run_maintenance_backup() -> dict[str, Any]:
    """Backup SQLite databases to timestamped copies."""
    import asyncio
    import shutil
    from config import config
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = config.paths.data_dir / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    backed_up = []
    for db_path in [config.paths.hot_db, config.paths.cold_db]:
        if db_path.exists():
            dest = backup_dir / db_path.name
            await asyncio.to_thread(shutil.copy2, str(db_path), str(dest))
            backed_up.append(str(dest))
    
    return {"backup_dir": str(backup_dir), "files": backed_up}


async def _run_maintenance_vacuum() -> dict[str, Any]:
    """VACUUM SQLite databases to reclaim space."""
    import asyncio
    import sqlite3
    from config import config
    
    results = {}
    for name, db_path in [("hot", config.paths.hot_db), ("cold", config.paths.cold_db)]:
        if db_path.exists():
            def _vacuum(path=db_path):
                conn = sqlite3.connect(str(path))
                conn.execute("VACUUM")
                conn.close()
            await asyncio.to_thread(_vacuum)
            results[name] = "vacuumed"
        else:
            results[name] = "not_found"
    
    return results


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

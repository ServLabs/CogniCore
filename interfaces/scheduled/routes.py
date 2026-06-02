"""
Scheduled Routes

REST endpoints for triggering and monitoring background tasks.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core import log
from audit import audit
from interfaces.scheduled.tasks import (
    trigger_task,
    get_task_status,
    list_recent_executions,
    list_available_tasks,
    TASKS,
)


router = APIRouter(prefix="/scheduled", tags=["scheduled"])


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class TriggerRequest(BaseModel):
    """Request to trigger a task."""
    params: dict[str, Any] = {}


class TaskResponse(BaseModel):
    """Response from task trigger."""
    task_id: str
    task_name: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    result: dict[str, Any] = {}
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/tasks")
async def list_tasks():
    """List all available scheduled tasks."""
    return {
        "tasks": list_available_tasks(),
        "categories": {
            "mml": ["consolidation", "cleanup", "reindex", "integrity", "promote"],
            "learning": [t for t in TASKS if t.startswith("learning/")],
            "maintenance": [t for t in TASKS if t.startswith("maintenance/")],
        },
    }


@router.get("/status")
async def get_status():
    """Get status of recent task executions."""
    executions = list_recent_executions(limit=20)
    
    return {
        "recent_executions": [
            TaskResponse(
                task_id=e.task_id,
                task_name=e.task_name,
                status=e.status,
                started_at=e.started_at.isoformat(),
                completed_at=e.completed_at.isoformat() if e.completed_at else None,
                duration_ms=e.duration_ms,
                result=e.result,
                error=e.error,
            )
            for e in executions
        ],
        "running": len([e for e in executions if e.status == "running"]),
    }


@router.get("/status/{task_id}", response_model=TaskResponse)
async def get_task_execution(task_id: str):
    """Get status of a specific task execution."""
    execution = get_task_status(task_id)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Task execution not found")
    
    return TaskResponse(
        task_id=execution.task_id,
        task_name=execution.task_name,
        status=execution.status,
        started_at=execution.started_at.isoformat(),
        completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
        duration_ms=execution.duration_ms,
        result=execution.result,
        error=execution.error,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MML Tasks
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/consolidation", response_model=TaskResponse)
async def run_consolidation(request: TriggerRequest = TriggerRequest()):
    """Trigger memory consolidation."""
    execution = await trigger_task("consolidation", request.params)
    return _to_response(execution)


@router.post("/cleanup", response_model=TaskResponse)
async def run_cleanup(request: TriggerRequest = TriggerRequest()):
    """Trigger stale data cleanup."""
    execution = await trigger_task("cleanup", request.params)
    return _to_response(execution)


@router.post("/reindex", response_model=TaskResponse)
async def run_reindex(request: TriggerRequest = TriggerRequest()):
    """Trigger index rebuild."""
    execution = await trigger_task("reindex", request.params)
    return _to_response(execution)


@router.post("/integrity", response_model=TaskResponse)
async def run_integrity(request: TriggerRequest = TriggerRequest()):
    """Trigger integrity check."""
    execution = await trigger_task("integrity", request.params)
    return _to_response(execution)


@router.post("/promote", response_model=TaskResponse)
async def run_promote(request: TriggerRequest = TriggerRequest()):
    """Trigger fact promotion (WM → SFM)."""
    execution = await trigger_task("promote", request.params)
    return _to_response(execution)


@router.post("/all", response_model=TaskResponse)
async def run_all(request: TriggerRequest = TriggerRequest()):
    """Run all MML maintenance tasks."""
    execution = await trigger_task("all", request.params)
    return _to_response(execution)


# ══════════════════════════════════════════════════════════════════════════════
# Learning Tasks
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/learning/generalization", response_model=TaskResponse)
async def run_generalization(request: TriggerRequest = TriggerRequest()):
    """Trigger experience generalization."""
    execution = await trigger_task("learning/generalization", request.params)
    return _to_response(execution)


@router.post("/learning/abstraction", response_model=TaskResponse)
async def run_abstraction(request: TriggerRequest = TriggerRequest()):
    """Trigger abstraction learning."""
    execution = await trigger_task("learning/abstraction", request.params)
    return _to_response(execution)


@router.post("/learning/analogical", response_model=TaskResponse)
async def run_analogical(request: TriggerRequest = TriggerRequest()):
    """Trigger analogical learning."""
    execution = await trigger_task("learning/analogical", request.params)
    return _to_response(execution)


@router.post("/learning/transfer", response_model=TaskResponse)
async def run_transfer(request: TriggerRequest = TriggerRequest()):
    """Trigger transfer learning."""
    execution = await trigger_task("learning/transfer", request.params)
    return _to_response(execution)


@router.post("/learning/meta", response_model=TaskResponse)
async def run_meta_learning(request: TriggerRequest = TriggerRequest()):
    """Trigger meta-learning analysis."""
    execution = await trigger_task("learning/meta", request.params)
    return _to_response(execution)


@router.post("/learning/contrastive", response_model=TaskResponse)
async def run_contrastive(request: TriggerRequest = TriggerRequest()):
    """Trigger contrastive pair generation."""
    execution = await trigger_task("learning/contrastive", request.params)
    return _to_response(execution)


# ══════════════════════════════════════════════════════════════════════════════
# Maintenance Tasks
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/maintenance/backup", response_model=TaskResponse)
async def run_backup(request: TriggerRequest = TriggerRequest()):
    """Trigger database backup."""
    execution = await trigger_task("maintenance/backup", request.params)
    return _to_response(execution)


@router.post("/maintenance/vacuum", response_model=TaskResponse)
async def run_vacuum(request: TriggerRequest = TriggerRequest()):
    """Trigger database vacuum."""
    execution = await trigger_task("maintenance/vacuum", request.params)
    return _to_response(execution)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_response(execution) -> TaskResponse:
    """Convert TaskExecution to TaskResponse."""
    return TaskResponse(
        task_id=execution.task_id,
        task_name=execution.task_name,
        status=execution.status,
        started_at=execution.started_at.isoformat(),
        completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
        duration_ms=execution.duration_ms,
        result=execution.result,
        error=execution.error,
    )

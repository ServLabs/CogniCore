"""
Admin Routes

REST endpoints for system administration.
"""

import json
import os
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import config
from logger import log
from observability import audit, get_analytics
from connectors import get_registry
from interfaces.scheduled import trigger_task
from prompts import prompts

_boot_time = _time.monotonic()


router = APIRouter(prefix="/admin", tags=["admin"])


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """System health response."""
    status: str
    timestamp: str
    components: dict[str, dict[str, Any]] = {}


class SystemStatus(BaseModel):
    """System status response."""
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    active_connections: int
    pending_tasks: int
    last_maintenance: Optional[str] = None


class MaintenanceRequest(BaseModel):
    """Request to trigger maintenance."""
    task: str  # "consolidation", "cleanup", "reindex", "all"
    params: dict[str, Any] = {}


class MaintenanceResponse(BaseModel):
    """Response from maintenance task."""
    task: str
    status: str
    duration_ms: float = 0.0
    details: dict[str, Any] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive system health check.
    
    Checks all components and returns their status.
    """
    components = {}
    
    # Check connectors
    try:
        registry = get_registry()
        health = await registry.health_check_all()
        components["connectors"] = {
            "status": "healthy" if all(health.values()) else "degraded",
            "details": health,
        }
    except Exception as e:
        components["connectors"] = {"status": "error", "error": str(e)}
    
    # Check memory
    try:
        components["memory"] = {"status": "healthy"}
    except Exception as e:
        components["memory"] = {"status": "error", "error": str(e)}
    
    # Check audit
    try:
        components["audit"] = {"status": "healthy"}
    except Exception as e:
        components["audit"] = {"status": "error", "error": str(e)}
    
    # Overall status
    statuses = [c.get("status") for c in components.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "error" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"
    
    return HealthResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )


@router.get("/config")
async def get_config():
    """
    Get current configuration.
    
    Returns non-sensitive configuration values.
    """
    return {
        "debug": config.debug,
        "domain": config.domain_config.default_domain,
        "domains": config.domains,
        "paths": {
            "data_dir": str(config.paths.data_dir),
            "logs_dir": str(config.paths.logs_dir),
        },
        "api": {
            "ws_host": config.api.ws_host,
            "ws_port": config.api.ws_port,
            "rest_host": config.api.rest_host,
            "rest_port": config.api.rest_port,
        },
        "logging": {
            "level": config.logging.level,
            "log_to_console": config.logging.log_to_console,
        },
    }


@router.get("/status", response_model=SystemStatus)
async def get_status():
    """
    Get system status.
    
    Returns uptime, resource usage, and task counts.
    """
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports ru_maxrss in bytes, Linux in KB
    import sys
    memory_mb = usage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else usage.ru_maxrss / 1024

    # Get live connection count from chat app
    try:
        from interfaces.chat.app import get_chat_app
        chat_app = get_chat_app()
        manager = chat_app.state.manager if hasattr(chat_app.state, "manager") else None
        active = len(manager._sessions) if manager else 0
    except Exception:
        active = 0

    return SystemStatus(
        uptime_seconds=_time.monotonic() - _boot_time,
        memory_usage_mb=memory_mb,
        cpu_percent=0.0,
        active_connections=active,
        pending_tasks=0,
        last_maintenance=None,
    )


@router.post("/maintenance", response_model=MaintenanceResponse)
async def trigger_maintenance(request: MaintenanceRequest):
    """
    Trigger a maintenance task.
    
    Available tasks: consolidation, cleanup, reindex, all
    """
    valid_tasks = {"consolidation", "cleanup", "reindex", "all"}
    
    if request.task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {request.task}. Valid: {valid_tasks}",
        )
    
    audit.log_raw("interface", "maintenance", "admin", "started", target=request.task)
    
    start = _time.monotonic()
    
    try:
        await trigger_task(request.task, request.params)
        
        duration_ms = (_time.monotonic() - start) * 1000
        
        audit.log_raw(
            "interface", "maintenance", "admin", "completed",
            target=request.task, duration_ms=duration_ms,
        )
        
        return MaintenanceResponse(
            task=request.task,
            status="completed",
            duration_ms=duration_ms,
        )
    
    except Exception as e:
        duration_ms = (_time.monotonic() - start) * 1000
        audit.log_raw("interface", "maintenance", "admin", "failed", error=str(e))
        
        return MaintenanceResponse(
            task=request.task,
            status="failed",
            duration_ms=duration_ms,
            details={"error": str(e)},
        )


@router.post("/cache/clear")
async def clear_cache(cache_type: str = "all"):
    """
    Clear caches.
    
    Cache types: prompts, memory, all
    """
    valid_types = {"prompts", "memory", "all"}
    
    if cache_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cache type: {cache_type}. Valid: {valid_types}",
        )
    
    audit.log_raw("interface", "cache_clear", "admin", "started", target=cache_type)
    
    cleared = []
    
    if cache_type in {"prompts", "all"}:
        try:
            prompts.reload()
            cleared.append("prompts")
        except Exception:
            pass
    
    if cache_type in {"memory", "all"}:
        analytics = get_analytics()
        analytics.clear_cache()
        cleared.append("memory")
    
    audit.log_raw("interface", "cache_clear", "admin", "completed", details={"cleared": cleared})
    
    return {"status": "completed", "cleared": cleared}


@router.get("/audit/recent")
async def get_recent_audit(
    limit: int = 100,
    component: Optional[str] = None,
):
    """Get recent audit events from today's JSONL log."""
    audit_dir = config.paths.audit_dir
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = audit_dir / f"{today}.jsonl"

    if not log_file.exists():
        return {"events": [], "limit": limit, "component": component}

    events: list[dict[str, Any]] = []
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if component and event.get("component") != component:
                        continue
                    events.append(event)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log.warning("Failed to read audit log: %s", e)

    # Return most recent N events
    return {
        "events": events[-limit:],
        "limit": limit,
        "component": component,
    }

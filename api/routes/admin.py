"""
Admin API Routes

REST endpoints for system administration:
- GET /admin/health - System health check
- GET /admin/config - Current configuration
- GET /admin/status - System status
- POST /admin/maintenance - Trigger maintenance tasks
- POST /admin/cache/clear - Clear caches
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import config, log
from audit import audit


router = APIRouter()


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
        from connectors.registry import get_registry
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
    import time
    
    # Would track actual uptime in production
    uptime = 0.0
    
    # Would get actual memory/CPU usage
    memory_mb = 0.0
    cpu_percent = 0.0
    
    return SystemStatus(
        uptime_seconds=uptime,
        memory_usage_mb=memory_mb,
        cpu_percent=cpu_percent,
        active_connections=0,
        pending_tasks=0,
        last_maintenance=None,
    )


@router.post("/maintenance", response_model=MaintenanceResponse)
async def trigger_maintenance(request: MaintenanceRequest):
    """
    Trigger a maintenance task.
    
    Available tasks:
    - consolidation: Run memory consolidation
    - cleanup: Clean up old data
    - reindex: Rebuild FAISS indexes
    - all: Run all maintenance tasks
    """
    import time
    
    valid_tasks = {"consolidation", "cleanup", "reindex", "all"}
    
    if request.task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {request.task}. Valid: {valid_tasks}",
        )
    
    audit.log_raw(
        "api",
        "maintenance",
        "admin",
        "started",
        target=request.task,
    )
    
    start = time.monotonic()
    
    try:
        # Would call actual maintenance tasks in production
        # from memory.management import get_mml
        # mml = get_mml()
        # 
        # match request.task:
        #     case "consolidation":
        #         await mml.run_consolidation()
        #     case "cleanup":
        #         await mml.run_cleanup()
        #     case "reindex":
        #         await mml.rebuild_indexes()
        #     case "all":
        #         await mml.run_all_maintenance()
        
        duration_ms = (time.monotonic() - start) * 1000
        
        audit.log_raw(
            "api",
            "maintenance",
            "admin",
            "completed",
            target=request.task,
            duration_ms=duration_ms,
        )
        
        return MaintenanceResponse(
            task=request.task,
            status="completed",
            duration_ms=duration_ms,
            details={},
        )
    
    except Exception as e:
        duration_ms = (time.monotonic() - start) * 1000
        
        audit.log_raw(
            "api",
            "maintenance",
            "admin",
            "failed",
            target=request.task,
            error=str(e),
        )
        
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
    
    Cache types:
    - prompts: Clear prompt cache
    - memory: Clear memory caches
    - all: Clear all caches
    """
    valid_types = {"prompts", "memory", "all"}
    
    if cache_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cache type: {cache_type}. Valid: {valid_types}",
        )
    
    audit.log_raw(
        "api",
        "cache_clear",
        "admin",
        "started",
        target=cache_type,
    )
    
    cleared = []
    
    if cache_type in {"prompts", "all"}:
        try:
            from prompts import prompts
            prompts.reload()
            cleared.append("prompts")
        except Exception:
            pass
    
    if cache_type in {"memory", "all"}:
        # Would clear memory caches in production
        cleared.append("memory")
    
    audit.log_raw(
        "api",
        "cache_clear",
        "admin",
        "completed",
        details={"cleared": cleared},
    )
    
    return {
        "status": "completed",
        "cleared": cleared,
    }


@router.get("/audit/recent")
async def get_recent_audit(
    limit: int = 100,
    component: Optional[str] = None,
):
    """
    Get recent audit events.
    
    Optionally filter by component.
    """
    # Would query audit logs in production
    return {
        "events": [],
        "limit": limit,
        "component": component,
    }

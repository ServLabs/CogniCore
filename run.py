#!/usr/bin/env python3
"""
CogniCore - Run Server

Wires all systems together and starts the servers.
Called once, runs forever (until shutdown signal).

Interfaces:
- Chat (WebSocket): ws://localhost:8765/ws
- REST API:         http://localhost:8080
  - /admin/*        Admin endpoints
  - /ingest/*       Ingestion endpoints
  - /metrics/*      Metrics endpoints
  - /evals/*        Evaluation endpoints
  - /scheduled/*    Background task triggers

Usage:
    python run.py
    
Environment Variables:
    COGNICORE_WS_PORT     - WebSocket server port (default: 8765)
    COGNICORE_REST_PORT   - REST API port (default: 8080)
    COGNICORE_DATA_DIR    - Data directory (default: /datadrive)
    COGNICORE_DEBUG       - Enable debug mode (default: false)
"""

import asyncio
import signal
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import config
from logger import log
from observability import audit

# Connectors
from connectors import get_registry

# Memory
from memory import get_meta_memory, get_short_form_memory, get_prospective_memory, get_mml

# Core
from observability import get_analytics
from response import get_pipeline

# Control
from control import get_salience_network, get_cen, get_dmn, get_governor

# Interfaces
from interfaces.chat import get_chat_app
from interfaces.admin import admin_router
from interfaces.service import service_router
from interfaces.scheduled import scheduled_router


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap
# ══════════════════════════════════════════════════════════════════════════════

async def bootstrap() -> dict[str, Any]:
    """
    Wire all systems and return the running components.
    
    Boot sequence:
    1. Config (already loaded via singleton)
    2. Connectors (registry, connect_all, health_check)
    3. Memory (SQLite, Redis, Kuzu, FAISS)
    4. Core Systems (MML, Analytics, Response)
    5. Control (Governor, Salience, DMN, CEN)
    6. Migrations (run pending)
    7. Servers (WebSocket + REST)
    
    Returns:
        Dict with all initialized components.
    """
    audit.log_raw("agent", "boot", "main", "started")
    log.info("CogniCore bootstrap starting...")
    
    components: dict[str, Any] = {}
    
    try:
        # Phase 1: Config (already loaded)
        audit.log_raw(
            "agent",
            "config_loaded",
            "main",
            "completed",
            details={"debug": config.debug},
        )
        log.info(f"Config loaded: debug={config.debug}")
        
        # Phase 2: Verify infrastructure
        await _verify_infrastructure()
        
        # Phase 3: Connectors
        registry = get_registry()
        connect_results, health = await registry.bootstrap()
        
        audit.log_raw(
            "agent",
            "connectors_ready",
            "main",
            "completed",
            details={"connect": connect_results, "health": health},
        )
        log.info(f"Connectors ready: {len(connect_results)} connected")
        
        components["registry"] = registry
        
        # Phase 4: Memory
        # Memory types are singletons, just import to initialize
        meta = get_meta_memory()
        sfm = get_short_form_memory()
        pm = get_prospective_memory()
        
        audit.log_raw("agent", "memory_ready", "main", "completed")
        log.info("Memory systems initialized")
        
        components["meta"] = meta
        components["sfm"] = sfm
        components["pm"] = pm
        
        # Phase 5: Core Systems
        mml = get_mml()
        analytics = get_analytics()
        await analytics.start()  # Start background flush loop
        pipeline = get_pipeline()
        
        audit.log_raw("agent", "core_ready", "main", "completed")
        log.info("Core systems initialized")
        
        components["mml"] = mml
        components["analytics"] = analytics
        components["pipeline"] = pipeline
        
        # Phase 6: Control
        governor = get_governor()
        salience = get_salience_network()
        dmn = get_dmn()
        cen = get_cen()
        
        audit.log_raw("agent", "control_ready", "main", "completed")
        log.info("Control systems initialized")
        
        components["governor"] = governor
        components["salience"] = salience
        components["dmn"] = dmn
        components["cen"] = cen
        
        # Phase 7: Start CEN
        await cen.start()
        
        audit.log_raw("agent", "boot", "main", "completed")
        log.info("CogniCore bootstrap complete")
        
        return components
    
    except Exception as e:
        audit.log_raw(
            "agent",
            "boot",
            "main",
            "failed",
            error=str(e),
        )
        log.error(f"Bootstrap failed: {e}", exc_info=True)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Infrastructure Verification
# ══════════════════════════════════════════════════════════════════════════════

async def _verify_infrastructure() -> None:
    """
    Verify all required infrastructure is reachable before proceeding.
    
    Checks:
    - SQLite directories exist (create if missing)
    - Redis is reachable (warn + fallback if not)
    - Kuzu graph DB is accessible
    - FAISS index directory exists
    
    Raises:
        RuntimeError: If critical infrastructure is unreachable.
    """
    import redis.asyncio as aioredis
    
    issues: list[str] = []
    
    # SQLite: Ensure data directories exist
    for dir_path in (config.paths.sqlite_dir, config.paths.faiss_dir, config.paths.logs_dir, config.paths.audit_dir):
        dir_path.mkdir(parents=True, exist_ok=True)
    log.info("Infrastructure: data directories verified")
    
    # Redis: Check connectivity
    try:
        redis = aioredis.from_url(
            config.redis.url(config.redis.db_working_memory),
        )
        await redis.ping()
        await redis.aclose()
        log.info("Infrastructure: Redis reachable at %s:%s", config.redis.host, config.redis.port)
    except Exception as e:
        if config.redis.fallback_enabled:
            log.warning("Infrastructure: Redis unavailable (%s) — using local fallback", e)
        else:
            issues.append(f"Redis unreachable: {e}")
    
    # Kuzu: Check graph DB directory
    kuzu_dir = config.paths.kuzu_dir
    kuzu_dir.mkdir(parents=True, exist_ok=True)
    log.info("Infrastructure: Kuzu directory verified at %s", kuzu_dir)
    
    # FAISS: Verify index directory
    faiss_dir = config.paths.faiss_dir
    faiss_dir.mkdir(parents=True, exist_ok=True)
    log.info("Infrastructure: FAISS directory verified at %s", faiss_dir)
    
    if issues:
        raise RuntimeError(f"Infrastructure check failed: {'; '.join(issues)}")
    
    audit.log_raw("agent", "infrastructure_verified", "main", "completed")


async def shutdown(components: dict[str, Any]) -> None:
    """
    Graceful shutdown — close everything in reverse order.
    
    Args:
        components: Dict of initialized components from bootstrap.
    """
    audit.log_raw("agent", "shutdown", "main", "started")
    log.info("Shutdown starting...")
    
    try:
        # Stop CEN
        cen = components.get("cen")
        if cen:
            await cen.stop()
        
        # Flush and stop analytics
        analytics = components.get("analytics")
        if analytics:
            await analytics.stop()
        
        # Disconnect connectors
        registry = components.get("registry")
        if registry:
            await registry.disconnect_all()
        
        # Close audit
        audit.close()
        
        audit.log_raw("agent", "shutdown", "main", "completed")
        log.info("Shutdown complete")
    
    except Exception as e:
        audit.log_raw(
            "agent",
            "shutdown",
            "main",
            "failed",
            error=str(e),
        )
        log.error(f"Shutdown error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main entry point for CogniCore.
    
    Starts both interfaces:
    - Chat (WebSocket) on WS_PORT
    - REST API on REST_PORT
    
    Usage: python run.py
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    components: dict[str, Any] = {}
    
    try:
        # Bootstrap core systems
        components = loop.run_until_complete(bootstrap())
        
        # Setup signal handlers for graceful shutdown
        def signal_handler():
            asyncio.ensure_future(shutdown(components))
            loop.stop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
        
        # Start servers
        print(f"CogniCore starting...")
        print(f"  Chat:     ws://{config.api.ws_host}:{config.api.ws_port}/ws")
        print(f"  REST API: http://{config.api.rest_host}:{config.api.rest_port}")
        
        # Create apps from interfaces
        chat_app = get_chat_app()
        
        # REST API app (admin + service + scheduled)
        rest_app = FastAPI(
            title="CogniCore API",
            description="REST API for admin, service, and scheduled tasks",
            version="1.0.0",
        )
        rest_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        rest_app.include_router(admin_router)
        rest_app.include_router(service_router)
        rest_app.include_router(scheduled_router)
        
        # Create server configs
        chat_config = uvicorn.Config(
            chat_app,
            host=config.api.ws_host,
            port=config.api.ws_port,
            log_level="info",
        )
        rest_config = uvicorn.Config(
            rest_app,
            host=config.api.rest_host,
            port=config.api.rest_port,
            log_level="info",
        )
        
        chat_server = uvicorn.Server(chat_config)
        rest_server = uvicorn.Server(rest_config)
        
        # Run both servers concurrently
        async def run_servers():
            await asyncio.gather(
                chat_server.serve(),
                rest_server.serve(),
            )
        
        loop.run_until_complete(run_servers())
    
    except KeyboardInterrupt:
        loop.run_until_complete(shutdown(components))
    
    finally:
        loop.close()


if __name__ == "__main__":
    main()

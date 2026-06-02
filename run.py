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
from typing import Any, Optional

from core import config, log
from audit import audit


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
        
        # Phase 2: Connectors
        from connectors.registry import get_registry
        
        registry = get_registry()
        
        # Register default connectors
        from connectors.data import FileConnector
        from connectors.ai import LLMConnector, EmbeddingConnector, NLIConnector
        from connectors.sandbox import PythonSandbox, SQLSandbox
        
        registry.register("file", FileConnector(base_dir=config.paths.data_dir))
        registry.register("llm", LLMConnector())
        registry.register("embedder", EmbeddingConnector())
        registry.register("nli", NLIConnector())
        registry.register("python_sandbox", PythonSandbox())
        registry.register("sql_sandbox", SQLSandbox())
        
        # Connect all configured connectors
        connect_results = await registry.connect_all()
        health = await registry.health_check_all()
        
        audit.log_raw(
            "agent",
            "connectors_ready",
            "main",
            "completed",
            details={"connect": connect_results, "health": health},
        )
        log.info(f"Connectors ready: {len(connect_results)} connected")
        
        components["registry"] = registry
        
        # Phase 3: Memory
        # Memory types are singletons, just import to initialize
        from memory.types.meta import get_meta_memory
        from memory.types.sfm import get_short_form_memory
        from memory.types.pm import get_prospective_memory
        
        meta = get_meta_memory()
        sfm = get_short_form_memory()
        pm = get_prospective_memory()
        
        audit.log_raw("agent", "memory_ready", "main", "completed")
        log.info("Memory systems initialized")
        
        components["meta"] = meta
        components["sfm"] = sfm
        components["pm"] = pm
        
        # Phase 4: Core Systems
        from memory.management import get_mml
        from analytics import get_analytics
        from response import get_pipeline
        
        mml = get_mml()
        analytics = get_analytics()
        pipeline = get_pipeline()
        
        audit.log_raw("agent", "core_ready", "main", "completed")
        log.info("Core systems initialized")
        
        components["mml"] = mml
        components["analytics"] = analytics
        components["pipeline"] = pipeline
        
        # Phase 5: Control
        from control import get_salience_network, get_cen, get_dmn, get_governor
        
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
        
        # Phase 6: Run migrations
        from migrations import migrate
        
        migrate(config.paths.hot_db)
        migrate(config.paths.cold_db)
        
        audit.log_raw("agent", "migrations_complete", "main", "completed")
        log.info("Migrations complete")
        
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
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
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
        from interfaces.chat import get_chat_app
        from interfaces.admin import admin_router
        from interfaces.service import service_router
        from interfaces.scheduled import scheduled_router
        
        # Chat app (WebSocket)
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

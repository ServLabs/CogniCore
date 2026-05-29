"""
Agent Bootstrap

Wires all systems together and starts the event loop.
Called once, runs forever (until shutdown signal).
"""

import asyncio
import signal
from typing import Any, Optional

from config import config
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
    6. Sessions + API (WebSocket server)
    
    Returns:
        Dict with all initialized components.
    """
    audit.log_raw("agent", "boot", "main", "started")
    
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
        
        components["governor"] = governor
        components["salience"] = salience
        components["dmn"] = dmn
        components["cen"] = cen
        
        # Phase 6: Sessions + API
        from agent.sessions import SessionManager
        from agent.api import WebSocketServer
        
        sessions = SessionManager()
        api = WebSocketServer(sessions)
        
        await api.start(
            host=config.api.ws_host,
            port=config.api.ws_port,
        )
        
        audit.log_raw(
            "agent",
            "api_ready",
            "main",
            "completed",
            details={"port": config.api.ws_port},
        )
        
        components["sessions"] = sessions
        components["api"] = api
        
        # Start CEN
        await cen.start()
        
        audit.log_raw("agent", "boot", "main", "completed")
        
        return components
    
    except Exception as e:
        audit.log_raw(
            "agent",
            "boot",
            "main",
            "failed",
            error=str(e),
        )
        raise


async def shutdown(components: dict[str, Any]) -> None:
    """
    Graceful shutdown — close everything in reverse order.
    
    Args:
        components: Dict of initialized components from bootstrap.
    """
    audit.log_raw("agent", "shutdown", "main", "started")
    
    try:
        # Stop CEN
        cen = components.get("cen")
        if cen:
            await cen.stop()
        
        # Stop API
        api = components.get("api")
        if api:
            await api.stop()
        
        # Disconnect connectors
        registry = components.get("registry")
        if registry:
            await registry.disconnect_all()
        
        # Close audit
        audit.close()
        
        audit.log_raw("agent", "shutdown", "main", "completed")
    
    except Exception as e:
        audit.log_raw(
            "agent",
            "shutdown",
            "main",
            "failed",
            error=str(e),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main entry point for the agent.
    
    Usage: python -m agent.bootstrap
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    components: dict[str, Any] = {}
    
    try:
        # Bootstrap
        components = loop.run_until_complete(bootstrap())
        
        # Setup signal handlers for graceful shutdown
        def signal_handler():
            asyncio.ensure_future(shutdown(components))
            loop.stop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
        
        # Run forever
        print(f"CogniCore agent running on ws://{config.api.ws_host}:{config.api.ws_port}")
        loop.run_forever()
    
    except KeyboardInterrupt:
        loop.run_until_complete(shutdown(components))
    
    finally:
        loop.close()


if __name__ == "__main__":
    main()

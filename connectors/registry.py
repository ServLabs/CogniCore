"""
Connector Registry

Central registry for all connectors. Provides:
- Registration and lookup
- Lifecycle management (connect all, disconnect all)
- Health checking
- Diagnostics
"""

from typing import Any, Optional

from logger import log
from observability import audit
from connectors.base import BaseConnector, ConnectorAuthError, ConnectorInfo, ConnectorStatus


class ConnectorRegistry:
    """
    Central registry for all connectors.
    
    Provides:
    - Registration and lookup by name
    - Bulk lifecycle management
    - Health checking across all connectors
    - Diagnostics and status reporting
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._connectors: dict[str, BaseConnector] = {}
    
    def register(self, name: str, connector: BaseConnector) -> None:
        """
        Register a connector.
        
        Args:
            name: Unique name for the connector.
            connector: Connector instance.
        """
        self._connectors[name] = connector
    
    def get(self, name: str) -> Optional[BaseConnector]:
        """
        Get a connector by name.
        
        Args:
            name: Connector name.
            
        Returns:
            Connector or None if not found.
        """
        return self._connectors.get(name)
    
    def get_required(self, name: str) -> BaseConnector:
        """
        Get a connector by name, raising if not found.
        
        Args:
            name: Connector name.
            
        Returns:
            Connector.
            
        Raises:
            KeyError: If connector not found.
        """
        connector = self._connectors.get(name)
        if connector is None:
            raise KeyError(f"Connector not found: {name}")
        return connector
    
    def list_names(self) -> list[str]:
        """List all registered connector names."""
        return list(self._connectors.keys())
    
    def list_all(self) -> list[ConnectorInfo]:
        """List info for all registered connectors."""
        return [c.info() for c in self._connectors.values()]
    
    async def connect_all(self) -> dict[str, bool]:
        """
        Connect all registered connectors.
        
        Returns:
            Dict of name -> success.
        """
        results = {}
        for name, connector in self._connectors.items():
            if not connector.is_configured():
                log.info("Connector '%s' not configured — skipping", name)
                audit.log_raw("connector", "connect", name, "skipped", details={"reason": "not_configured"})
                results[name] = False
                continue
            try:
                await connector.connect()
                results[name] = True
                log.info("Connector '%s' connected", name)
                audit.log_raw("connector", "connect", name, "completed")
            except ConnectorAuthError as e:
                results[name] = False
                log.error("Connector '%s' auth failed: %s", name, e)
                audit.log_raw("connector", "connect", name, "failed", error=f"AUTH: {e}")
            except Exception as e:
                results[name] = False
                log.error("Connector '%s' connect failed: %s", name, e)
                audit.log_raw("connector", "connect", name, "failed", error=str(e))
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all registered connectors."""
        for name, connector in self._connectors.items():
            try:
                await connector.disconnect()
                log.debug("Connector '%s' disconnected", name)
            except Exception as e:
                log.warning("Connector '%s' disconnect error: %s", name, e)
    
    async def health_check_all(self) -> dict[str, bool]:
        """
        Health check all registered connectors.
        
        Returns:
            Dict of name -> healthy.
        """
        results = {}
        for name, connector in self._connectors.items():
            try:
                healthy = await connector.health_check()
                results[name] = healthy
                if not healthy:
                    log.warning("Connector '%s' health check unhealthy", name)
            except Exception as e:
                results[name] = False
                log.warning("Connector '%s' health check error: %s", name, e)
        return results
    
    def get_status(self) -> dict[str, Any]:
        """
        Get status of all connectors.
        
        Returns:
            Dict with connector statuses.
        """
        return {
            "connectors": {
                name: connector.info().to_dict()
                for name, connector in self._connectors.items()
            },
            "total": len(self._connectors),
            "connected": sum(
                1 for c in self._connectors.values()
                if c.info().status == ConnectorStatus.CONNECTED
            ),
        }
    
    async def bootstrap(self) -> tuple[dict[str, bool], dict[str, bool]]:
        """
        Register all default connectors, connect, and health-check.
        
        This is the single entry point for connector setup.
        Adding/removing connectors only requires changing this method.
        
        Returns:
            Tuple of (connect_results, health_results).
        """
        from config import config
        from connectors.data import FileConnector
        from connectors.ai import LLMConnector
        from connectors._internal import EmbeddingConnector, NLIConnector
        from connectors.sandbox import PythonSandbox, SQLSandbox
        
        # ── Core connectors (always registered) ──
        self.register("file", FileConnector(base_dir=config.paths.data_dir))
        self.register("llm", LLMConnector())
        self.register("embedder", EmbeddingConnector())
        self.register("nli", NLIConnector())
        self.register("python_sandbox", PythonSandbox())
        self.register("sql_sandbox", SQLSandbox())
        
        # ── Optional data connectors (register only if configured) ──
        from connectors.data import SnowflakeConnector, DatabricksConnector, AzureSQLConnector
        
        if config.snowflake.account:
            self.register("snowflake", SnowflakeConnector())
        if config.databricks.host:
            self.register("databricks", DatabricksConnector())
        if config.azure_sql.server or config.azure_sql.connection_string:
            self.register("azure_sql", AzureSQLConnector())
        
        # ── Connect and health-check ──
        connect_results = await self.connect_all()
        health_results = await self.health_check_all()
        
        return connect_results, health_results
    
    # ══════════════════════════════════════════════════════════════════════════
    # Tool Discovery & Execution
    # ══════════════════════════════════════════════════════════════════════════
    
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Collect tool schemas from all registered connectors that expose one.
        
        Returns:
            List of OpenAI function-calling format tool schemas.
        """
        schemas = []
        for name, connector in self._connectors.items():
            schema = connector.tool_schema()
            if schema is not None:
                schemas.append(schema)
        return schemas
    
    def get_tool_names(self) -> list[str]:
        """List names of all tool-capable connectors."""
        return [
            name for name, connector in self._connectors.items()
            if connector.tool_schema() is not None
        ]
    
    async def execute_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool by name with given parameters.
        
        Args:
            tool_name: Name of the tool (matches tool_schema()['name']).
            params: Parameters for tool execution.
            
        Returns:
            Tool result dict.
            
        Raises:
            KeyError: If tool_name not found.
            ValueError: If connector doesn't support tool execution.
        """
        # Find connector by tool schema name
        for name, connector in self._connectors.items():
            schema = connector.tool_schema()
            if schema and schema["name"] == tool_name:
                return await connector.execute_tool(params)
        
        raise KeyError(f"Tool not found: {tool_name}")


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_registry_instance: Optional[ConnectorRegistry] = None


def get_registry() -> ConnectorRegistry:
    """Get the singleton ConnectorRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ConnectorRegistry()
    return _registry_instance

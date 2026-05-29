"""
Connector Registry

Central registry for all connectors. Provides:
- Registration and lookup
- Lifecycle management (connect all, disconnect all)
- Health checking
- Diagnostics
"""

from typing import Any, Optional

from connectors.base import BaseConnector, ConnectorInfo, ConnectorStatus


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
            try:
                if connector.is_configured():
                    await connector.connect()
                    results[name] = True
                else:
                    results[name] = False
            except Exception:
                results[name] = False
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all registered connectors."""
        for connector in self._connectors.values():
            try:
                await connector.disconnect()
            except Exception:
                pass  # Best effort
    
    async def health_check_all(self) -> dict[str, bool]:
        """
        Health check all registered connectors.
        
        Returns:
            Dict of name -> healthy.
        """
        results = {}
        for name, connector in self._connectors.items():
            try:
                results[name] = await connector.health_check()
            except Exception:
                results[name] = False
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

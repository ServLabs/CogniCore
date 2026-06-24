"""
Connector Base Interface

All connectors implement this contract for consistent lifecycle management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class ConnectorType(Enum):
    """Type of connector."""
    DATA = "data"        # REST, File
    AI = "ai"            # LLM, Embedder, NLI, Reranker
    SANDBOX = "sandbox"  # Python, SQL sandboxes


class ConnectorStatus(Enum):
    """Status of a connector."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConnectorInfo:
    """Connector metadata for registry and diagnostics."""
    name: str
    connector_type: ConnectorType
    status: ConnectorStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "connector_type": self.connector_type.value,
            "status": self.status.value,
            "metadata": self.metadata,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════════════════════

class ConnectorAuthError(Exception):
    """Raised when a connector fails to authenticate or authorize."""

    def __init__(self, connector_name: str, message: str = ""):
        self.connector_name = connector_name
        super().__init__(f"Auth failed for '{connector_name}': {message}" if message else f"Auth failed for '{connector_name}'")


# ══════════════════════════════════════════════════════════════════════════════
# Base Connector
# ══════════════════════════════════════════════════════════════════════════════

class BaseConnector(ABC):
    """
    Base interface for all connectors.
    
    Every connector implements this contract for:
    - Connection lifecycle (connect, disconnect)
    - Health checking
    - Configuration validation
    - Metadata reporting
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection.
        
        Called once at startup or on first use.
        """
        ...
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Clean shutdown.
        
        Release pools, close connections.
        """
        ...
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if connector is alive and responsive.
        
        Returns:
            True if healthy.
        """
        ...
    
    @abstractmethod
    def info(self) -> ConnectorInfo:
        """
        Return connector metadata.
        
        Used by registry and diagnostics.
        
        Returns:
            ConnectorInfo with name, type, status, metadata.
        """
        ...
    
    def is_configured(self) -> bool:
        """
        Check if required configuration is present.
        
        Override per connector to check env vars, config values, etc.
        
        Returns:
            True if configured.
        """
        return True
    
    def tool_schema(self) -> Optional[dict[str, Any]]:
        """
        Return OpenAI function-calling format schema if this connector is a tool.
        
        Override to expose this connector as an agent-usable tool.
        Returns None by default (not a tool).
        
        Expected format:
            {
                "name": "tool_name",
                "description": "What this tool does",
                "parameters": {
                    "type": "object",
                    "properties": { ... },
                    "required": [...]
                }
            }
        """
        return None
    
    async def execute_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute this connector as a tool with the given parameters.
        
        Override in connectors that expose tool_schema().
        
        Args:
            params: Parameters matching the tool_schema definition.
            
        Returns:
            Dict with at minimum {"result": ...} or {"error": ...}.
        """
        return {"error": f"{self.__class__.__name__} does not support tool execution"}


# ══════════════════════════════════════════════════════════════════════════════
# Data Connector Base
# ══════════════════════════════════════════════════════════════════════════════

class BaseDataConnector(BaseConnector):
    """
    Extended interface for data source connectors.
    
    Adds query execution and schema introspection.
    """
    
    connector_type = ConnectorType.DATA
    
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a parameterized query.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            
        Returns:
            List of row dicts.
        """
        ...
    
    @abstractmethod
    async def execute_query_df(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Execute query and return a pandas DataFrame.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            
        Returns:
            pandas DataFrame.
        """
        ...
    
    @abstractmethod
    async def get_schema(
        self,
        table: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Return schema info for introspection.
        
        Args:
            table: Specific table name (None = list all tables).
            
        Returns:
            Dict with tables or columns info.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Sandbox Base
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SandboxResult:
    """Result of sandbox code execution."""
    success: bool
    output: str  # stdout / return value
    error: Optional[str] = None  # stderr / exception message
    execution_time_ms: float = 0.0
    truncated: bool = False  # True if output was cut to size limit
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "truncated": self.truncated,
        }


class BaseSandbox(BaseConnector):
    """
    Base interface for sandbox connectors.
    
    Sandboxes execute code in isolated environments with security controls.
    """
    
    connector_type = ConnectorType.SANDBOX
    
    @abstractmethod
    async def execute(
        self,
        code: str,
        context: Optional[dict[str, Any]] = None,
    ) -> SandboxResult:
        """
        Execute code in a sandboxed environment.
        
        Args:
            code: Code to execute.
            context: Variables to inject.
            
        Returns:
            SandboxResult with output or error.
        """
        ...

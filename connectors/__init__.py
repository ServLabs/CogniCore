"""
Connectors Layer

Interface to external systems:
- AI Tools: LLM (exposed as runtime tool)
- Internal: GenAI gateway, Embedder, NLI (agent infrastructure)
- Data Connectors: Snowflake, Databricks, Azure SQL, REST, File
- Sandbox Connectors: Python, SQL sandboxes

All connectors implement BaseConnector for consistent lifecycle management.
"""

from connectors.base import (
    BaseConnector,
    BaseDataConnector,
    BaseSandbox,
    ConnectorAuthError,
    ConnectorType,
    ConnectorStatus,
    ConnectorInfo,
    SandboxResult,
)
from connectors.registry import (
    ConnectorRegistry,
    get_registry,
)
from connectors.ai import LLMConnector
from connectors._internal import genai, EmbeddingConnector, NLIConnector
from connectors.data import (
    SnowflakeConnector,
    DatabricksConnector,
    AzureSQLConnector,
    RESTAPIConnector,
    FileConnector,
)
from connectors.sandbox import PythonSandbox, SQLSandbox

__all__ = [
    # Base
    "BaseConnector",
    "BaseDataConnector",
    "BaseSandbox",
    "ConnectorAuthError",
    "ConnectorType",
    "ConnectorStatus",
    "ConnectorInfo",
    "SandboxResult",
    # Registry
    "ConnectorRegistry",
    "get_registry",
    # AI
    "LLMConnector",
    "EmbeddingConnector",
    "NLIConnector",
    "genai",
    # Data
    "SnowflakeConnector",
    "DatabricksConnector",
    "AzureSQLConnector",
    "RESTAPIConnector",
    "FileConnector",
    # Sandbox
    "PythonSandbox",
    "SQLSandbox",
]

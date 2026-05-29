"""
Connectors Layer

Interface to external systems:
- Data Connectors: Snowflake, Databricks, Azure SQL, REST, File
- AI Connectors: LLM, Embedder, NLI, Reranker
- Sandbox Connectors: Python, SQL sandboxes

All connectors implement BaseConnector for consistent lifecycle management.
"""

from connectors.base import (
    BaseConnector,
    ConnectorType,
    ConnectorStatus,
    ConnectorInfo,
)
from connectors.registry import (
    ConnectorRegistry,
    get_registry,
)

__all__ = [
    # Base
    "BaseConnector",
    "ConnectorType",
    "ConnectorStatus",
    "ConnectorInfo",
    # Registry
    "ConnectorRegistry",
    "get_registry",
]

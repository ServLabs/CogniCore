"""
SQL Sandbox

Read-only SQL query execution with validation.
Blocks dangerous statements (DROP, DELETE, UPDATE, etc.).
"""

import re
import time
from typing import Any, Optional

from config import config
from connectors.base import BaseSandbox, SandboxResult, ConnectorInfo, ConnectorStatus, ConnectorType


class SQLSandbox(BaseSandbox):
    """
    SQL query sandbox.
    
    Provides:
    - Read-only query validation
    - Dangerous statement blocking
    - Timeout enforcement
    - Result truncation
    """
    
    # Dangerous SQL patterns to block
    BLOCKED_PATTERNS = [
        r"\bDROP\b",
        r"\bDELETE\b",
        r"\bTRUNCATE\b",
        r"\bUPDATE\b",
        r"\bINSERT\b",
        r"\bALTER\b",
        r"\bCREATE\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bEXEC\b",
        r"\bEXECUTE\b",
        r"(?:^|\s)--",    # SQL line comments (preceded by whitespace or start of line)
        r"/\*",           # Block comments
        r";\s*\S",        # Multiple statements (semicolon followed by non-whitespace)
    ]
    
    def __init__(self, connector=None):
        """
        Initialize SQL sandbox.
        
        Args:
            connector: Data connector to use for queries.
        """
        self._connector = connector
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BLOCKED_PATTERNS
        ]
    
    def set_connector(self, connector) -> None:
        """Set the data connector to use."""
        self._connector = connector
    
    async def connect(self) -> None:
        """Connect underlying data connector."""
        if self._connector:
            await self._connector.connect()
    
    async def disconnect(self) -> None:
        """Disconnect underlying data connector."""
        if self._connector:
            await self._connector.disconnect()
    
    async def health_check(self) -> bool:
        """Check if sandbox is working."""
        if not self._connector:
            return False
        return await self._connector.health_check()
    
    async def execute(
        self,
        code: str,
        context: Optional[dict[str, Any]] = None,
    ) -> SandboxResult:
        """
        Execute SQL query in sandbox.
        
        Args:
            code: SQL query to execute.
            context: Query parameters.
            
        Returns:
            SandboxResult with output or error.
        """
        if not self._connector:
            return SandboxResult(
                success=False,
                output="",
                error="No data connector configured",
                execution_time_ms=0,
                truncated=False,
            )
        
        # 1. Validate query
        violation = self._validate(code)
        if violation:
            return SandboxResult(
                success=False,
                output="",
                error=f"Security violation: {violation}",
                execution_time_ms=0,
                truncated=False,
            )
        
        # 2. Execute query
        start = time.monotonic()
        try:
            results = await self._connector.execute_query(code, context)
            elapsed_ms = (time.monotonic() - start) * 1000
            
            # Format results
            import json
            output = json.dumps(results, indent=2, default=str)
            
            # Truncate if needed
            truncated = False
            max_output = config.sandbox.max_output_bytes
            if len(output) > max_output:
                output = output[:max_output] + "\n... (truncated)"
                truncated = True
            
            return SandboxResult(
                success=True,
                output=output,
                error=None,
                execution_time_ms=elapsed_ms,
                truncated=truncated,
            )
        
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            return SandboxResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=elapsed_ms,
                truncated=False,
            )
    
    def _validate(self, query: str) -> Optional[str]:
        """
        Validate SQL query.
        
        Args:
            query: SQL query to validate.
            
        Returns:
            Violation message or None if valid.
        """
        # Check for blocked patterns
        for pattern in self._compiled_patterns:
            if pattern.search(query):
                return f"Blocked pattern: {pattern.pattern}"
        
        # Must start with SELECT, WITH, or EXPLAIN
        query_upper = query.strip().upper()
        if not any(query_upper.startswith(kw) for kw in ["SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE"]):
            return "Only SELECT, WITH, EXPLAIN, SHOW, DESCRIBE queries are allowed"
        
        return None
    
    def tool_schema(self) -> dict[str, Any]:
        """Expose SQL sandbox as a tool."""
        return {
            "name": "run_sql",
            "description": (
                "Execute read-only SQL queries against the connected database. "
                "Only SELECT, WITH, EXPLAIN, SHOW, DESCRIBE are allowed. "
                "Returns JSON-formatted query results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute (read-only).",
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters for parameterized queries.",
                    },
                },
                "required": ["query"],
            },
        }
    
    async def execute_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute SQL query via tool interface."""
        result = await self.execute(
            code=params["query"],
            context=params.get("params"),
        )
        if result.success:
            return {"result": result.output, "execution_time_ms": result.execution_time_ms}
        return {"error": result.error, "execution_time_ms": result.execution_time_ms}
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        return ConnectorInfo(
            name="sql_sandbox",
            connector_type=ConnectorType.SANDBOX,
            status=ConnectorStatus.CONNECTED if self._connector else ConnectorStatus.DISCONNECTED,
            metadata={
                "connector": self._connector.info().name if self._connector else None,
            },
        )

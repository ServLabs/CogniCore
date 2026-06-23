"""
Databricks Connector

Async Databricks connector. Wraps sync driver with asyncio.to_thread().
"""

import asyncio
from typing import Any, Optional

from config import config
from connectors.base import BaseDataConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class DatabricksConnector(BaseDataConnector):
    """
    Async Databricks connector.
    
    Wraps the sync databricks-sql-connector with asyncio.to_thread()
    for non-blocking operation.
    """
    
    def __init__(self):
        """Initialize connector."""
        self._conn = None
    
    def is_configured(self) -> bool:
        """Check if Databricks credentials are configured."""
        db = config.databricks
        return all([db.host, db.token, db.http_path])
    
    async def connect(self) -> None:
        """Establish Databricks connection."""
        if not self.is_configured():
            return
        
        from databricks import sql as databricks_sql
        
        db = config.databricks
        self._conn = await asyncio.to_thread(
            databricks_sql.connect,
            server_hostname=db.host,
            http_path=db.http_path,
            access_token=db.token,
            catalog=db.catalog,
            schema=db.schema,
        )
    
    async def disconnect(self) -> None:
        """Close Databricks connection."""
        if self._conn:
            await asyncio.to_thread(self._conn.close)
            self._conn = None
    
    async def health_check(self) -> bool:
        """Check if connection is alive."""
        if not self._conn:
            return False
        
        try:
            def _check():
                cur = self._conn.cursor()
                cur.execute("SELECT 1")
                return True
            return await asyncio.to_thread(_check)
        except Exception:
            return False
    
    async def execute_query(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        def _execute():
            cur = self._conn.cursor()
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
        
        return await asyncio.to_thread(_execute)
    
    async def execute_query_df(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Execute query and return pandas DataFrame."""
        import pandas as pd
        
        rows = await self.execute_query(query, params)
        return pd.DataFrame(rows)
    
    async def get_schema(
        self,
        table: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get schema information."""
        def _get():
            cur = self._conn.cursor()
            if table:
                # Sanitize: only allow alphanumeric, underscore, dot (schema.table)
                import re
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', table):
                    raise ValueError(f"Invalid table name: {table}")
                cur.execute(f"DESCRIBE TABLE {table}")
                return {
                    "table": table,
                    "columns": [{"name": r[0], "type": r[1]} for r in cur.fetchall()],
                }
            cur.execute("SHOW TABLES")
            return {"tables": [r[1] for r in cur.fetchall()]}
        
        return await asyncio.to_thread(_get)
    
    def tool_schema(self) -> dict[str, Any]:
        """Expose Databricks as a tool."""
        db = config.databricks
        return {
            "name": "query_databricks",
            "description": (
                f"Execute read-only SQL queries against Databricks "
                f"(catalog: {db.catalog}). "
                "Use for querying structured data, getting schema info, or exploring tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query", "schema"],
                        "description": "'query' to run SQL, 'schema' to get table/column info.",
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query (for action='query').",
                    },
                    "table": {
                        "type": "string",
                        "description": "Table name (for action='schema'). Omit to list all tables.",
                    },
                },
                "required": ["action"],
            },
        }
    
    async def execute_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute Databricks operation via tool interface."""
        try:
            match params["action"]:
                case "query":
                    rows = await self.execute_query(params["query"])
                    return {"result": rows}
                case "schema":
                    schema = await self.get_schema(params.get("table"))
                    return {"result": schema}
                case _:
                    return {"error": f"Unknown action: {params['action']}"}
        except Exception as e:
            return {"error": str(e)}
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        db = config.databricks
        return ConnectorInfo(
            name="databricks",
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED if self._conn else ConnectorStatus.DISCONNECTED,
            metadata={
                "host": db.host,
                "catalog": db.catalog,
            },
        )

"""
Azure SQL Connector

Async Azure SQL connector. Wraps sync pyodbc driver with asyncio.to_thread().
"""

import asyncio
from typing import Any, Optional

from core import config
from connectors.base import BaseDataConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class AzureSQLConnector(BaseDataConnector):
    """
    Async Azure SQL connector.
    
    Wraps the sync pyodbc driver with asyncio.to_thread()
    for non-blocking operation.
    """
    
    def __init__(self):
        """Initialize connector."""
        self._conn = None
    
    def is_configured(self) -> bool:
        """Check if Azure SQL credentials are configured."""
        az = config.azure_sql
        return az.connection_string is not None or (
            az.server is not None and az.database is not None
        )
    
    async def connect(self) -> None:
        """Establish Azure SQL connection."""
        if not self.is_configured():
            return
        
        import pyodbc
        
        az = config.azure_sql
        
        def _connect():
            if az.connection_string:
                return pyodbc.connect(az.connection_string)
            else:
                conn_str = (
                    f"DRIVER={{{az.driver}}};"
                    f"SERVER={az.server};"
                    f"DATABASE={az.database};"
                    "Authentication=ActiveDirectoryInteractive;"
                )
                return pyodbc.connect(conn_str)
        
        self._conn = await asyncio.to_thread(_connect)
    
    async def disconnect(self) -> None:
        """Close Azure SQL connection."""
        if self._conn:
            await asyncio.to_thread(self._conn.close)
            self._conn = None
    
    async def health_check(self) -> bool:
        """Check if connection is alive."""
        if not self._conn:
            return False
        
        try:
            def _check():
                self._conn.cursor().execute("SELECT 1").fetchone()
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
            if params:
                cur.execute(query, list(params.values()))
            else:
                cur.execute(query)
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
                cur.execute(
                    "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?",
                    [table]
                )
                return {
                    "table": table,
                    "columns": [{"name": r[0], "type": r[1]} for r in cur.fetchall()],
                }
            cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
            return {"tables": [r[0] for r in cur.fetchall()]}
        
        return await asyncio.to_thread(_get)
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        az = config.azure_sql
        return ConnectorInfo(
            name="azure_sql",
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED if self._conn else ConnectorStatus.DISCONNECTED,
            metadata={
                "server": az.server,
                "database": az.database,
            },
        )

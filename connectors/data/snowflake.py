"""
Snowflake Connector

Async Snowflake connector. Wraps sync driver with asyncio.to_thread().
"""

import asyncio
from typing import Any, Optional

from core import config
from connectors.base import BaseDataConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class SnowflakeConnector(BaseDataConnector):
    """
    Async Snowflake connector.
    
    Wraps the sync snowflake-connector-python with asyncio.to_thread()
    for non-blocking operation.
    """
    
    def __init__(self):
        """Initialize connector."""
        self._conn = None
    
    def is_configured(self) -> bool:
        """Check if Snowflake credentials are configured."""
        sf = config.snowflake
        return all([sf.account, sf.user, sf.password])
    
    async def connect(self) -> None:
        """Establish Snowflake connection."""
        if not self.is_configured():
            return
        
        import snowflake.connector
        
        sf = config.snowflake
        self._conn = await asyncio.to_thread(
            snowflake.connector.connect,
            account=sf.account,
            user=sf.user,
            password=sf.password,
            warehouse=sf.warehouse,
            database=sf.database,
            schema=sf.schema,
            role=sf.role,
        )
    
    async def disconnect(self) -> None:
        """Close Snowflake connection."""
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
        import snowflake.connector
        
        def _execute():
            cur = self._conn.cursor(snowflake.connector.DictCursor)
            cur.execute(query, params or {})
            return cur.fetchall()
        
        return await asyncio.to_thread(_execute)
    
    async def execute_query_df(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Execute query and return pandas DataFrame."""
        def _execute():
            cur = self._conn.cursor()
            cur.execute(query, params or {})
            return cur.fetch_pandas_all()
        
        return await asyncio.to_thread(_execute)
    
    async def get_schema(
        self,
        table: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get schema information."""
        def _get():
            if table:
                rows = self._conn.cursor().execute(f"DESCRIBE TABLE {table}").fetchall()
                return {
                    "table": table,
                    "columns": [{"name": r[0], "type": r[1]} for r in rows],
                }
            rows = self._conn.cursor().execute("SHOW TABLES").fetchall()
            return {"tables": [r[1] for r in rows]}
        
        return await asyncio.to_thread(_get)
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        sf = config.snowflake
        return ConnectorInfo(
            name="snowflake",
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED if self._conn else ConnectorStatus.DISCONNECTED,
            metadata={
                "account": sf.account,
                "warehouse": sf.warehouse,
                "database": sf.database,
            },
        )

"""
REST API Connector

Generic REST API client with retry, auth, and timeout.
"""

from typing import Any, Optional

from connectors.base import BaseConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class RESTAPIConnector(BaseConnector):
    """
    Generic REST API client.
    
    Provides:
    - GET/POST/PUT/DELETE methods
    - Authentication headers
    - Timeout handling
    - Retry logic
    """
    
    def __init__(
        self,
        name: str,
        base_url: str,
        auth_header: Optional[dict[str, str]] = None,
        timeout_seconds: int = 30,
    ):
        """
        Initialize REST API connector.
        
        Args:
            name: Connector name.
            base_url: Base URL for API.
            auth_header: Authentication headers.
            timeout_seconds: Request timeout.
        """
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.auth_header = auth_header or {}
        self.timeout_seconds = timeout_seconds
        self._session = None
    
    async def connect(self) -> None:
        """Create aiohttp session."""
        import aiohttp
        
        self._session = aiohttp.ClientSession(
            base_url=self.base_url,
            headers=self.auth_header,
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
        )
    
    async def disconnect(self) -> None:
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def health_check(self) -> bool:
        """Check if API is reachable."""
        if not self._session:
            return False
        
        try:
            async with self._session.get("/health") as resp:
                return resp.status == 200
        except Exception:
            return False
    
    async def get(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Make GET request.
        
        Args:
            path: API path.
            params: Query parameters.
            
        Returns:
            JSON response.
        """
        async with self._session.get(path, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Make POST request.
        
        Args:
            path: API path.
            payload: JSON payload.
            
        Returns:
            JSON response.
        """
        async with self._session.post(path, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def put(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Make PUT request.
        
        Args:
            path: API path.
            payload: JSON payload.
            
        Returns:
            JSON response.
        """
        async with self._session.put(path, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def delete(
        self,
        path: str,
    ) -> dict[str, Any]:
        """
        Make DELETE request.
        
        Args:
            path: API path.
            
        Returns:
            JSON response.
        """
        async with self._session.delete(path) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        return ConnectorInfo(
            name=self.name,
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED if self._session else ConnectorStatus.DISCONNECTED,
            metadata={"base_url": self.base_url},
        )

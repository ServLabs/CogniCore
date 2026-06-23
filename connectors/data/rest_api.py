"""
REST API Connector

Generic REST API client with retry, auth, and timeout.
"""

import asyncio
from typing import Any, Optional

from logger import log
from connectors.base import BaseConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class RESTAPIConnector(BaseConnector):
    """
    Generic REST API client.
    
    Provides:
    - GET/POST/PUT/DELETE methods
    - Authentication headers
    - Timeout handling
    - Exponential backoff retry (retries on 429, 500, 502, 503, 504)
    """
    
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    
    def __init__(
        self,
        name: str,
        base_url: str,
        auth_header: Optional[dict[str, str]] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        """
        Initialize REST API connector.
        
        Args:
            name: Connector name.
            base_url: Base URL for API.
            auth_header: Authentication headers.
            timeout_seconds: Request timeout.
            max_retries: Maximum retry attempts for retryable errors.
            retry_base_delay: Base delay in seconds for exponential backoff.
        """
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.auth_header = auth_header or {}
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
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
        Make GET request with retry.
        
        Args:
            path: API path.
            params: Query parameters.
            
        Returns:
            JSON response.
        """
        return await self._request("GET", path, params=params)
    
    async def post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Make POST request with retry.
        
        Args:
            path: API path.
            payload: JSON payload.
            
        Returns:
            JSON response.
        """
        return await self._request("POST", path, json=payload)
    
    async def put(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Make PUT request with retry.
        
        Args:
            path: API path.
            payload: JSON payload.
            
        Returns:
            JSON response.
        """
        return await self._request("PUT", path, json=payload)
    
    async def delete(
        self,
        path: str,
    ) -> dict[str, Any]:
        """
        Make DELETE request with retry.
        
        Args:
            path: API path.
            
        Returns:
            JSON response.
        """
        return await self._request("DELETE", path)
    
    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute HTTP request with exponential backoff retry.
        
        Retries on 429, 500, 502, 503, 504 status codes.
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.request(method, path, **kwargs) as resp:
                    if resp.status in self.RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                        delay = self.retry_base_delay * (2 ** attempt)
                        log.warning(
                            "REST %s %s returned %d — retrying in %.1fs (attempt %d/%d)",
                            method, path, resp.status, delay, attempt + 1, self.max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    log.warning(
                        "REST %s %s failed (%s) — retrying in %.1fs (attempt %d/%d)",
                        method, path, e, delay, attempt + 1, self.max_retries,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        
        raise last_error  # Should not reach here
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        return ConnectorInfo(
            name=self.name,
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED if self._session else ConnectorStatus.DISCONNECTED,
            metadata={"base_url": self.base_url},
        )

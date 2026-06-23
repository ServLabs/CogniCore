"""
Embedding Connector

Pluggable embedding model connector. Supports:
- Local models: sentence-transformers compatible (BAAI/bge-*, nomic-*, etc.)
- API models: OpenAI text-embedding-3-* via API
"""

import asyncio
from typing import Any, Optional

import numpy as np

from config import config
from connectors.base import BaseConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class EmbeddingConnector(BaseConnector):
    """
    Pluggable embedding model connector.
    
    Supports:
    - Local sentence-transformers models
    - OpenAI embedding API
    
    Dimension is auto-detected from model.
    """
    
    def __init__(self):
        """Initialize embedding connector."""
        self._model = None
        self._api_client = None
        self._dimension: Optional[int] = None
    
    def is_configured(self) -> bool:
        """Check if embedding is configured."""
        return config.embedding.model_name is not None
    
    async def connect(self) -> None:
        """Initialize embedding model."""
        if config.embedding.use_api:
            # API-based embeddings (OpenAI)
            from openai import AsyncOpenAI
            self._api_client = AsyncOpenAI(api_key=config.embedding.api_key)
            
            # Dimension lookup for known models
            dim_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            self._dimension = dim_map.get(config.embedding.model_name, 1536)
        else:
            # Local sentence-transformers model
            from sentence_transformers import SentenceTransformer
            
            path = config.embedding.model_path or config.embedding.model_name
            self._model = await asyncio.to_thread(SentenceTransformer, path)
            self._dimension = self._model.get_sentence_embedding_dimension()
    
    async def disconnect(self) -> None:
        """Release embedding model."""
        self._model = None
        self._api_client = None
    
    async def health_check(self) -> bool:
        """Check if embedding model is working."""
        try:
            await self.embed_async("test")
            return True
        except Exception:
            return False
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension. Auto-detected from model."""
        return self._dimension or config.embedding.dimension
    
    async def embed_async(self, text: str) -> np.ndarray:
        """
        Embed a single text asynchronously.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding as float32 numpy array.
        """
        if config.embedding.use_api:
            resp = await self._api_client.embeddings.create(
                model=config.embedding.model_name,
                input=text,
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        else:
            return await asyncio.to_thread(
                self._model.encode,
                text,
                normalize_embeddings=True,
            )
    
    async def embed_batch_async(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts asynchronously.
        
        Args:
            texts: Texts to embed.
            
        Returns:
            Embeddings as (N, dim) float32 numpy array.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)
        
        if config.embedding.use_api:
            resp = await self._api_client.embeddings.create(
                model=config.embedding.model_name,
                input=texts,
            )
            return np.array([d.embedding for d in resp.data], dtype=np.float32)
        else:
            return await asyncio.to_thread(
                self._model.encode,
                texts,
                batch_size=config.embedding.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        return ConnectorInfo(
            name="embedder",
            connector_type=ConnectorType.AI,
            status=ConnectorStatus.CONNECTED if (self._model or self._api_client) else ConnectorStatus.DISCONNECTED,
            metadata={
                "model": config.embedding.model_name,
                "dimension": self._dimension,
                "use_api": config.embedding.use_api,
            },
        )

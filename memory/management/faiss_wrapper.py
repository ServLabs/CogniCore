"""
FAISS Auditing (Section 20)

Audited FAISS wrapper with full logging of all operations.
"""

import json
from pathlib import Path
from typing import Any, Optional

from audit import audit
from core import log


class AuditedFAISSIndex:
    """
    FAISS index wrapper with full audit logging.
    
    All operations (load, add, search, save) are logged.
    """
    
    def __init__(
        self,
        name: str,
        dimension: int,
        index_path: Path,
    ):
        """
        Initialize audited FAISS index.
        
        Args:
            name: Index name (for logging).
            dimension: Embedding dimension.
            index_path: Path to index file.
        """
        self.name = name
        self.dimension = dimension
        self.index_path = index_path
        self._index = None
        self._id_map: dict[int, str] = {}  # FAISS ID → external ID
    
    async def load(self) -> bool:
        """
        Load index from disk.
        
        Returns:
            True if loaded successfully.
        """
        try:
            import faiss
        except ImportError:
            log.error("faiss-cpu not installed")
            return False
        
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._load_id_map()
            
            audit.log_raw(
                "faiss",
                "load",
                "faiss_wrapper",
                "completed",
                target=self.name,
                details={"count": self._index.ntotal},
            )
            
            log.info(f"FAISS index {self.name} loaded: {self._index.ntotal} vectors")
            return True
        else:
            self._index = faiss.IndexFlatIP(self.dimension)
            
            audit.log_raw(
                "faiss",
                "create",
                "faiss_wrapper",
                "completed",
                target=self.name,
                details={"dimension": self.dimension},
            )
            
            log.info(f"FAISS index {self.name} created: dimension={self.dimension}")
            return True
    
    async def add(self, external_id: str, embedding) -> int:
        """
        Add a single embedding.
        
        Args:
            external_id: External ID for the embedding.
            embedding: Embedding vector (numpy array).
            
        Returns:
            FAISS ID assigned.
        """
        import numpy as np
        
        if self._index is None:
            await self.load()
        
        faiss_id = self._index.ntotal
        self._index.add(embedding.reshape(1, -1).astype(np.float32))
        self._id_map[faiss_id] = external_id
        
        audit.log_raw(
            "faiss",
            "add",
            "faiss_wrapper",
            "completed",
            target=self.name,
            details={"external_id": external_id, "faiss_id": faiss_id},
        )
        
        return faiss_id
    
    async def add_batch(
        self,
        external_ids: list[str],
        embeddings,
    ) -> int:
        """
        Add multiple embeddings.
        
        Args:
            external_ids: External IDs for the embeddings.
            embeddings: Embedding vectors (numpy array).
            
        Returns:
            Number of vectors added.
        """
        import numpy as np
        
        if self._index is None:
            await self.load()
        
        start_id = self._index.ntotal
        self._index.add(embeddings.astype(np.float32))
        
        for i, ext_id in enumerate(external_ids):
            self._id_map[start_id + i] = ext_id
        
        audit.log_raw(
            "faiss",
            "add_batch",
            "faiss_wrapper",
            "completed",
            target=self.name,
            details={"count": len(external_ids)},
        )
        
        return len(external_ids)
    
    async def search(
        self,
        query_embedding,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector (numpy array).
            top_k: Number of results to return.
            
        Returns:
            List of (external_id, score) tuples.
        """
        import numpy as np
        
        if self._index is None:
            await self.load()
        
        if self._index.ntotal == 0:
            return []
        
        distances, indices = self._index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(top_k, self._index.ntotal),
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self._id_map:
                results.append((self._id_map[idx], float(dist)))
        
        audit.log_raw(
            "faiss",
            "search",
            "faiss_wrapper",
            "completed",
            target=self.name,
            details={"top_k": top_k, "results": len(results)},
        )
        
        return results
    
    async def remove(self, external_id: str) -> bool:
        """
        Remove an embedding by external ID.
        
        Note: FAISS doesn't support efficient removal.
        This marks the ID as removed in the map.
        
        Args:
            external_id: External ID to remove.
            
        Returns:
            True if found and marked.
        """
        # Find FAISS ID
        faiss_id = None
        for fid, eid in self._id_map.items():
            if eid == external_id:
                faiss_id = fid
                break
        
        if faiss_id is not None:
            del self._id_map[faiss_id]
            
            audit.log_raw(
                "faiss",
                "remove",
                "faiss_wrapper",
                "completed",
                target=self.name,
                details={"external_id": external_id, "faiss_id": faiss_id},
            )
            
            return True
        
        return False
    
    async def save(self) -> bool:
        """
        Save index to disk.
        
        Returns:
            True if saved successfully.
        """
        try:
            import faiss
        except ImportError:
            return False
        
        if self._index is None:
            return False
        
        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self._index, str(self.index_path))
        self._save_id_map()
        
        audit.log_raw(
            "faiss",
            "save",
            "faiss_wrapper",
            "completed",
            target=self.name,
            details={"count": self._index.ntotal},
        )
        
        log.info(f"FAISS index {self.name} saved: {self._index.ntotal} vectors")
        
        return True
    
    def _load_id_map(self) -> None:
        """Load ID map from disk."""
        map_path = self.index_path.with_suffix(".idmap.json")
        
        if map_path.exists():
            try:
                data = json.loads(map_path.read_text())
                self._id_map = {int(k): v for k, v in data.items()}
            except (json.JSONDecodeError, IOError):
                self._id_map = {}
    
    def _save_id_map(self) -> None:
        """Save ID map to disk."""
        map_path = self.index_path.with_suffix(".idmap.json")
        map_path.write_text(json.dumps(self._id_map))
    
    @property
    def count(self) -> int:
        """Number of vectors in index."""
        if self._index is None:
            return 0
        return self._index.ntotal
    
    def get_status(self) -> dict[str, Any]:
        """Get index status."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "count": self.count,
            "id_map_size": len(self._id_map),
            "path": str(self.index_path),
        }

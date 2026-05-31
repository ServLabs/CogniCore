"""
Short Form Memory (SFM)

The agent's quick-reference cache — frequently accessed facts, compressed
summaries, common patterns, and hot knowledge. Analogous to sticky notes.

This module provides:
- Fact dataclass for compressed knowledge
- ShortFormMemory manager with in-memory + SQLite + FAISS storage
- Semantic fact search
- LRU eviction with pinning support
- Promotion tracking from LFM

Note: Domain values come from config.py for platform-agnostic design.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import numpy as np

from core import config


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Fact:
    """
    A compressed fact or summary.
    
    Short-line knowledge for instant recall.
    """
    id: str
    fact: str  # The actual fact text
    domain: str  # From config.domains
    source: str  # "promoted_from_lfm", "admin", "system_generated"
    source_ref: Optional[str] = None  # Reference to LFM chunk/document
    tags: list[str] = field(default_factory=list)
    pinned: bool = False  # Exempt from eviction
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def create(
        cls,
        fact: str,
        domain: Optional[str] = None,
        source: str = "admin",
        source_ref: Optional[str] = None,
        tags: Optional[list[str]] = None,
        pinned: bool = False,
    ) -> "Fact":
        """Factory method to create a new fact."""
        return cls(
            id=str(uuid.uuid4()),
            fact=fact,
            domain=domain or config.domain_config.default_domain,
            source=source,
            source_ref=source_ref,
            tags=tags or [],
            pinned=pinned,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "fact": self.fact,
            "domain": self.domain,
            "source": self.source,
            "source_ref": self.source_ref,
            "tags": self.tags,
            "pinned": self.pinned,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            fact=data["fact"],
            domain=data["domain"],
            source=data["source"],
            source_ref=data.get("source_ref"),
            tags=data.get("tags", []),
            pinned=data.get("pinned", False),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
        )
    
    def touch(self) -> None:
        """Update access stats."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


@dataclass
class FactSearchResult:
    """Result from fact search."""
    fact: Fact
    score: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "score": self.score,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Short Form Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class ShortFormMemory:
    """
    Manager for compressed facts and quick-reference knowledge.
    
    Provides:
    - In-memory hot cache for fastest reads
    - SQLite persistence with metadata
    - FAISS semantic search
    - LRU eviction with pinning support
    
    Attributes:
        db_path: Path to SQLite database.
        index_path: Path to FAISS index.
        embedder: Function to generate embeddings.
        max_facts: Maximum facts in memory (LRU eviction).
    """
    
    DEFAULT_MAX_FACTS = 1000
    PROMOTION_THRESHOLD = 10  # Access count to promote from LFM
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        index_path: Optional[str] = None,
        embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
        embedding_dim: int = 1024,
        max_facts: int = DEFAULT_MAX_FACTS,
    ):
        """
        Initialize Short Form Memory.
        
        Args:
            db_path: Path to SQLite database.
            index_path: Path to FAISS index.
            embedder: Async function to generate embeddings.
            embedding_dim: Dimension of embeddings.
            max_facts: Maximum facts before LRU eviction.
        """
        self.db_path = db_path or str(config.paths.hot_db)
        self.index_path = index_path or str(config.paths.faiss_dir / "sfm.index")
        self.id_map_path = self.index_path.replace(".index", ".idmap.json")
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.max_facts = max_facts
        
        # Lazy initialization
        self._faiss_index = None
        self._id_map: dict[int, str] = {}  # FAISS ID → Fact ID
        self._reverse_id_map: dict[str, int] = {}  # Fact ID → FAISS ID
        self._next_faiss_id = 0
        
        # In-memory hot cache
        self._facts: dict[str, Fact] = {}
        
        self._initialized = False
    
    # ── Initialization ──
    
    async def _init(self) -> None:
        """Initialize database and FAISS index."""
        if self._initialized:
            return
        
        await self._init_db()
        await self._init_faiss()
        await self._load_hot_facts()
        self._initialized = True
    
    async def _init_db(self) -> None:
        """Initialize SQLite tables."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS short_form_facts (
                    id TEXT PRIMARY KEY,
                    fact TEXT NOT NULL,
                    domain TEXT,
                    source TEXT DEFAULT 'admin',
                    source_ref TEXT,
                    tags TEXT,
                    pinned INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sfm_domain 
                ON short_form_facts(domain)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sfm_access 
                ON short_form_facts(access_count DESC)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sfm_pinned 
                ON short_form_facts(pinned)
            """)
            
            await db.commit()
    
    async def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        import faiss
        from pathlib import Path
        
        index_path = Path(self.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        if index_path.exists():
            self._faiss_index = faiss.read_index(str(index_path))
            
            id_map_path = Path(self.id_map_path)
            if id_map_path.exists():
                data = json.loads(id_map_path.read_text())
                self._id_map = {int(k): v for k, v in data["id_map"].items()}
                self._reverse_id_map = {v: int(k) for k, v in data["id_map"].items()}
                self._next_faiss_id = data.get("next_id", len(self._id_map))
        else:
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
    
    async def _save_faiss(self) -> None:
        """Save FAISS index to disk."""
        import faiss
        from pathlib import Path
        
        faiss.write_index(self._faiss_index, self.index_path)
        
        data = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "next_id": self._next_faiss_id,
        }
        Path(self.id_map_path).write_text(json.dumps(data, indent=2))
    
    async def _load_hot_facts(self) -> None:
        """Load most accessed facts into memory."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Load pinned facts first, then by access count
            async with db.execute(
                """
                SELECT * FROM short_form_facts 
                ORDER BY pinned DESC, access_count DESC 
                LIMIT ?
                """,
                (self.max_facts,)
            ) as cursor:
                async for row in cursor:
                    fact = self._row_to_fact(dict(row))
                    self._facts[fact.id] = fact
    
    def _row_to_fact(self, row: dict) -> Fact:
        """Convert database row to Fact."""
        return Fact(
            id=row["id"],
            fact=row["fact"],
            domain=row["domain"] or config.domain_config.default_domain,
            source=row["source"] or "admin",
            source_ref=row.get("source_ref"),
            tags=json.loads(row["tags"]) if row.get("tags") else [],
            pinned=bool(row.get("pinned", 0)),
            access_count=row.get("access_count", 0),
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row.get("last_accessed") else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
    
    # ── Database Operations ──
    
    async def _save_fact(self, fact: Fact) -> None:
        """Save fact to database."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO short_form_facts 
                (id, fact, domain, source, source_ref, tags, pinned, 
                 access_count, last_accessed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.id, fact.fact, fact.domain, fact.source, fact.source_ref,
                json.dumps(fact.tags), int(fact.pinned),
                fact.access_count,
                fact.last_accessed.isoformat() if fact.last_accessed else None,
                fact.created_at.isoformat(),
            ))
            await db.commit()
    
    async def _update_access(self, fact: Fact) -> None:
        """Update access stats in database."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE short_form_facts 
                SET access_count = ?, last_accessed = ?
                WHERE id = ?
            """, (
                fact.access_count,
                fact.last_accessed.isoformat() if fact.last_accessed else None,
                fact.id,
            ))
            await db.commit()
    
    async def _evict_if_needed(self) -> None:
        """Evict least recently used unpinned facts if over limit."""
        if len(self._facts) <= self.max_facts:
            return
        
        # Sort by pinned (keep), then by last_accessed (LRU)
        unpinned = [f for f in self._facts.values() if not f.pinned]
        unpinned.sort(key=lambda f: f.last_accessed or datetime.min.replace(tzinfo=timezone.utc))
        
        # Evict oldest unpinned
        to_evict = len(self._facts) - self.max_facts
        for fact in unpinned[:to_evict]:
            del self._facts[fact.id]
    
    # ── Public API ──
    
    async def add_fact(
        self,
        fact_text: str,
        domain: Optional[str] = None,
        source: str = "admin",
        source_ref: Optional[str] = None,
        tags: Optional[list[str]] = None,
        pinned: bool = False,
    ) -> Fact:
        """
        Add a new fact.
        
        Args:
            fact_text: The fact content.
            domain: Domain from config.
            source: "admin", "promoted_from_lfm", or "system_generated".
            source_ref: Reference to source (e.g., LFM chunk ID).
            tags: Searchable tags.
            pinned: Whether to pin (exempt from eviction).
            
        Returns:
            The created Fact.
        """
        await self._init()
        
        fact = Fact.create(
            fact=fact_text,
            domain=domain,
            source=source,
            source_ref=source_ref,
            tags=tags,
            pinned=pinned,
        )
        
        # Save to database
        await self._save_fact(fact)
        
        # Add to FAISS
        if self.embedder:
            import faiss
            
            embedding = await self.embedder(fact_text)
            embedding_np = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_np)
            
            self._faiss_index.add(embedding_np)
            self._id_map[self._next_faiss_id] = fact.id
            self._reverse_id_map[fact.id] = self._next_faiss_id
            self._next_faiss_id += 1
            
            await self._save_faiss()
        
        # Add to memory cache
        self._facts[fact.id] = fact
        await self._evict_if_needed()
        
        return fact
    
    async def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID."""
        await self._init()
        
        # Check memory cache
        if fact_id in self._facts:
            fact = self._facts[fact_id]
            fact.touch()
            await self._update_access(fact)
            return fact
        
        # Load from database
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM short_form_facts WHERE id = ?",
                (fact_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    fact = self._row_to_fact(dict(row))
                    fact.touch()
                    await self._update_access(fact)
                    
                    # Add to cache
                    self._facts[fact.id] = fact
                    await self._evict_if_needed()
                    
                    return fact
        
        return None
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
        domain: Optional[str] = None,
    ) -> list[FactSearchResult]:
        """
        Search for facts semantically.
        
        Args:
            query: Search query.
            top_k: Number of results.
            threshold: Minimum similarity score.
            domain: Filter by domain.
            
        Returns:
            List of FactSearchResult, ranked by relevance.
        """
        await self._init()
        
        if not self.embedder or self._faiss_index.ntotal == 0:
            return []
        
        import faiss
        
        # Embed query
        embedding = await self.embedder(query)
        embedding_np = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_np)
        
        # Search
        scores, indices = self._faiss_index.search(embedding_np, top_k * 2)  # Over-fetch for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            
            fact_id = self._id_map.get(int(idx))
            if not fact_id:
                continue
            
            fact = await self.get_fact(fact_id)
            if fact:
                # Apply domain filter
                if domain and fact.domain != domain:
                    continue
                
                results.append(FactSearchResult(fact=fact, score=float(score)))
                
                if len(results) >= top_k:
                    break
        
        return results
    
    async def find_by_domain(
        self,
        domain: str,
        limit: int = 50,
    ) -> list[Fact]:
        """Find facts by domain."""
        await self._init()
        
        import aiosqlite
        
        facts = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM short_form_facts 
                WHERE domain = ? 
                ORDER BY access_count DESC 
                LIMIT ?
                """,
                (domain, limit)
            ) as cursor:
                async for row in cursor:
                    facts.append(self._row_to_fact(dict(row)))
        
        return facts
    
    async def find_by_tags(
        self,
        tags: list[str],
        limit: int = 50,
    ) -> list[Fact]:
        """Find facts containing any of the specified tags."""
        await self._init()
        
        import aiosqlite
        
        # SQLite JSON search
        tag_conditions = " OR ".join(
            f"tags LIKE '%\"{tag}\"%'" for tag in tags
        )
        
        facts = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"""
                SELECT * FROM short_form_facts 
                WHERE {tag_conditions}
                ORDER BY access_count DESC 
                LIMIT ?
                """,
                (limit,)
            ) as cursor:
                async for row in cursor:
                    facts.append(self._row_to_fact(dict(row)))
        
        return facts
    
    async def pin_fact(self, fact_id: str) -> bool:
        """Pin a fact (exempt from eviction)."""
        await self._init()
        
        fact = await self.get_fact(fact_id)
        if not fact:
            return False
        
        fact.pinned = True
        await self._save_fact(fact)
        return True
    
    async def unpin_fact(self, fact_id: str) -> bool:
        """Unpin a fact."""
        await self._init()
        
        fact = await self.get_fact(fact_id)
        if not fact:
            return False
        
        fact.pinned = False
        await self._save_fact(fact)
        return True
    
    async def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact."""
        await self._init()
        
        import aiosqlite
        
        # Remove from memory
        if fact_id in self._facts:
            del self._facts[fact_id]
        
        # Remove from database
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM short_form_facts WHERE id = ?",
                (fact_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
    
    async def promote_from_lfm(
        self,
        lfm_chunk_id: str,
        compressed_fact: str,
        domain: str,
        tags: Optional[list[str]] = None,
    ) -> Fact:
        """
        Promote a frequently accessed LFM chunk to SFM.
        
        Args:
            lfm_chunk_id: ID of the source LFM chunk.
            compressed_fact: Compressed/summarized fact text.
            domain: Domain from config.
            tags: Tags for the fact.
            
        Returns:
            The created Fact.
        """
        return await self.add_fact(
            fact_text=compressed_fact,
            domain=domain,
            source="promoted_from_lfm",
            source_ref=lfm_chunk_id,
            tags=tags,
        )
    
    async def get_stats(self) -> dict[str, Any]:
        """Get SFM statistics."""
        await self._init()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            # Total facts
            async with db.execute("SELECT COUNT(*) FROM short_form_facts") as cursor:
                total = (await cursor.fetchone())[0]
            
            # Pinned facts
            async with db.execute("SELECT COUNT(*) FROM short_form_facts WHERE pinned = 1") as cursor:
                pinned = (await cursor.fetchone())[0]
            
            # By source
            async with db.execute(
                "SELECT source, COUNT(*) FROM short_form_facts GROUP BY source"
            ) as cursor:
                by_source = {row[0]: row[1] async for row in cursor}
        
        return {
            "total_facts": total,
            "in_memory": len(self._facts),
            "pinned": pinned,
            "by_source": by_source,
            "faiss_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_sfm_instance: Optional[ShortFormMemory] = None


def get_short_form_memory() -> ShortFormMemory:
    """
    Get the singleton ShortFormMemory instance.
    
    Returns:
        The ShortFormMemory instance.
    """
    global _sfm_instance
    if _sfm_instance is None:
        _sfm_instance = ShortFormMemory()
    return _sfm_instance


# ── Convenience Functions ──

async def add_fact(
    fact_text: str,
    domain: Optional[str] = None,
    tags: Optional[list[str]] = None,
    pinned: bool = False,
) -> Fact:
    """Add a fact to SFM."""
    return await get_short_form_memory().add_fact(
        fact_text=fact_text,
        domain=domain,
        tags=tags,
        pinned=pinned,
    )


async def search_facts(
    query: str,
    top_k: int = 10,
    domain: Optional[str] = None,
) -> list[FactSearchResult]:
    """Search for facts."""
    return await get_short_form_memory().search(
        query=query,
        top_k=top_k,
        domain=domain,
    )


async def get_fact(fact_id: str) -> Optional[Fact]:
    """Get a fact by ID."""
    return await get_short_form_memory().get_fact(fact_id)

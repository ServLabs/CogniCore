"""
Meta Memory

The agent's library card catalog — knows what the agent knows, where it lives,
and what's missing. A pointer-based index over all other memory types.

This module provides:
- MemoryPointer dataclass for routing to other memories
- KnowledgeGap dataclass for tracking unknown topics
- MetaMemory manager with in-memory + SQLite + FAISS storage
- Query routing with confidence-based fallback
- Gap detection and logging

Note: Domain values come from config.py for platform-agnostic design.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import numpy as np

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Enums (System values - universal, not domain-specific)
# ══════════════════════════════════════════════════════════════════════════════

class MemoryType(Enum):
    """Target memory types for routing."""
    ABM = "abm"  # Autobiographical Memory
    PM = "pm"    # Prospective Memory
    WM = "wm"    # Working Memory
    EM = "em"    # Emotional Memory
    AM = "am"    # Associative Memory
    MM = "mm"    # Motor Memory
    SFM = "sfm"  # Short Form Memory
    LFM = "lfm"  # Long Form Memory


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryPointer:
    """
    A pointer to knowledge in another memory.
    
    Short topic label pointing to the memory type and reference ID.
    """
    id: str
    topic: str  # Short topic label (2-5 words)
    memory_type: MemoryType
    ref_id: str  # Reference ID in target memory
    domain: str  # From config.domains
    faiss_id: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def create(
        cls,
        topic: str,
        memory_type: MemoryType,
        ref_id: str,
        domain: Optional[str] = None,
    ) -> "MemoryPointer":
        """Factory method to create a new pointer."""
        return cls(
            id=str(uuid.uuid4()),
            topic=topic,
            memory_type=memory_type,
            ref_id=ref_id,
            domain=domain or config.domain_config.default_domain,
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "memory_type": self.memory_type.value,
            "ref_id": self.ref_id,
            "domain": self.domain,
            "faiss_id": self.faiss_id,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryPointer":
        return cls(
            id=data["id"],
            topic=data["topic"],
            memory_type=MemoryType(data["memory_type"]),
            ref_id=data["ref_id"],
            domain=data["domain"],
            faiss_id=data.get("faiss_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class KnowledgeGap:
    """
    A detected gap in knowledge.
    
    Logged when queries have no match. High ask_count = priority ingestion target.
    """
    id: str
    topic: str  # What was asked about
    first_asked: datetime
    last_asked: datetime
    ask_count: int = 1
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_ref: Optional[str] = None  # Pointer ID that filled the gap
    
    @classmethod
    def create(cls, topic: str) -> "KnowledgeGap":
        """Factory method to create a new gap."""
        now = datetime.now(timezone.utc)
        return cls(
            id=str(uuid.uuid4()),
            topic=topic,
            first_asked=now,
            last_asked=now,
        )
    
    def touch(self) -> None:
        """Update ask stats."""
        self.ask_count += 1
        self.last_asked = datetime.now(timezone.utc)
    
    def resolve(self, pointer_id: str) -> None:
        """Mark gap as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now(timezone.utc)
        self.resolved_ref = pointer_id
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "first_asked": self.first_asked.isoformat(),
            "last_asked": self.last_asked.isoformat(),
            "ask_count": self.ask_count,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_ref": self.resolved_ref,
        }


@dataclass
class RouteResult:
    """Result of query routing."""
    pointer: Optional[MemoryPointer]
    confidence: float
    is_gap: bool = False
    gap_id: Optional[str] = None
    
    @property
    def has_match(self) -> bool:
        return self.pointer is not None and self.confidence > 0.5
    
    @property
    def needs_fallback(self) -> bool:
        """Low confidence - should also search SFM/LFM."""
        return self.pointer is not None and 0.3 < self.confidence <= 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Meta Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class MetaMemory:
    """
    Manager for memory pointers and knowledge gaps.
    
    Provides:
    - Pointer registration during consolidation
    - Semantic query routing
    - Gap detection and logging
    - Knowledge inventory
    
    Attributes:
        db_path: Path to SQLite database.
        index_path: Path to FAISS index.
        embedder: Function to generate embeddings.
    """
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.7
    LOW_CONFIDENCE = 0.3
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        index_path: Optional[str] = None,
        embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
        embedding_dim: int = 1024,
    ):
        """
        Initialize Meta Memory.
        
        Args:
            db_path: Path to SQLite database.
            index_path: Path to FAISS index.
            embedder: Async function to generate embeddings.
            embedding_dim: Dimension of embeddings.
        """
        self.db_path = db_path or str(config.paths.hot_db)
        self.cold_db_path = str(config.paths.cold_db)
        self.index_path = index_path or str(config.paths.faiss_dir / "meta.index")
        self.id_map_path = self.index_path.replace(".index", ".idmap.json")
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        
        # Lazy initialization
        self._faiss_index = None
        self._id_map: dict[int, str] = {}  # FAISS ID → Pointer ID
        self._reverse_id_map: dict[str, int] = {}  # Pointer ID → FAISS ID
        self._next_faiss_id = 0
        
        # In-memory cache
        self._pointers: dict[str, MemoryPointer] = {}
        self._by_topic: dict[str, str] = {}  # topic → pointer_id
        self._gaps: dict[str, KnowledgeGap] = {}
        
        self._initialized = False
    
    # ── Initialization ──
    
    async def _init(self) -> None:
        """Initialize database and FAISS index."""
        if self._initialized:
            return
        
        await self._init_db()
        await self._init_faiss()
        await self._load_pointers()
        self._initialized = True
    
    async def _init_db(self) -> None:
        """Initialize SQLite tables."""
        import aiosqlite
        
        # Pointers in hot DB
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_pointers (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    domain TEXT,
                    faiss_id INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_pointers_topic 
                ON memory_pointers(topic)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_pointers_type 
                ON memory_pointers(memory_type)
            """)
            
            await db.commit()
        
        # Gaps in cold DB
        async with aiosqlite.connect(self.cold_db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_gaps (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    first_asked TEXT NOT NULL,
                    last_asked TEXT NOT NULL,
                    ask_count INTEGER DEFAULT 1,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT,
                    resolved_ref TEXT
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_gaps_topic 
                ON knowledge_gaps(topic)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_gaps_count 
                ON knowledge_gaps(ask_count DESC)
            """)
            
            await db.commit()
    
    async def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        import faiss
        
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
        
        faiss.write_index(self._faiss_index, self.index_path)
        
        data = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "next_id": self._next_faiss_id,
        }
        Path(self.id_map_path).write_text(json.dumps(data, indent=2))
    
    async def _load_pointers(self) -> None:
        """Load all pointers into memory."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM memory_pointers") as cursor:
                async for row in cursor:
                    pointer = self._row_to_pointer(dict(row))
                    self._pointers[pointer.id] = pointer
                    self._by_topic[pointer.topic.lower()] = pointer.id
    
    def _row_to_pointer(self, row: dict) -> MemoryPointer:
        """Convert database row to MemoryPointer."""
        return MemoryPointer(
            id=row["id"],
            topic=row["topic"],
            memory_type=MemoryType(row["memory_type"]),
            ref_id=row["ref_id"],
            domain=row["domain"] or config.domain_config.default_domain,
            faiss_id=row.get("faiss_id"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
    
    def _row_to_gap(self, row: dict) -> KnowledgeGap:
        """Convert database row to KnowledgeGap."""
        return KnowledgeGap(
            id=row["id"],
            topic=row["topic"],
            first_asked=datetime.fromisoformat(row["first_asked"]),
            last_asked=datetime.fromisoformat(row["last_asked"]),
            ask_count=row["ask_count"],
            resolved=bool(row.get("resolved", 0)),
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row.get("resolved_at") else None,
            resolved_ref=row.get("resolved_ref"),
        )
    
    # ── Database Operations ──
    
    async def _save_pointer(self, pointer: MemoryPointer) -> None:
        """Save pointer to database."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO memory_pointers 
                (id, topic, memory_type, ref_id, domain, faiss_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pointer.id, pointer.topic, pointer.memory_type.value,
                pointer.ref_id, pointer.domain, pointer.faiss_id,
                pointer.created_at.isoformat(),
            ))
            await db.commit()
        
        self._pointers[pointer.id] = pointer
        self._by_topic[pointer.topic.lower()] = pointer.id
    
    async def _save_gap(self, gap: KnowledgeGap) -> None:
        """Save gap to database."""
        import aiosqlite
        
        async with aiosqlite.connect(self.cold_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO knowledge_gaps 
                (id, topic, first_asked, last_asked, ask_count, 
                 resolved, resolved_at, resolved_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gap.id, gap.topic, gap.first_asked.isoformat(),
                gap.last_asked.isoformat(), gap.ask_count,
                int(gap.resolved),
                gap.resolved_at.isoformat() if gap.resolved_at else None,
                gap.resolved_ref,
            ))
            await db.commit()
        
        self._gaps[gap.id] = gap
    
    # ── Public API ──
    
    async def register_pointer(
        self,
        topic: str,
        memory_type: MemoryType,
        ref_id: str,
        domain: Optional[str] = None,
    ) -> MemoryPointer:
        """
        Register a pointer to knowledge.
        
        Called during MML consolidation, not on every write.
        
        Args:
            topic: Short topic label (2-5 words).
            memory_type: Target memory type.
            ref_id: Reference ID in target memory.
            domain: Domain from config.
            
        Returns:
            The created MemoryPointer.
        """
        await self._init()
        
        pointer = MemoryPointer.create(
            topic=topic,
            memory_type=memory_type,
            ref_id=ref_id,
            domain=domain,
        )
        
        # Embed and add to FAISS
        if self.embedder:
            import faiss
            
            embedding = await self.embedder(topic)
            embedding_np = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_np)
            
            self._faiss_index.add(embedding_np)
            pointer.faiss_id = self._next_faiss_id
            self._id_map[self._next_faiss_id] = pointer.id
            self._reverse_id_map[pointer.id] = self._next_faiss_id
            self._next_faiss_id += 1
            
            await self._save_faiss()
        
        await self._save_pointer(pointer)
        
        # Check if this resolves any gaps
        await self._check_gap_resolution(topic, pointer.id)
        
        return pointer
    
    async def _check_gap_resolution(self, topic: str, pointer_id: str) -> None:
        """Check if a new pointer resolves any knowledge gaps."""
        import aiosqlite
        
        # Simple substring match for now
        topic_lower = topic.lower()
        
        async with aiosqlite.connect(self.cold_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM knowledge_gaps WHERE resolved = 0"
            ) as cursor:
                async for row in cursor:
                    gap = self._row_to_gap(dict(row))
                    if topic_lower in gap.topic.lower() or gap.topic.lower() in topic_lower:
                        gap.resolve(pointer_id)
                        await self._save_gap(gap)
    
    async def route_query(self, query: str) -> RouteResult:
        """
        Route a query to the appropriate memory.
        
        Args:
            query: User query.
            
        Returns:
            RouteResult with pointer and confidence.
        """
        await self._init()
        
        # Rule-based routing for identity queries
        identity_keywords = ["who are you", "what are you", "your name", "about yourself"]
        if any(kw in query.lower() for kw in identity_keywords):
            # Route to ABM
            abm_pointer = MemoryPointer.create(
                topic="agent identity",
                memory_type=MemoryType.ABM,
                ref_id="identity",
            )
            return RouteResult(pointer=abm_pointer, confidence=1.0)
        
        # Semantic search
        if not self.embedder or self._faiss_index.ntotal == 0:
            return RouteResult(pointer=None, confidence=0.0, is_gap=True)
        
        import faiss
        
        embedding = await self.embedder(query)
        embedding_np = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_np)
        
        scores, indices = self._faiss_index.search(embedding_np, 1)
        
        if indices[0][0] == -1 or scores[0][0] < self.LOW_CONFIDENCE:
            # No match - log gap
            gap = await self._log_gap(query)
            return RouteResult(
                pointer=None,
                confidence=float(scores[0][0]) if indices[0][0] != -1 else 0.0,
                is_gap=True,
                gap_id=gap.id,
            )
        
        pointer_id = self._id_map.get(int(indices[0][0]))
        if not pointer_id or pointer_id not in self._pointers:
            gap = await self._log_gap(query)
            return RouteResult(pointer=None, confidence=0.0, is_gap=True, gap_id=gap.id)
        
        pointer = self._pointers[pointer_id]
        confidence = float(scores[0][0])
        
        return RouteResult(pointer=pointer, confidence=confidence)
    
    async def _log_gap(self, query: str) -> KnowledgeGap:
        """Log a knowledge gap."""
        # Extract topic from query (simplified - could use LLM)
        topic = query[:100]  # Truncate long queries
        
        # Check if gap already exists
        import aiosqlite
        
        async with aiosqlite.connect(self.cold_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM knowledge_gaps WHERE topic = ? AND resolved = 0",
                (topic,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    gap = self._row_to_gap(dict(row))
                    gap.touch()
                    await self._save_gap(gap)
                    return gap
        
        # Create new gap
        gap = KnowledgeGap.create(topic)
        await self._save_gap(gap)
        return gap
    
    async def get_pointer(self, pointer_id: str) -> Optional[MemoryPointer]:
        """Get a pointer by ID."""
        await self._init()
        return self._pointers.get(pointer_id)
    
    async def find_by_topic(self, topic: str) -> Optional[MemoryPointer]:
        """Find pointer by exact topic match."""
        await self._init()
        pointer_id = self._by_topic.get(topic.lower())
        return self._pointers.get(pointer_id) if pointer_id else None
    
    async def find_by_memory_type(
        self,
        memory_type: MemoryType,
        limit: int = 50,
    ) -> list[MemoryPointer]:
        """Find pointers by memory type."""
        await self._init()
        
        return [
            p for p in self._pointers.values()
            if p.memory_type == memory_type
        ][:limit]
    
    async def get_knowledge_inventory(
        self,
        domain: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """
        Get inventory of what the agent knows.
        
        Returns:
            Dict mapping memory type to list of topics.
        """
        await self._init()
        
        inventory: dict[str, list[str]] = {}
        
        for pointer in self._pointers.values():
            if domain and pointer.domain != domain:
                continue
            
            mem_type = pointer.memory_type.value
            if mem_type not in inventory:
                inventory[mem_type] = []
            inventory[mem_type].append(pointer.topic)
        
        return inventory
    
    async def get_top_gaps(self, limit: int = 20) -> list[KnowledgeGap]:
        """Get top knowledge gaps by ask count."""
        await self._init()
        
        import aiosqlite
        
        gaps = []
        async with aiosqlite.connect(self.cold_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM knowledge_gaps 
                WHERE resolved = 0 
                ORDER BY ask_count DESC 
                LIMIT ?
                """,
                (limit,)
            ) as cursor:
                async for row in cursor:
                    gaps.append(self._row_to_gap(dict(row)))
        
        return gaps
    
    async def delete_pointer(self, pointer_id: str) -> bool:
        """Delete a pointer."""
        await self._init()
        
        import aiosqlite
        
        if pointer_id in self._pointers:
            pointer = self._pointers[pointer_id]
            del self._pointers[pointer_id]
            if pointer.topic.lower() in self._by_topic:
                del self._by_topic[pointer.topic.lower()]
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM memory_pointers WHERE id = ?",
                (pointer_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
    
    async def get_stats(self) -> dict[str, Any]:
        """Get Meta Memory statistics."""
        await self._init()
        
        import aiosqlite
        
        # Pointers by type
        by_type: dict[str, int] = {}
        for pointer in self._pointers.values():
            mem_type = pointer.memory_type.value
            by_type[mem_type] = by_type.get(mem_type, 0) + 1
        
        # Gap stats
        async with aiosqlite.connect(self.cold_db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM knowledge_gaps WHERE resolved = 0") as cursor:
                open_gaps = (await cursor.fetchone())[0]
            
            async with db.execute("SELECT COUNT(*) FROM knowledge_gaps WHERE resolved = 1") as cursor:
                resolved_gaps = (await cursor.fetchone())[0]
        
        return {
            "total_pointers": len(self._pointers),
            "by_memory_type": by_type,
            "open_gaps": open_gaps,
            "resolved_gaps": resolved_gaps,
            "faiss_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_meta_instance: Optional[MetaMemory] = None


def get_meta_memory() -> MetaMemory:
    """
    Get the singleton MetaMemory instance.
    
    Returns:
        The MetaMemory instance.
    """
    global _meta_instance
    if _meta_instance is None:
        _meta_instance = MetaMemory()
    return _meta_instance


# ── Convenience Functions ──

async def register_knowledge(
    topic: str,
    memory_type: MemoryType,
    ref_id: str,
    domain: Optional[str] = None,
) -> MemoryPointer:
    """Register a pointer to knowledge."""
    return await get_meta_memory().register_pointer(
        topic=topic,
        memory_type=memory_type,
        ref_id=ref_id,
        domain=domain,
    )


async def route_query(query: str) -> RouteResult:
    """Route a query to the appropriate memory."""
    return await get_meta_memory().route_query(query)


async def what_do_i_know(domain: Optional[str] = None) -> dict[str, list[str]]:
    """Get knowledge inventory."""
    return await get_meta_memory().get_knowledge_inventory(domain)


async def what_dont_i_know(limit: int = 20) -> list[KnowledgeGap]:
    """Get top knowledge gaps."""
    return await get_meta_memory().get_top_gaps(limit)

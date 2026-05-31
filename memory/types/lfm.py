"""
Long Form Memory (LFM)

The agent's deep knowledge store — full documents, detailed experiences,
ingested domain knowledge. Analogous to a filing cabinet and library.

This module provides:
- Document and Chunk dataclasses
- LongFormMemory manager with file + SQLite + FAISS storage
- Chunking and embedding for RAG retrieval
- Access tracking for SFM promotion

Note: Domain values come from config.py for platform-agnostic design.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import numpy as np

from core import config


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CHUNK_SIZE = 512  # tokens
DEFAULT_CHUNK_OVERLAP = 64  # tokens
PROMOTION_THRESHOLD = 10  # Access count to trigger SFM promotion


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """
    A chunk of a document for RAG retrieval.
    """
    id: str
    document_id: str
    chunk_index: int
    content: str
    faiss_id: Optional[int] = None
    access_count: int = 0
    promoted_to_sfm: bool = False
    
    @classmethod
    def create(
        cls,
        document_id: str,
        chunk_index: int,
        content: str,
    ) -> "Chunk":
        """Factory method to create a new chunk."""
        return cls(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "faiss_id": self.faiss_id,
            "access_count": self.access_count,
            "promoted_to_sfm": self.promoted_to_sfm,
        }
    
    def touch(self) -> None:
        """Update access count."""
        self.access_count += 1


@dataclass
class Document:
    """
    A full document in long form memory.
    """
    id: str
    title: str
    file_path: str
    domain: str  # From config.domains
    source: str  # "admin_upload", "ingested", "learned"
    tags: list[str] = field(default_factory=list)
    chunk_count: int = 0
    access_count: int = 0
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def create(
        cls,
        title: str,
        file_path: str,
        domain: Optional[str] = None,
        source: str = "admin_upload",
        tags: Optional[list[str]] = None,
    ) -> "Document":
        """Factory method to create a new document."""
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            file_path=file_path,
            domain=domain or config.domain_config.default_domain,
            source=source,
            tags=tags or [],
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "file_path": self.file_path,
            "domain": self.domain,
            "source": self.source,
            "tags": self.tags,
            "chunk_count": self.chunk_count,
            "access_count": self.access_count,
            "ingested_at": self.ingested_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            id=data["id"],
            title=data["title"],
            file_path=data["file_path"],
            domain=data["domain"],
            source=data["source"],
            tags=data.get("tags", []),
            chunk_count=data.get("chunk_count", 0),
            access_count=data.get("access_count", 0),
            ingested_at=datetime.fromisoformat(data["ingested_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class ChunkSearchResult:
    """Result from chunk search."""
    chunk: Chunk
    document: Document
    score: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "document": self.document.to_dict(),
            "score": self.score,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Long Form Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class LongFormMemory:
    """
    Manager for full documents and deep knowledge.
    
    Provides:
    - File-based document storage (human-editable .md files)
    - SQLite metadata index
    - FAISS semantic search over chunks
    - Access tracking for SFM promotion
    
    Attributes:
        docs_dir: Directory for document files.
        db_path: Path to SQLite database.
        index_path: Path to FAISS index.
        embedder: Function to generate embeddings.
    """
    
    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        db_path: Optional[str] = None,
        index_path: Optional[str] = None,
        embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
        embedding_dim: int = 1024,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize Long Form Memory.
        
        Args:
            docs_dir: Directory for document files.
            db_path: Path to SQLite database.
            index_path: Path to FAISS index.
            embedder: Async function to generate embeddings.
            embedding_dim: Dimension of embeddings.
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
        """
        self.docs_dir = docs_dir or config.paths.md_dir
        self.db_path = db_path or str(config.paths.hot_db)
        self.index_path = index_path or str(config.paths.faiss_dir / "lfm.index")
        self.id_map_path = self.index_path.replace(".index", ".idmap.json")
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Lazy initialization
        self._faiss_index = None
        self._id_map: dict[int, str] = {}  # FAISS ID → Chunk ID
        self._reverse_id_map: dict[str, int] = {}  # Chunk ID → FAISS ID
        self._next_faiss_id = 0
        
        # In-memory cache
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        
        self._initialized = False
    
    # ── Initialization ──
    
    async def _init(self) -> None:
        """Initialize database and FAISS index."""
        if self._initialized:
            return
        
        await self._init_db()
        await self._init_faiss()
        self._initialized = True
    
    async def _init_db(self) -> None:
        """Initialize SQLite tables."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            # Documents table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS long_form_documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    domain TEXT,
                    source TEXT DEFAULT 'admin_upload',
                    tags TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    ingested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Chunks table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS lfm_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    faiss_id INTEGER,
                    access_count INTEGER DEFAULT 0,
                    promoted_to_sfm INTEGER DEFAULT 0,
                    FOREIGN KEY (document_id) REFERENCES long_form_documents(id)
                )
            """)
            
            # Indexes
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_lfm_docs_domain 
                ON long_form_documents(domain)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_lfm_chunks_doc 
                ON lfm_chunks(document_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_lfm_chunks_access 
                ON lfm_chunks(access_count DESC)
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
    
    # ── Chunking ──
    
    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks.
        
        Uses simple character-based chunking with overlap.
        For production, consider tiktoken for accurate token counts.
        """
        # Rough estimate: 4 chars per token
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_chunk_size
            
            # Try to break at paragraph or sentence
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + char_chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", ".\n", "! ", "? "]:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + char_chunk_size // 2:
                            end = sent_break + len(sep)
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - char_overlap
        
        return chunks
    
    # ── Database Operations ──
    
    async def _save_document(self, doc: Document) -> None:
        """Save document to database."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO long_form_documents 
                (id, title, file_path, domain, source, tags, 
                 chunk_count, access_count, ingested_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.title, doc.file_path, doc.domain, doc.source,
                json.dumps(doc.tags), doc.chunk_count, doc.access_count,
                doc.ingested_at.isoformat(), doc.updated_at.isoformat(),
            ))
            await db.commit()
        
        self._documents[doc.id] = doc
    
    async def _save_chunk(self, chunk: Chunk) -> None:
        """Save chunk to database."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO lfm_chunks 
                (id, document_id, chunk_index, content, faiss_id, 
                 access_count, promoted_to_sfm)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.id, chunk.document_id, chunk.chunk_index, chunk.content,
                chunk.faiss_id, chunk.access_count, int(chunk.promoted_to_sfm),
            ))
            await db.commit()
        
        self._chunks[chunk.id] = chunk
    
    async def _update_chunk_access(self, chunk: Chunk) -> None:
        """Update chunk access count."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE lfm_chunks SET access_count = ? WHERE id = ?
            """, (chunk.access_count, chunk.id))
            
            # Also update document access count
            await db.execute("""
                UPDATE long_form_documents 
                SET access_count = access_count + 1 
                WHERE id = ?
            """, (chunk.document_id,))
            
            await db.commit()
    
    def _row_to_document(self, row: dict) -> Document:
        """Convert database row to Document."""
        return Document(
            id=row["id"],
            title=row["title"],
            file_path=row["file_path"],
            domain=row["domain"] or config.domain_config.default_domain,
            source=row["source"] or "admin_upload",
            tags=json.loads(row["tags"]) if row.get("tags") else [],
            chunk_count=row.get("chunk_count", 0),
            access_count=row.get("access_count", 0),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
    
    def _row_to_chunk(self, row: dict) -> Chunk:
        """Convert database row to Chunk."""
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            faiss_id=row.get("faiss_id"),
            access_count=row.get("access_count", 0),
            promoted_to_sfm=bool(row.get("promoted_to_sfm", 0)),
        )
    
    # ── Public API ──
    
    async def ingest_document(
        self,
        title: str,
        content: str,
        domain: Optional[str] = None,
        source: str = "admin_upload",
        tags: Optional[list[str]] = None,
        file_path: Optional[str] = None,
    ) -> Document:
        """
        Ingest a document into LFM.
        
        Chunks the content, embeds chunks, and indexes in FAISS.
        
        Args:
            title: Document title.
            content: Full document content.
            domain: Domain from config.
            source: "admin_upload", "ingested", or "learned".
            tags: Searchable tags.
            file_path: Optional path to save .md file.
            
        Returns:
            The created Document.
        """
        await self._init()
        
        # Save to file if path provided, otherwise generate one
        if not file_path:
            safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
            safe_title = safe_title.replace(" ", "_").lower()
            file_path = str(self.docs_dir / domain or "general" / f"{safe_title}.md")
        
        # Ensure directory exists and write file
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        
        # Create document
        doc = Document.create(
            title=title,
            file_path=file_path,
            domain=domain,
            source=source,
            tags=tags,
        )
        
        # Chunk content
        chunk_texts = self._chunk_text(content)
        doc.chunk_count = len(chunk_texts)
        
        # Save document
        await self._save_document(doc)
        
        # Create and embed chunks
        for i, chunk_text in enumerate(chunk_texts):
            chunk = Chunk.create(
                document_id=doc.id,
                chunk_index=i,
                content=chunk_text,
            )
            
            # Embed and add to FAISS
            if self.embedder:
                import faiss
                
                embedding = await self.embedder(chunk_text)
                embedding_np = np.array([embedding], dtype=np.float32)
                faiss.normalize_L2(embedding_np)
                
                self._faiss_index.add(embedding_np)
                chunk.faiss_id = self._next_faiss_id
                self._id_map[self._next_faiss_id] = chunk.id
                self._reverse_id_map[chunk.id] = self._next_faiss_id
                self._next_faiss_id += 1
            
            await self._save_chunk(chunk)
        
        # Save FAISS index
        if self.embedder:
            await self._save_faiss()
        
        return doc
    
    async def ingest_file(
        self,
        file_path: str,
        domain: Optional[str] = None,
        source: str = "admin_upload",
        tags: Optional[list[str]] = None,
    ) -> Document:
        """
        Ingest a document from a file.
        
        Args:
            file_path: Path to .md file.
            domain: Domain from config.
            source: Source type.
            tags: Searchable tags.
            
        Returns:
            The created Document.
        """
        path = Path(file_path)
        content = path.read_text()
        title = path.stem.replace("_", " ").title()
        
        return await self.ingest_document(
            title=title,
            content=content,
            domain=domain,
            source=source,
            tags=tags,
            file_path=file_path,
        )
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        await self._init()
        
        if doc_id in self._documents:
            return self._documents[doc_id]
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM long_form_documents WHERE id = ?",
                (doc_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    doc = self._row_to_document(dict(row))
                    self._documents[doc.id] = doc
                    return doc
        
        return None
    
    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        await self._init()
        
        if chunk_id in self._chunks:
            return self._chunks[chunk_id]
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM lfm_chunks WHERE id = ?",
                (chunk_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    chunk = self._row_to_chunk(dict(row))
                    self._chunks[chunk.id] = chunk
                    return chunk
        
        return None
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
        domain: Optional[str] = None,
    ) -> list[ChunkSearchResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query.
            top_k: Number of results.
            threshold: Minimum similarity score.
            domain: Filter by domain.
            
        Returns:
            List of ChunkSearchResult, ranked by relevance.
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
        scores, indices = self._faiss_index.search(embedding_np, top_k * 2)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            
            chunk_id = self._id_map.get(int(idx))
            if not chunk_id:
                continue
            
            chunk = await self.get_chunk(chunk_id)
            if not chunk:
                continue
            
            doc = await self.get_document(chunk.document_id)
            if not doc:
                continue
            
            # Apply domain filter
            if domain and doc.domain != domain:
                continue
            
            # Update access count
            chunk.touch()
            await self._update_chunk_access(chunk)
            
            results.append(ChunkSearchResult(
                chunk=chunk,
                document=doc,
                score=float(score),
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    async def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        await self._init()
        
        import aiosqlite
        
        chunks = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM lfm_chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,)
            ) as cursor:
                async for row in cursor:
                    chunks.append(self._row_to_chunk(dict(row)))
        
        return chunks
    
    async def find_by_domain(
        self,
        domain: str,
        limit: int = 50,
    ) -> list[Document]:
        """Find documents by domain."""
        await self._init()
        
        import aiosqlite
        
        docs = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM long_form_documents 
                WHERE domain = ? 
                ORDER BY access_count DESC 
                LIMIT ?
                """,
                (domain, limit)
            ) as cursor:
                async for row in cursor:
                    docs.append(self._row_to_document(dict(row)))
        
        return docs
    
    async def get_promotion_candidates(
        self,
        threshold: int = PROMOTION_THRESHOLD,
        limit: int = 50,
    ) -> list[Chunk]:
        """
        Get chunks that should be promoted to SFM.
        
        Args:
            threshold: Minimum access count.
            limit: Maximum candidates.
            
        Returns:
            List of chunks ready for promotion.
        """
        await self._init()
        
        import aiosqlite
        
        chunks = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM lfm_chunks 
                WHERE access_count >= ? AND promoted_to_sfm = 0
                ORDER BY access_count DESC 
                LIMIT ?
                """,
                (threshold, limit)
            ) as cursor:
                async for row in cursor:
                    chunks.append(self._row_to_chunk(dict(row)))
        
        return chunks
    
    async def mark_promoted(self, chunk_id: str) -> None:
        """Mark a chunk as promoted to SFM."""
        await self._init()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE lfm_chunks SET promoted_to_sfm = 1 WHERE id = ?",
                (chunk_id,)
            )
            await db.commit()
        
        if chunk_id in self._chunks:
            self._chunks[chunk_id].promoted_to_sfm = True
    
    async def scan_directory(self, domain: Optional[str] = None) -> list[Document]:
        """
        Scan documents directory and ingest new files.
        
        Args:
            domain: Domain to assign to new documents.
            
        Returns:
            List of newly ingested documents.
        """
        await self._init()
        
        import aiosqlite
        
        # Get already ingested files
        ingested_paths = set()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT file_path FROM long_form_documents") as cursor:
                async for row in cursor:
                    ingested_paths.add(row[0])
        
        # Find new .md files
        new_docs = []
        for md_file in self.docs_dir.rglob("*.md"):
            if str(md_file) not in ingested_paths:
                # Infer domain from directory structure
                rel_path = md_file.relative_to(self.docs_dir)
                inferred_domain = domain or (
                    rel_path.parts[0] if len(rel_path.parts) > 1 else config.domain_config.default_domain
                )
                
                doc = await self.ingest_file(
                    file_path=str(md_file),
                    domain=inferred_domain,
                )
                new_docs.append(doc)
        
        return new_docs
    
    async def get_stats(self) -> dict[str, Any]:
        """Get LFM statistics."""
        await self._init()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            # Total documents
            async with db.execute("SELECT COUNT(*) FROM long_form_documents") as cursor:
                total_docs = (await cursor.fetchone())[0]
            
            # Total chunks
            async with db.execute("SELECT COUNT(*) FROM lfm_chunks") as cursor:
                total_chunks = (await cursor.fetchone())[0]
            
            # Promoted chunks
            async with db.execute("SELECT COUNT(*) FROM lfm_chunks WHERE promoted_to_sfm = 1") as cursor:
                promoted = (await cursor.fetchone())[0]
            
            # By domain
            async with db.execute(
                "SELECT domain, COUNT(*) FROM long_form_documents GROUP BY domain"
            ) as cursor:
                by_domain = {row[0]: row[1] async for row in cursor}
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "promoted_to_sfm": promoted,
            "by_domain": by_domain,
            "faiss_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_lfm_instance: Optional[LongFormMemory] = None


def get_long_form_memory() -> LongFormMemory:
    """
    Get the singleton LongFormMemory instance.
    
    Returns:
        The LongFormMemory instance.
    """
    global _lfm_instance
    if _lfm_instance is None:
        _lfm_instance = LongFormMemory()
    return _lfm_instance


# ── Convenience Functions ──

async def ingest_document(
    title: str,
    content: str,
    domain: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Document:
    """Ingest a document into LFM."""
    return await get_long_form_memory().ingest_document(
        title=title,
        content=content,
        domain=domain,
        tags=tags,
    )


async def search_documents(
    query: str,
    top_k: int = 10,
    domain: Optional[str] = None,
) -> list[ChunkSearchResult]:
    """Search for relevant document chunks."""
    return await get_long_form_memory().search(
        query=query,
        top_k=top_k,
        domain=domain,
    )


async def get_document(doc_id: str) -> Optional[Document]:
    """Get a document by ID."""
    return await get_long_form_memory().get_document(doc_id)

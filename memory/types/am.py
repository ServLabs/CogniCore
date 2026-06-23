"""
Associative Memory (AM)

The agent's knowledge graph — connects concepts, entities, and relationships
to enable multi-hop reasoning and semantic discovery.

This module provides:
- Entity and Relationship dataclasses
- Hybrid architecture: Kuzu (explicit graph) + FAISS (semantic vectors)
- Multi-hop traversal and semantic similarity search
- Merged results for GraphRAG-style reasoning

Note: Entity types, relation types, and domains are loaded from config.py
to keep the platform domain-agnostic. No hardcoded enums.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import numpy as np

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """
    A node in the knowledge graph.
    
    Represents a concept, dataset, user, or any other entity.
    Entity types and domains are strings loaded from config.
    """
    id: str
    name: str
    entity_type: str  # From config.entity_types
    domain: str = ""  # From config.domains, defaults to config.domain_config.default_domain
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Set default domain from config if not provided."""
        if not self.domain:
            self.domain = config.domain_config.default_domain
    
    @classmethod
    def create(
        cls,
        name: str,
        entity_type: str,
        domain: Optional[str] = None,
        description: str = "",
        properties: Optional[dict[str, Any]] = None,
    ) -> "Entity":
        """
        Factory method to create a new entity.
        
        Args:
            name: Entity name.
            entity_type: Type from config.entity_types.
            domain: Domain from config.domains (uses default if not provided).
            description: Entity description.
            properties: Additional properties.
            
        Returns:
            New Entity instance.
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            entity_type=entity_type,
            domain=domain or config.domain_config.default_domain,
            description=description,
            properties=properties or {},
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "domain": self.domain,
            "description": self.description,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data["entity_type"],
            domain=data["domain"],
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )
    
    def embedding_text(self) -> str:
        """Text to embed for semantic search."""
        return f"{self.name}: {self.description}" if self.description else self.name


@dataclass
class Relationship:
    """
    An edge in the knowledge graph.
    
    Represents a typed, directed relationship between two entities.
    Relation types are strings loaded from config.
    """
    source_id: str
    target_id: str
    relation_type: str  # From config.relation_types
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SearchResult:
    """
    A result from associative memory search.
    
    Combines explicit (graph) and implicit (vector) matches.
    """
    entity: Entity
    score: float
    source: str  # "graph", "vector", or "merged"
    path: Optional[list[str]] = None  # For multi-hop graph results
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity.to_dict(),
            "score": self.score,
            "source": self.source,
            "path": self.path,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Associative Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class AssociativeMemory:
    """
    Hybrid knowledge graph with explicit (Kuzu) and implicit (FAISS) associations.
    
    Provides:
    - Entity and relationship CRUD
    - Multi-hop graph traversal
    - Semantic similarity search
    - Merged results for rich context
    
    Attributes:
        graph_dir: Directory for Kuzu graph database.
        index_path: Path to FAISS index file.
        embedder: Function to generate embeddings.
    """
    
    def __init__(
        self,
        graph_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
        embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
        embedding_dim: int = 1024,
    ):
        """
        Initialize Associative Memory.
        
        Args:
            graph_dir: Directory for Kuzu database.
            index_path: Path to FAISS index file.
            embedder: Async function to generate embeddings.
            embedding_dim: Dimension of embeddings.
        """
        self.graph_dir = graph_dir or config.paths.kuzu_dir
        self.index_path = index_path or config.paths.faiss_dir / "am.index"
        self.id_map_path = self.index_path.with_suffix(".idmap.json")
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        
        # Lazy initialization
        self._kuzu_db = None
        self._kuzu_conn = None
        self._faiss_index = None
        self._id_map: dict[int, str] = {}  # FAISS ID → Entity ID
        self._reverse_id_map: dict[str, int] = {}  # Entity ID → FAISS ID
        self._next_faiss_id = 0
        
        # In-memory entity cache
        self._entities: dict[str, Entity] = {}
        
        self._initialized = False
    
    # ── Initialization ──
    
    async def _init(self) -> None:
        """Initialize Kuzu and FAISS."""
        if self._initialized:
            return
        
        await self._init_kuzu()
        await self._init_faiss()
        self._initialized = True
    
    async def _init_kuzu(self) -> None:
        """Initialize Kuzu graph database."""
        import kuzu
        
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        
        self._kuzu_db = kuzu.Database(str(self.graph_dir))
        self._kuzu_conn = kuzu.Connection(self._kuzu_db)
        
        # Create schema if not exists
        try:
            self._kuzu_conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity (
                    id STRING PRIMARY KEY,
                    name STRING,
                    entity_type STRING,
                    domain STRING,
                    description STRING,
                    properties STRING,
                    created_at STRING
                )
            """)
        except Exception:
            pass  # Table already exists
        
        # Create relationship tables
        rel_types = [
            ("RELATED_TO", "relation STRING, weight DOUBLE"),
            ("CAUSED_BY", "confidence DOUBLE, source STRING"),
            ("CONTAINS", ""),
            ("ASKED_ABOUT", "count INT64, last_at STRING"),
            ("BELONGS_TO", ""),
            ("SIMILAR_TO", "score DOUBLE"),
        ]
        
        for rel_name, props in rel_types:
            try:
                prop_str = f", {props}" if props else ""
                self._kuzu_conn.execute(f"""
                    CREATE REL TABLE IF NOT EXISTS {rel_name} (
                        FROM Entity TO Entity{prop_str}, created_at STRING
                    )
                """)
            except Exception:
                pass  # Table already exists
    
    async def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        import faiss
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index_path.exists():
            # Load existing index
            self._faiss_index = faiss.read_index(str(self.index_path))
            
            # Load ID mapping
            if self.id_map_path.exists():
                data = json.loads(self.id_map_path.read_text())
                self._id_map = {int(k): v for k, v in data["id_map"].items()}
                self._reverse_id_map = {v: int(k) for k, v in data["id_map"].items()}
                self._next_faiss_id = data.get("next_id", len(self._id_map))
        else:
            # Create new index (Inner Product for cosine similarity with normalized vectors)
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
    
    async def _save_faiss(self) -> None:
        """Save FAISS index and ID mapping to disk."""
        import faiss
        
        faiss.write_index(self._faiss_index, str(self.index_path))
        
        data = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "next_id": self._next_faiss_id,
        }
        self.id_map_path.write_text(json.dumps(data, indent=2))
    
    # ── Entity Operations ──
    
    async def add_entity(self, entity: Entity) -> Entity:
        """
        Add an entity to the knowledge graph.
        
        Inserts into both Kuzu (explicit) and FAISS (implicit).
        
        Args:
            entity: The entity to add.
            
        Returns:
            The added entity.
        """
        await self._init()
        
        # Insert into Kuzu
        self._kuzu_conn.execute("""
            CREATE (e:Entity {
                id: $id,
                name: $name,
                entity_type: $entity_type,
                domain: $domain,
                description: $description,
                properties: $properties,
                created_at: $created_at
            })
        """, {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "domain": entity.domain.value,
            "description": entity.description,
            "properties": json.dumps(entity.properties),
            "created_at": entity.created_at.isoformat(),
        })
        
        # Embed and add to FAISS
        if self.embedder:
            embedding = await self.embedder(entity.embedding_text())
            embedding_np = np.array([embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding_np)
            
            self._faiss_index.add(embedding_np)
            self._id_map[self._next_faiss_id] = entity.id
            self._reverse_id_map[entity.id] = self._next_faiss_id
            self._next_faiss_id += 1
            
            await self._save_faiss()
        
        # Cache
        self._entities[entity.id] = entity
        
        return entity
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        await self._init()
        
        # Check cache
        if entity_id in self._entities:
            return self._entities[entity_id]
        
        # Query Kuzu
        result = self._kuzu_conn.execute(
            "MATCH (e:Entity {id: $id}) RETURN e",
            {"id": entity_id}
        )
        
        while result.has_next():
            row = result.get_next()
            node = row[0]
            entity = Entity(
                id=node["id"],
                name=node["name"],
                entity_type=node["entity_type"],
                domain=node["domain"],
                description=node["description"],
                properties=json.loads(node["properties"]) if node["properties"] else {},
                created_at=datetime.fromisoformat(node["created_at"]),
            )
            self._entities[entity_id] = entity
            return entity
        
        return None
    
    async def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 50,
    ) -> list[Entity]:
        """
        Find entities by criteria.
        
        Args:
            name: Filter by name (partial match).
            entity_type: Filter by type.
            domain: Filter by domain.
            limit: Maximum results.
            
        Returns:
            List of matching entities.
        """
        await self._init()
        
        conditions = []
        params = {}
        
        if name:
            conditions.append("e.name CONTAINS $name")
            params["name"] = name
        if entity_type:
            conditions.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type
        if domain:
            conditions.append("e.domain = $domain")
            params["domain"] = domain
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        query = f"MATCH (e:Entity) {where_clause} RETURN e LIMIT {limit}"
        result = self._kuzu_conn.execute(query, params)
        
        entities = []
        while result.has_next():
            row = result.get_next()
            node = row[0]
            entity = Entity(
                id=node["id"],
                name=node["name"],
                entity_type=node["entity_type"],
                domain=node["domain"],
                description=node["description"],
                properties=json.loads(node["properties"]) if node["properties"] else {},
                created_at=datetime.fromisoformat(node["created_at"]),
            )
            entities.append(entity)
        
        return entities
    
    # ── Relationship Operations ──
    
    async def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship between entities.
        
        Args:
            relationship: The relationship to add.
        """
        await self._init()
        
        rel_type = relationship.relation_type.value
        
        # Build property string based on relationship type
        if rel_type == "RELATED_TO":
            props = f"relation: '{relationship.properties.get('relation', '')}', weight: {relationship.weight}"
        elif rel_type == "CAUSED_BY":
            props = f"confidence: {relationship.properties.get('confidence', 1.0)}, source: '{relationship.properties.get('source', '')}'"
        elif rel_type == "ASKED_ABOUT":
            props = f"count: {relationship.properties.get('count', 1)}, last_at: '{datetime.now(timezone.utc).isoformat()}'"
        elif rel_type == "SIMILAR_TO":
            props = f"score: {relationship.weight}"
        else:
            props = ""
        
        prop_str = f", {{{props}, created_at: '{relationship.created_at.isoformat()}'}}" if props else f", {{created_at: '{relationship.created_at.isoformat()}'}}"
        
        query = f"""
            MATCH (a:Entity {{id: $source_id}}), (b:Entity {{id: $target_id}})
            CREATE (a)-[:{rel_type}{prop_str}]->(b)
        """
        
        self._kuzu_conn.execute(query, {
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
        })
    
    async def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing",  # "outgoing", "incoming", "both"
        limit: int = 50,
    ) -> list[tuple[Entity, Relationship]]:
        """
        Get neighboring entities connected by relationships.
        
        Args:
            entity_id: Source entity ID.
            relation_type: Filter by relationship type.
            direction: Relationship direction.
            limit: Maximum results.
            
        Returns:
            List of (entity, relationship) tuples.
        """
        await self._init()
        
        if direction == "outgoing":
            pattern = "(a)-[r]->(b)"
        elif direction == "incoming":
            pattern = "(a)<-[r]-(b)"
        else:
            pattern = "(a)-[r]-(b)"
        
        rel_filter = f":{relation_type}" if relation_type else ""
        pattern = pattern.replace("[r]", f"[r{rel_filter}]")
        
        query = f"""
            MATCH {pattern}
            WHERE a.id = $entity_id
            RETURN b, type(r) as rel_type
            LIMIT {limit}
        """
        
        result = self._kuzu_conn.execute(query, {"entity_id": entity_id})
        
        neighbors = []
        while result.has_next():
            row = result.get_next()
            node = row[0]
            rel_type_str = row[1]
            
            entity = Entity(
                id=node["id"],
                name=node["name"],
                entity_type=node["entity_type"],
                domain=node["domain"],
                description=node["description"],
                properties=json.loads(node["properties"]) if node["properties"] else {},
                created_at=datetime.fromisoformat(node["created_at"]),
            )
            
            rel = Relationship(
                source_id=entity_id,
                target_id=entity.id,
                relation_type=rel_type_str,
            )
            
            neighbors.append((entity, rel))
        
        return neighbors
    
    # ── Search Operations ──
    
    async def search_semantic(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[SearchResult]:
        """
        Search for semantically similar entities using FAISS.
        
        Args:
            query: Search query text.
            top_k: Number of results.
            threshold: Minimum similarity score.
            
        Returns:
            List of SearchResult with vector source.
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
        scores, indices = self._faiss_index.search(embedding_np, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            
            entity_id = self._id_map.get(int(idx))
            if not entity_id:
                continue
            
            entity = await self.get_entity(entity_id)
            if entity:
                results.append(SearchResult(
                    entity=entity,
                    score=float(score),
                    source="vector",
                ))
        
        return results
    
    async def search_graph(
        self,
        start_entity_id: str,
        max_hops: int = 2,
        relation_types: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Multi-hop graph traversal from a starting entity.
        
        Args:
            start_entity_id: Starting entity ID.
            max_hops: Maximum traversal depth.
            relation_types: Filter by relationship types.
            
        Returns:
            List of SearchResult with graph source and paths.
        """
        await self._init()
        
        rel_filter = ""
        if relation_types:
            rel_names = "|".join(relation_types)
            rel_filter = f":{rel_names}"
        
        query = f"""
            MATCH path = (start:Entity {{id: $start_id}})-[r{rel_filter}*1..{max_hops}]->(end:Entity)
            RETURN end, length(path) as hops, [n in nodes(path) | n.name] as path_names
            ORDER BY hops
            LIMIT 50
        """
        
        result = self._kuzu_conn.execute(query, {"start_id": start_entity_id})
        
        results = []
        seen = set()
        
        while result.has_next():
            row = result.get_next()
            node = row[0]
            hops = row[1]
            path_names = row[2]
            
            if node["id"] in seen:
                continue
            seen.add(node["id"])
            
            entity = Entity(
                id=node["id"],
                name=node["name"],
                entity_type=node["entity_type"],
                domain=node["domain"],
                description=node["description"],
                properties=json.loads(node["properties"]) if node["properties"] else {},
                created_at=datetime.fromisoformat(node["created_at"]),
            )
            
            # Score inversely proportional to hops
            score = 1.0 / (1 + hops)
            
            results.append(SearchResult(
                entity=entity,
                score=score,
                source="graph",
                path=path_names,
            ))
        
        return results
    
    async def search(
        self,
        query: str,
        start_entity_id: Optional[str] = None,
        top_k: int = 10,
        include_graph: bool = True,
        include_vector: bool = True,
    ) -> list[SearchResult]:
        """
        Hybrid search combining graph traversal and semantic similarity.
        
        Args:
            query: Search query text.
            start_entity_id: Optional starting entity for graph traversal.
            top_k: Number of results per source.
            include_graph: Include graph traversal results.
            include_vector: Include vector similarity results.
            
        Returns:
            Merged and ranked SearchResults.
        """
        results = []
        
        # Vector search
        if include_vector:
            vector_results = await self.search_semantic(query, top_k)
            results.extend(vector_results)
        
        # Graph search (if starting point provided or found via vector)
        if include_graph:
            start_ids = []
            
            if start_entity_id:
                start_ids.append(start_entity_id)
            elif vector_results:
                # Use top vector result as starting point
                start_ids.append(vector_results[0].entity.id)
            
            for start_id in start_ids[:3]:  # Limit starting points
                graph_results = await self.search_graph(start_id, max_hops=2)
                results.extend(graph_results)
        
        # Deduplicate and merge scores
        merged: dict[str, SearchResult] = {}
        for r in results:
            eid = r.entity.id
            if eid in merged:
                # Combine scores (RRF-style)
                existing = merged[eid]
                existing.score = existing.score + r.score
                existing.source = "merged"
                if r.path and not existing.path:
                    existing.path = r.path
            else:
                merged[eid] = r
        
        # Sort by score descending
        final_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        
        return final_results[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_am_instance: Optional[AssociativeMemory] = None


def get_associative_memory() -> AssociativeMemory:
    """
    Get the singleton AssociativeMemory instance.
    
    Returns:
        The AssociativeMemory instance.
    """
    global _am_instance
    if _am_instance is None:
        _am_instance = AssociativeMemory()
    return _am_instance


# ── Convenience Functions ──

async def add_entity(
    name: str,
    entity_type: str,
    domain: Optional[str] = None,
    description: str = "",
) -> Entity:
    """Add an entity to the knowledge graph."""
    am = get_associative_memory()
    entity = Entity.create(name, entity_type, domain, description)
    return await am.add_entity(entity)


async def link_entities(
    source_id: str,
    target_id: str,
    relation_type: str,
    weight: float = 1.0,
) -> None:
    """Create a relationship between entities."""
    am = get_associative_memory()
    rel = Relationship(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        weight=weight,
    )
    await am.add_relationship(rel)


async def search_knowledge(query: str, top_k: int = 10) -> list[SearchResult]:
    """Search the knowledge graph."""
    return await get_associative_memory().search(query, top_k=top_k)

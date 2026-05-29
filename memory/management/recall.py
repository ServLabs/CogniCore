"""
Memory Recall Engine

Tiered search across all memory types with:
- Reciprocal Rank Fusion (RRF) for multi-signal fusion
- Maximal Marginal Relevance (MMR) for redundancy removal
- Confidence scoring and ranking
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import numpy as np

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RecallConfig:
    """Configuration for recall engine."""
    # Search limits
    top_k_per_source: int = 10
    final_top_k: int = 20
    
    # Confidence thresholds
    high_confidence: float = 0.7
    low_confidence: float = 0.3
    
    # RRF parameters
    rrf_k: int = 60  # Smoothing constant
    
    # MMR parameters
    mmr_lambda: float = 0.7  # Balance relevance vs diversity
    
    # Signal weights for RRF
    weights: dict[str, float] = field(default_factory=lambda: {
        "similarity": 1.0,
        "recency": 0.3,
        "frequency": 0.2,
        "source_priority": 0.5,
    })
    
    # Source priorities (higher = more trusted for factual queries)
    source_priorities: dict[str, int] = field(default_factory=lambda: {
        "sfm": 5,   # Compressed facts - highest for quick answers
        "lfm": 4,   # Detailed documents
        "am": 3,    # Associations
        "mm": 2,    # Procedures
        "wm": 1,    # Current conversation
    })


@dataclass
class RecallItem:
    """A single item from recall."""
    id: str
    content: str
    source: str  # Memory type: sfm, lfm, am, mm, wm
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    
    # Ranking signals
    similarity: float = 0.0
    recency_score: float = 0.0
    frequency_score: float = 0.0
    source_priority: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class RecallResult:
    """Result of memory recall."""
    items: list[RecallItem]
    query: str
    total_searched: int = 0
    search_time_ms: float = 0.0
    sources_searched: list[str] = field(default_factory=list)
    
    @property
    def has_results(self) -> bool:
        return len(self.items) > 0
    
    @property
    def top_item(self) -> Optional[RecallItem]:
        return self.items[0] if self.items else None
    
    @property
    def high_confidence_items(self) -> list[RecallItem]:
        return [i for i in self.items if i.score >= 0.7]
    
    def to_context(self, max_items: int = 10) -> str:
        """Convert to context string for LLM."""
        lines = []
        for item in self.items[:max_items]:
            source_label = item.source.upper()
            lines.append(f"[{source_label}] {item.content}")
        return "\n\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "items": [i.to_dict() for i in self.items],
            "total_searched": self.total_searched,
            "search_time_ms": self.search_time_ms,
            "sources_searched": self.sources_searched,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Recall Engine
# ══════════════════════════════════════════════════════════════════════════════

class RecallEngine:
    """
    Memory recall engine with tiered search and ranking.
    
    Provides:
    - Tiered search: Meta → SFM → LFM → AM → MM
    - Multi-signal ranking with RRF
    - Redundancy removal with MMR
    - Confidence-based routing
    
    Attributes:
        config: Recall configuration.
        embedder: Function to generate embeddings.
    """
    
    def __init__(
        self,
        recall_config: Optional[RecallConfig] = None,
        embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
    ):
        """
        Initialize Recall Engine.
        
        Args:
            recall_config: Recall configuration.
            embedder: Async function to generate embeddings.
        """
        self.config = recall_config or RecallConfig()
        self.embedder = embedder
    
    async def recall(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        include_wm: bool = True,
        user_id: Optional[str] = None,
        convo_id: Optional[str] = None,
    ) -> RecallResult:
        """
        Recall relevant memories for a query.
        
        Args:
            query: User query.
            sources: Specific sources to search (None = all).
            top_k: Number of results (None = config default).
            include_wm: Include working memory context.
            user_id: User ID for WM lookup.
            convo_id: Conversation ID for WM lookup.
            
        Returns:
            RecallResult with ranked items.
        """
        import time
        start_time = time.perf_counter()
        
        top_k = top_k or self.config.final_top_k
        sources = sources or ["meta", "sfm", "lfm", "am", "mm"]
        
        all_items: list[RecallItem] = []
        sources_searched = []
        
        # 1. Check Meta Memory for routing
        if "meta" in sources:
            from memory.types.meta import get_meta_memory
            
            meta = get_meta_memory()
            route_result = await meta.route_query(query)
            
            if route_result.has_match and route_result.pointer:
                # High confidence - route directly to target memory
                target_source = route_result.pointer.memory_type.value
                sources = [target_source]
                sources_searched.append("meta")
        
        # 2. Search each source
        if "sfm" in sources:
            sfm_items = await self._search_sfm(query)
            all_items.extend(sfm_items)
            sources_searched.append("sfm")
        
        if "lfm" in sources:
            lfm_items = await self._search_lfm(query)
            all_items.extend(lfm_items)
            sources_searched.append("lfm")
        
        if "am" in sources:
            am_items = await self._search_am(query)
            all_items.extend(am_items)
            sources_searched.append("am")
        
        if "mm" in sources:
            mm_items = await self._search_mm(query)
            all_items.extend(mm_items)
            sources_searched.append("mm")
        
        if include_wm and "wm" in sources and user_id and convo_id:
            wm_items = await self._search_wm(query, user_id, convo_id)
            all_items.extend(wm_items)
            sources_searched.append("wm")
        
        # 3. Rank with RRF
        ranked_items = self._rank_rrf(all_items)
        
        # 4. Apply MMR for diversity
        diverse_items = self._apply_mmr(ranked_items, top_k)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return RecallResult(
            items=diverse_items,
            query=query,
            total_searched=len(all_items),
            search_time_ms=elapsed_ms,
            sources_searched=sources_searched,
        )
    
    # ── Source-Specific Search ──
    
    async def _search_sfm(self, query: str) -> list[RecallItem]:
        """Search Short Form Memory."""
        from memory.types.sfm import get_short_form_memory
        
        sfm = get_short_form_memory()
        results = await sfm.search(query, top_k=self.config.top_k_per_source)
        
        items = []
        for r in results:
            items.append(RecallItem(
                id=r.fact.id,
                content=r.fact.fact,
                source="sfm",
                score=r.score,
                similarity=r.score,
                frequency_score=min(1.0, r.fact.access_count / 100),
                recency_score=self._recency_score(r.fact.last_accessed),
                source_priority=self.config.source_priorities["sfm"],
                metadata={
                    "domain": r.fact.domain,
                    "tags": r.fact.tags,
                    "pinned": r.fact.pinned,
                },
            ))
        
        return items
    
    async def _search_lfm(self, query: str) -> list[RecallItem]:
        """Search Long Form Memory."""
        from memory.types.lfm import get_long_form_memory
        
        lfm = get_long_form_memory()
        results = await lfm.search(query, top_k=self.config.top_k_per_source)
        
        items = []
        for r in results:
            items.append(RecallItem(
                id=r.chunk.id,
                content=r.chunk.content,
                source="lfm",
                score=r.score,
                similarity=r.score,
                frequency_score=min(1.0, r.chunk.access_count / 50),
                source_priority=self.config.source_priorities["lfm"],
                metadata={
                    "document_id": r.document.id,
                    "document_title": r.document.title,
                    "domain": r.document.domain,
                    "chunk_index": r.chunk.chunk_index,
                },
            ))
        
        return items
    
    async def _search_am(self, query: str) -> list[RecallItem]:
        """Search Associative Memory."""
        from memory.types.am import get_associative_memory
        
        am = get_associative_memory()
        results = await am.search_graph(query, top_k=self.config.top_k_per_source)
        
        items = []
        for r in results:
            content = f"{r.entity.name}: {r.entity.description}" if r.entity.description else r.entity.name
            items.append(RecallItem(
                id=r.entity.id,
                content=content,
                source="am",
                score=r.score,
                similarity=r.score,
                source_priority=self.config.source_priorities["am"],
                metadata={
                    "entity_type": r.entity.entity_type,
                    "domain": r.entity.domain,
                    "properties": r.entity.properties,
                },
            ))
        
        return items
    
    async def _search_mm(self, query: str) -> list[RecallItem]:
        """Search Motor Memory (procedures)."""
        from memory.types.mm import get_motor_memory
        
        mm = get_motor_memory()
        results = await mm.find_by_intent(query, top_k=self.config.top_k_per_source)
        
        items = []
        for proc in results:
            content = f"Procedure: {proc.name}\n{proc.description}"
            if proc.steps_nl:
                content += "\nSteps:\n" + "\n".join(f"- {s}" for s in proc.steps_nl[:3])
            
            items.append(RecallItem(
                id=proc.id,
                content=content,
                source="mm",
                score=0.5,  # Procedures matched by intent
                similarity=0.5,
                frequency_score=min(1.0, proc.usage_count / 20),
                source_priority=self.config.source_priorities["mm"],
                metadata={
                    "name": proc.name,
                    "domain": proc.domain,
                    "has_executable": proc.has_executable,
                    "success_rate": proc.success_rate,
                },
            ))
        
        return items
    
    async def _search_wm(
        self,
        query: str,
        user_id: str,
        convo_id: str,
    ) -> list[RecallItem]:
        """Search Working Memory (current conversation)."""
        from memory.types.wm import get_working_memory
        
        wm = get_working_memory()
        context = await wm.get_context(user_id, convo_id)
        
        if not context:
            return []
        
        # Include recent messages and summary
        items = []
        
        if context.get("summary"):
            items.append(RecallItem(
                id=f"wm_summary_{convo_id}",
                content=f"Conversation summary: {context['summary']}",
                source="wm",
                score=0.8,  # High relevance for current context
                similarity=0.8,
                recency_score=1.0,
                source_priority=self.config.source_priorities["wm"],
                metadata={"type": "summary"},
            ))
        
        return items
    
    # ── Ranking Algorithms ──
    
    def _rank_rrf(self, items: list[RecallItem]) -> list[RecallItem]:
        """
        Rank items using Reciprocal Rank Fusion.
        
        RRF formula: score(doc) = Σ w_i / (k + rank_i(doc))
        """
        if not items:
            return []
        
        k = self.config.rrf_k
        weights = self.config.weights
        
        # Create ranked lists for each signal
        by_similarity = sorted(items, key=lambda x: x.similarity, reverse=True)
        by_recency = sorted(items, key=lambda x: x.recency_score, reverse=True)
        by_frequency = sorted(items, key=lambda x: x.frequency_score, reverse=True)
        by_priority = sorted(items, key=lambda x: x.source_priority, reverse=True)
        
        # Build rank maps
        def build_rank_map(sorted_items: list[RecallItem]) -> dict[str, int]:
            return {item.id: rank for rank, item in enumerate(sorted_items)}
        
        sim_ranks = build_rank_map(by_similarity)
        rec_ranks = build_rank_map(by_recency)
        freq_ranks = build_rank_map(by_frequency)
        prio_ranks = build_rank_map(by_priority)
        
        # Calculate RRF scores
        for item in items:
            rrf_score = 0.0
            rrf_score += weights["similarity"] / (k + sim_ranks.get(item.id, len(items)))
            rrf_score += weights["recency"] / (k + rec_ranks.get(item.id, len(items)))
            rrf_score += weights["frequency"] / (k + freq_ranks.get(item.id, len(items)))
            rrf_score += weights["source_priority"] / (k + prio_ranks.get(item.id, len(items)))
            item.score = rrf_score
        
        # Sort by final RRF score
        return sorted(items, key=lambda x: x.score, reverse=True)
    
    def _apply_mmr(
        self,
        items: list[RecallItem],
        top_k: int,
    ) -> list[RecallItem]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        MMR = argmax[λ * sim(q,d) - (1-λ) * max(sim(d,d'))]
        """
        if len(items) <= top_k:
            return items
        
        if not self.embedder:
            # No embeddings - just return top k
            return items[:top_k]
        
        # For now, simple diversity by source
        # TODO: Implement full MMR with embeddings
        selected: list[RecallItem] = []
        source_counts: dict[str, int] = {}
        max_per_source = max(2, top_k // 3)
        
        for item in items:
            source = item.source
            if source_counts.get(source, 0) < max_per_source:
                selected.append(item)
                source_counts[source] = source_counts.get(source, 0) + 1
            
            if len(selected) >= top_k:
                break
        
        return selected
    
    def _recency_score(self, last_accessed: Optional[datetime]) -> float:
        """Calculate recency score (0-1) based on last access time."""
        if not last_accessed:
            return 0.0
        
        now = datetime.now(timezone.utc)
        age_hours = (now - last_accessed).total_seconds() / 3600
        
        # Exponential decay: score = e^(-age/decay_rate)
        decay_rate = 168  # 1 week half-life
        return math.exp(-age_hours / decay_rate)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_recall_instance: Optional[RecallEngine] = None


def get_recall_engine() -> RecallEngine:
    """Get the singleton RecallEngine instance."""
    global _recall_instance
    if _recall_instance is None:
        _recall_instance = RecallEngine()
    return _recall_instance

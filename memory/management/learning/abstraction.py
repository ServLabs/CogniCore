"""
Abstraction Learning (Learning Type 3)

Move from specific instances to general concepts.
Build a hierarchy of understanding.

Levels:
- Level 0 (specific): "User spent $150 on dining on 2026-03-15"
- Level 1 (pattern): "Dining expenses spike on weekends"
- Level 2 (abstract): "Discretionary spending increases during leisure time"

Algorithm: Hierarchical clustering + multi-level summarization
Trigger: Sleep mode
LLM: Yes (cheap)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import numpy as np

from core import config, log
from audit import audit


@dataclass
class AbstractionLevel:
    """A level in the abstraction hierarchy."""
    level: int  # 0 = specific, 1 = pattern, 2 = abstract
    content: str
    source_ids: list[str]  # IDs of lower-level items
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "content": self.content,
            "source_ids": self.source_ids,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AbstractionResult:
    """Result of abstraction learning."""
    abstractions: list[AbstractionLevel]
    facts_processed: int
    clusters_found: int
    llm_calls: int
    duration_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "abstractions": [a.to_dict() for a in self.abstractions],
            "facts_processed": self.facts_processed,
            "clusters_found": self.clusters_found,
            "llm_calls": self.llm_calls,
            "duration_ms": self.duration_ms,
        }


def agglomerative_cluster(
    embeddings: np.ndarray,
    distance_threshold: float = 0.5,
) -> np.ndarray:
    """
    Hierarchical clustering using agglomerative approach.
    
    Args:
        embeddings: N x D embedding matrix.
        distance_threshold: Distance threshold for clustering.
        
    Returns:
        Cluster labels.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average',
            metric='cosine',
        )
        return clustering.fit_predict(embeddings)
    except ImportError:
        # Fallback: simple single-linkage clustering
        n = len(embeddings)
        labels = list(range(n))  # Each point starts as its own cluster
        
        # Compute pairwise distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = 1 - np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                distances.append((dist, i, j))
        
        distances.sort()
        
        # Merge clusters
        for dist, i, j in distances:
            if dist > distance_threshold:
                break
            # Merge clusters
            old_label = labels[j]
            new_label = labels[i]
            for k in range(n):
                if labels[k] == old_label:
                    labels[k] = new_label
        
        # Renumber labels
        unique = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique)}
        return np.array([label_map[l] for l in labels])


class AbstractionLearner:
    """
    Learns abstractions from specific facts.
    
    Builds a hierarchy: specific → pattern → abstract concept.
    
    Attributes:
        llm: LLM callable for summarization.
        embedder: Embedding function.
    """
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        embedder: Optional[Callable[[str], Awaitable[np.ndarray]]] = None,
        min_cluster_size: int = 3,
        distance_threshold: float = 0.5,
    ):
        """
        Initialize abstraction learner.
        
        Args:
            llm: LLM callable for summarization.
            embedder: Embedding function.
            min_cluster_size: Minimum facts to form a cluster.
            distance_threshold: Distance threshold for clustering.
        """
        self.llm = llm
        self.embedder = embedder
        self.min_cluster_size = min_cluster_size
        self.distance_threshold = distance_threshold
    
    async def abstract_from_facts(
        self,
        facts: list[dict[str, Any]],
        target_level: int = 1,
    ) -> list[AbstractionLevel]:
        """
        Create abstractions from specific facts.
        
        Args:
            facts: List of fact dicts with 'id', 'content', 'embedding'.
            target_level: Target abstraction level (1 or 2).
            
        Returns:
            List of abstractions.
        """
        import time
        start = time.monotonic()
        
        if len(facts) < self.min_cluster_size:
            return []
        
        # Ensure embeddings
        embeddings = []
        for fact in facts:
            if fact.get("embedding") is not None:
                embeddings.append(fact["embedding"])
            elif self.embedder:
                emb = await self.embedder(fact["content"])
                fact["embedding"] = emb
                embeddings.append(emb)
            else:
                return []  # Can't cluster without embeddings
        
        embeddings = np.array(embeddings)
        
        # Cluster facts
        labels = agglomerative_cluster(embeddings, self.distance_threshold)
        
        # Group by cluster
        clusters: dict[int, list[dict]] = {}
        for fact, label in zip(facts, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(fact)
        
        # Filter small clusters
        valid_clusters = {
            k: v for k, v in clusters.items()
            if len(v) >= self.min_cluster_size
        }
        
        # Summarize each cluster
        abstractions = []
        llm_calls = 0
        
        for cluster_id, cluster_facts in valid_clusters.items():
            if self.llm:
                summary = await self._summarize_cluster(cluster_facts, target_level)
                llm_calls += 1
            else:
                # Fallback: simple concatenation
                summary = f"Pattern from {len(cluster_facts)} observations"
            
            abstraction = AbstractionLevel(
                level=target_level,
                content=summary,
                source_ids=[f["id"] for f in cluster_facts],
            )
            
            # Embed the abstraction
            if self.embedder:
                abstraction.embedding = await self.embedder(summary)
            
            abstractions.append(abstraction)
        
        duration_ms = (time.monotonic() - start) * 1000
        
        audit.log_raw(
            "learning",
            "abstraction",
            "mml",
            "completed",
            facts_processed=len(facts),
            clusters_found=len(valid_clusters),
            abstractions_created=len(abstractions),
        )
        
        return abstractions
    
    async def _summarize_cluster(
        self,
        facts: list[dict[str, Any]],
        level: int,
    ) -> str:
        """Summarize a cluster of facts into an abstraction."""
        fact_list = "\n".join(f"- {f['content']}" for f in facts[:10])  # Limit
        
        if level == 1:
            prompt = f"""
Summarize these specific observations into one general pattern:

{fact_list}

Write a single sentence that captures the common pattern.
"""
        else:  # level 2
            prompt = f"""
These are patterns observed in the data:

{fact_list}

Synthesize these into one abstract principle or concept.
Write a single sentence at a high level of abstraction.
"""
        
        return await self.llm(prompt)
    
    async def build_hierarchy(
        self,
        level0_facts: list[dict[str, Any]],
    ) -> dict[int, list[AbstractionLevel]]:
        """
        Build full abstraction hierarchy.
        
        Args:
            level0_facts: Specific facts (level 0).
            
        Returns:
            Dict mapping level -> abstractions.
        """
        hierarchy: dict[int, list[AbstractionLevel]] = {0: []}
        
        # Level 0: wrap raw facts
        for fact in level0_facts:
            hierarchy[0].append(AbstractionLevel(
                level=0,
                content=fact["content"],
                source_ids=[fact["id"]],
                embedding=fact.get("embedding"),
            ))
        
        # Level 1: patterns from specific facts
        level1 = await self.abstract_from_facts(level0_facts, target_level=1)
        hierarchy[1] = level1
        
        # Level 2: abstract concepts from patterns
        if len(level1) >= self.min_cluster_size:
            level1_as_facts = [
                {"id": f"l1_{i}", "content": a.content, "embedding": a.embedding}
                for i, a in enumerate(level1)
            ]
            level2 = await self.abstract_from_facts(level1_as_facts, target_level=2)
            hierarchy[2] = level2
        else:
            hierarchy[2] = []
        
        return hierarchy


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_abstraction_learner: Optional[AbstractionLearner] = None


def get_abstraction_learner() -> AbstractionLearner:
    """Get the singleton AbstractionLearner instance."""
    global _abstraction_learner
    if _abstraction_learner is None:
        _abstraction_learner = AbstractionLearner()
    return _abstraction_learner

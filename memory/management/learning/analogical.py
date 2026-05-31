"""
Analogical Learning (Learning Type 4)

"This is like that" — map structure from a known domain to a new domain.
Bootstrap understanding of unfamiliar areas.

Algorithm: Cross-domain embedding alignment + structure mapping
Trigger: Sleep mode (when new domain is ingested)
LLM: Yes (expensive - analogy verification requires reasoning)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import numpy as np

from core import config, log
from audit import audit


@dataclass
class Analogy:
    """A detected analogy between domains."""
    source_entity: str
    source_domain: str
    target_entity: str
    target_domain: str
    similarity: float
    explanation: str
    verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_entity": self.source_entity,
            "source_domain": self.source_domain,
            "target_entity": self.target_entity,
            "target_domain": self.target_domain,
            "similarity": self.similarity,
            "explanation": self.explanation,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AnalogicalResult:
    """Result of analogical learning."""
    analogies_found: int
    analogies_verified: int
    domains_compared: list[str]
    llm_calls: int
    duration_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analogies_found": self.analogies_found,
            "analogies_verified": self.analogies_verified,
            "domains_compared": self.domains_compared,
            "llm_calls": self.llm_calls,
            "duration_ms": self.duration_ms,
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class AnalogicalLearner:
    """
    Learns analogies between domains.
    
    Finds structural similarities between concepts in different domains
    and creates cross-domain mappings.
    
    Attributes:
        llm: LLM callable for analogy verification.
        embedder: Embedding function.
        graph_query: Function to query AM graph.
    """
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        embedder: Optional[Callable[[str], Awaitable[np.ndarray]]] = None,
        graph_query: Optional[Callable[[str], Awaitable[list[dict]]]] = None,
        similarity_threshold: float = 0.6,
    ):
        """
        Initialize analogical learner.
        
        Args:
            llm: LLM callable for verification.
            embedder: Embedding function.
            graph_query: AM graph query function.
            similarity_threshold: Minimum similarity for analogy.
        """
        self.llm = llm
        self.embedder = embedder
        self.graph_query = graph_query
        self.similarity_threshold = similarity_threshold
    
    async def find_analogies(
        self,
        source_domain: str,
        target_domain: str,
        source_entities: Optional[list[dict[str, Any]]] = None,
        target_entities: Optional[list[dict[str, Any]]] = None,
    ) -> list[Analogy]:
        """
        Find analogies between two domains.
        
        Args:
            source_domain: Known domain.
            target_domain: New domain.
            source_entities: Entities from source domain.
            target_entities: Entities from target domain.
            
        Returns:
            List of detected analogies.
        """
        import time
        start = time.monotonic()
        
        # Get entities from graph if not provided
        if source_entities is None and self.graph_query:
            source_entities = await self.graph_query(
                f"MATCH (n:Entity) WHERE n.domain = '{source_domain}' RETURN n"
            )
        if target_entities is None and self.graph_query:
            target_entities = await self.graph_query(
                f"MATCH (n:Entity) WHERE n.domain = '{target_domain}' RETURN n"
            )
        
        if not source_entities or not target_entities:
            return []
        
        # Ensure embeddings
        await self._ensure_embeddings(source_entities)
        await self._ensure_embeddings(target_entities)
        
        # Find similar pairs
        candidates = []
        for s in source_entities:
            if s.get("embedding") is None:
                continue
            for t in target_entities:
                if t.get("embedding") is None:
                    continue
                if s.get("name") == t.get("name"):
                    continue  # Skip same-name entities
                
                sim = cosine_similarity(s["embedding"], t["embedding"])
                if sim >= self.similarity_threshold:
                    candidates.append(Analogy(
                        source_entity=s.get("name", ""),
                        source_domain=source_domain,
                        target_entity=t.get("name", ""),
                        target_domain=target_domain,
                        similarity=sim,
                        explanation="",
                    ))
        
        # Sort by similarity
        candidates.sort(key=lambda a: a.similarity, reverse=True)
        
        # Verify top candidates with LLM
        verified = []
        llm_calls = 0
        
        for analogy in candidates[:10]:  # Limit verification
            if self.llm:
                is_valid, explanation = await self._verify_analogy(analogy)
                llm_calls += 1
                
                if is_valid:
                    analogy.verified = True
                    analogy.explanation = explanation
                    verified.append(analogy)
            else:
                # Without LLM, accept high-similarity candidates
                if analogy.similarity >= 0.8:
                    analogy.explanation = f"High embedding similarity ({analogy.similarity:.2f})"
                    verified.append(analogy)
        
        duration_ms = (time.monotonic() - start) * 1000
        
        audit.log_raw(
            "learning",
            "analogical",
            "mml",
            "completed",
            source_domain=source_domain,
            target_domain=target_domain,
            candidates=len(candidates),
            verified=len(verified),
        )
        
        return verified
    
    async def _ensure_embeddings(self, entities: list[dict[str, Any]]) -> None:
        """Ensure all entities have embeddings."""
        if not self.embedder:
            return
        
        for entity in entities:
            if entity.get("embedding") is None:
                text = entity.get("description") or entity.get("name", "")
                if text:
                    entity["embedding"] = await self.embedder(text)
    
    async def _verify_analogy(self, analogy: Analogy) -> tuple[bool, str]:
        """Verify an analogy using LLM."""
        prompt = f"""
Is '{analogy.source_entity}' in the {analogy.source_domain} domain 
analogous to '{analogy.target_entity}' in the {analogy.target_domain} domain?

Consider:
- Do they serve similar functions in their respective domains?
- Do they have similar relationships to other concepts?
- Can knowledge about one inform understanding of the other?

Answer with YES or NO, followed by a brief explanation.
"""
        
        response = await self.llm(prompt)
        response_lower = response.lower()
        
        is_valid = response_lower.startswith("yes") or "yes," in response_lower[:20]
        
        # Extract explanation
        explanation = response
        if ":" in response:
            explanation = response.split(":", 1)[1].strip()
        elif "." in response:
            explanation = response.split(".", 1)[1].strip()
        
        return is_valid, explanation[:200]  # Limit length
    
    async def transfer_knowledge(
        self,
        analogy: Analogy,
        source_facts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Transfer knowledge from source to target using analogy.
        
        Args:
            analogy: Verified analogy.
            source_facts: Facts about source entity.
            
        Returns:
            Transferred facts for target entity.
        """
        if not self.llm:
            return []
        
        fact_list = "\n".join(f"- {f['content']}" for f in source_facts[:10])
        
        prompt = f"""
Given this analogy:
- '{analogy.source_entity}' ({analogy.source_domain}) is like 
  '{analogy.target_entity}' ({analogy.target_domain})
- Explanation: {analogy.explanation}

These facts are known about '{analogy.source_entity}':
{fact_list}

What analogous facts might apply to '{analogy.target_entity}'?
List 3-5 transferred insights, adapting the concepts appropriately.
"""
        
        response = await self.llm(prompt)
        
        # Parse response into facts
        transferred = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                # Clean up the line
                content = line.lstrip("-•0123456789.) ").strip()
                if content:
                    transferred.append({
                        "content": content,
                        "source": f"analogical_transfer:{analogy.source_entity}",
                        "confidence": analogy.similarity * 0.8,  # Reduce confidence
                    })
        
        return transferred


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_analogical_learner: Optional[AnalogicalLearner] = None


def get_analogical_learner() -> AnalogicalLearner:
    """Get the singleton AnalogicalLearner instance."""
    global _analogical_learner
    if _analogical_learner is None:
        _analogical_learner = AnalogicalLearner()
    return _analogical_learner

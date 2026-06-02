"""
Contrastive Learning (Learning Type 10)

Learn what something IS by understanding what it ISN'T.
Store positive-negative fact pairs to reduce hallucination and confusion.

Algorithm: Positive-negative pairing + NOT_SAME_AS edges
Trigger: Real-time (from corrections) + Sleep mode (LLM-generated negatives)
LLM: Yes (cheap - generating "what it's not" counterparts)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from core import config, log
from core import audit


@dataclass
class ContrastivePair:
    """A positive-negative fact pair."""
    pair_id: str
    positive: str  # What it IS
    negative: str  # What it is NOT
    explanation: str
    source: str  # "correction", "confusion", "admin", "generated"
    entity_id: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "positive": self.positive,
            "negative": self.negative,
            "explanation": self.explanation,
            "source": self.source,
            "entity_id": self.entity_id,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ConfusionEvent:
    """Record of a confusion between two concepts."""
    entity_a: str
    entity_b: str
    context: str  # Query or situation where confusion occurred
    count: int = 1
    last_occurred: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "context": self.context,
            "count": self.count,
            "last_occurred": self.last_occurred.isoformat(),
        }


class ContrastiveLearner:
    """
    Learns contrastive pairs to prevent confusion and hallucination.
    
    Sources of contrastive pairs:
    1. Corrective learning: Every user correction generates a pair
    2. Confusion detection: When agent retrieves A but user wanted B
    3. Admin-curated: Domain experts define common mistakes
    4. Sleep mode generation: LLM generates "what this is NOT"
    
    Attributes:
        pairs: List of contrastive pairs.
        confusions: Tracked confusion events.
        llm: LLM callable for generating negatives.
    """
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        graph_add_edge: Optional[Callable[[str, str, str, dict], Awaitable[bool]]] = None,
    ):
        """
        Initialize contrastive learner.
        
        Args:
            llm: LLM callable for generating negatives.
            graph_add_edge: Function to add edge to AM graph.
        """
        self.llm = llm
        self.graph_add_edge = graph_add_edge
        self.pairs: list[ContrastivePair] = []
        self.confusions: dict[tuple[str, str], ConfusionEvent] = {}
    
    def add_from_correction(
        self,
        wrong_statement: str,
        correct_statement: str,
        entity_id: Optional[str] = None,
    ) -> ContrastivePair:
        """
        Add contrastive pair from a correction.
        
        Args:
            wrong_statement: What the agent incorrectly said.
            correct_statement: The correct information.
            entity_id: Optional entity this relates to.
            
        Returns:
            Created ContrastivePair.
        """
        import uuid
        
        pair = ContrastivePair(
            pair_id=str(uuid.uuid4()),
            positive=correct_statement,
            negative=f"NOT: {wrong_statement}",
            explanation=f"Corrected: {wrong_statement} → {correct_statement}",
            source="correction",
            entity_id=entity_id,
            confidence=1.0,  # High confidence from explicit correction
        )
        
        self.pairs.append(pair)
        
        audit.log_raw(
            "learning",
            "contrastive_correction",
            "mml",
            "completed",
            pair_id=pair.pair_id,
        )
        
        return pair
    
    def record_confusion(
        self,
        retrieved_entity: str,
        wanted_entity: str,
        context: str,
    ) -> ConfusionEvent:
        """
        Record a confusion between two entities.
        
        Args:
            retrieved_entity: What the agent retrieved.
            wanted_entity: What the user actually wanted.
            context: Query or situation.
            
        Returns:
            ConfusionEvent (new or updated).
        """
        key = (min(retrieved_entity, wanted_entity), max(retrieved_entity, wanted_entity))
        
        if key in self.confusions:
            event = self.confusions[key]
            event.count += 1
            event.last_occurred = datetime.now(timezone.utc)
        else:
            event = ConfusionEvent(
                entity_a=key[0],
                entity_b=key[1],
                context=context,
            )
            self.confusions[key] = event
        
        return event
    
    def add_admin_pair(
        self,
        positive: str,
        negative: str,
        explanation: str,
        entity_id: Optional[str] = None,
    ) -> ContrastivePair:
        """
        Add admin-curated contrastive pair.
        
        Args:
            positive: What it IS.
            negative: What it is NOT.
            explanation: Why this distinction matters.
            entity_id: Optional entity this relates to.
            
        Returns:
            Created ContrastivePair.
        """
        import uuid
        
        pair = ContrastivePair(
            pair_id=str(uuid.uuid4()),
            positive=positive,
            negative=negative,
            explanation=explanation,
            source="admin",
            entity_id=entity_id,
            confidence=1.0,  # High confidence from admin
        )
        
        self.pairs.append(pair)
        
        return pair
    
    async def generate_negatives(
        self,
        facts: list[dict[str, Any]],
        max_pairs: int = 10,
    ) -> list[ContrastivePair]:
        """
        Generate "what it's NOT" counterparts for facts.
        
        Args:
            facts: Facts to generate negatives for.
            max_pairs: Maximum pairs to generate.
            
        Returns:
            List of generated ContrastivePairs.
        """
        if not self.llm:
            return []
        
        generated = []
        
        for fact in facts[:max_pairs]:
            content = fact.get("content", "")
            if not content:
                continue
            
            prompt = f"""
Given this fact:
"{content}"

Generate 2-3 common misconceptions or things this could be confused with.
Format each as: "NOT: [misconception]"

Example:
Fact: "Due date is when payment must be received"
NOT: Due date is NOT the statement closing date
NOT: Due date is NOT the transaction date
"""
            
            response = await self.llm(prompt)
            
            # Parse negatives from response
            for line in response.split("\n"):
                line = line.strip()
                if line.upper().startswith("NOT:"):
                    negative = line[4:].strip()
                    
                    import uuid
                    pair = ContrastivePair(
                        pair_id=str(uuid.uuid4()),
                        positive=content,
                        negative=f"NOT: {negative}",
                        explanation="LLM-generated contrast",
                        source="generated",
                        entity_id=fact.get("id"),
                        confidence=0.7,  # Lower confidence for generated
                    )
                    
                    self.pairs.append(pair)
                    generated.append(pair)
        
        if generated:
            audit.log_raw(
                "learning",
                "contrastive_generated",
                "mml",
                "completed",
                pairs_generated=len(generated),
            )
        
        return generated
    
    async def create_graph_edges(self) -> int:
        """
        Create NOT_SAME_AS edges in AM graph for all pairs.
        
        Returns:
            Number of edges created.
        """
        if not self.graph_add_edge:
            return 0
        
        created = 0
        
        for pair in self.pairs:
            if pair.entity_id:
                success = await self.graph_add_edge(
                    pair.entity_id,
                    f"contrast_{pair.pair_id}",
                    "NOT_SAME_AS",
                    {"explanation": pair.explanation},
                )
                if success:
                    created += 1
        
        # Also create COMMONLY_CONFUSED_WITH edges from confusion events
        for (entity_a, entity_b), event in self.confusions.items():
            if event.count >= 2:  # Only if confused multiple times
                success = await self.graph_add_edge(
                    entity_a,
                    entity_b,
                    "COMMONLY_CONFUSED_WITH",
                    {"count": event.count, "context": event.context},
                )
                if success:
                    created += 1
        
        return created
    
    def get_contrasts_for_entity(self, entity_id: str) -> list[ContrastivePair]:
        """Get all contrastive pairs for an entity."""
        return [p for p in self.pairs if p.entity_id == entity_id]
    
    def get_common_confusions(self, min_count: int = 2) -> list[ConfusionEvent]:
        """Get commonly confused entity pairs."""
        return [e for e in self.confusions.values() if e.count >= min_count]
    
    def retrieve_with_contrast(
        self,
        entity_id: str,
        fact: str,
    ) -> dict[str, Any]:
        """
        Retrieve a fact with its contrasts for LLM context.
        
        Args:
            entity_id: Entity ID.
            fact: The fact content.
            
        Returns:
            Dict with fact and contrasts.
        """
        contrasts = self.get_contrasts_for_entity(entity_id)
        
        return {
            "fact": fact,
            "not_this": [p.negative for p in contrasts],
            "explanations": [p.explanation for p in contrasts],
        }
    
    def get_stats(self) -> dict[str, Any]:
        """Get contrastive learning statistics."""
        source_counts = {}
        for pair in self.pairs:
            source_counts[pair.source] = source_counts.get(pair.source, 0) + 1
        
        return {
            "total_pairs": len(self.pairs),
            "pairs_by_source": source_counts,
            "confusion_events": len(self.confusions),
            "high_confusion_pairs": len(self.get_common_confusions(3)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_contrastive_learner: Optional[ContrastiveLearner] = None


def get_contrastive_learner() -> ContrastiveLearner:
    """Get the singleton ContrastiveLearner instance."""
    global _contrastive_learner
    if _contrastive_learner is None:
        _contrastive_learner = ContrastiveLearner()
    return _contrastive_learner

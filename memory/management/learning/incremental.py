"""
Incremental / Online Learning (Learning Type 8)

Continuously update models and indexes without full retraining.
The agent gets smarter with every interaction without a "retrain" step.

Algorithm: Online index updates + streaming bandit updates
Trigger: Real-time (on every interaction)
LLM: No (pure computation)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable

import numpy as np

from core import config, log
from core import audit


@dataclass
class IncrementalUpdate:
    """Record of an incremental update."""
    update_type: str  # "faiss", "bm25", "bandit", "retention", "access"
    target: str  # What was updated
    details: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "update_type": self.update_type,
            "target": self.target,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class EbbinghausRetention:
    """
    Ebbinghaus forgetting curve for memory retention scoring.
    
    Retention = e^(-t/S) where:
    - t = time since last access
    - S = stability (increases with each successful recall)
    """
    
    def __init__(self, initial_stability: float = 1.0):
        """
        Initialize retention tracker.
        
        Args:
            initial_stability: Initial stability value (days).
        """
        self.initial_stability = initial_stability
        self._memories: dict[str, dict[str, Any]] = {}
    
    def register(self, memory_id: str) -> None:
        """Register a new memory."""
        self._memories[memory_id] = {
            "stability": self.initial_stability,
            "last_access": datetime.now(timezone.utc),
            "access_count": 0,
        }
    
    def access(self, memory_id: str) -> float:
        """
        Record an access and return new retention score.
        
        Args:
            memory_id: Memory being accessed.
            
        Returns:
            New retention score (0-1).
        """
        if memory_id not in self._memories:
            self.register(memory_id)
        
        mem = self._memories[memory_id]
        
        # Increase stability with each successful recall
        # Stability grows logarithmically
        mem["access_count"] += 1
        mem["stability"] = self.initial_stability * (1 + np.log1p(mem["access_count"]))
        mem["last_access"] = datetime.now(timezone.utc)
        
        return self.get_retention(memory_id)
    
    def get_retention(self, memory_id: str) -> float:
        """
        Get current retention score for a memory.
        
        Args:
            memory_id: Memory to check.
            
        Returns:
            Retention score (0-1).
        """
        if memory_id not in self._memories:
            return 0.0
        
        mem = self._memories[memory_id]
        
        # Time since last access in days
        now = datetime.now(timezone.utc)
        t = (now - mem["last_access"]).total_seconds() / 86400.0
        
        # Retention = e^(-t/S)
        retention = np.exp(-t / mem["stability"])
        
        return float(retention)
    
    def get_weak_memories(self, threshold: float = 0.3) -> list[str]:
        """Get memories with low retention."""
        return [
            mid for mid in self._memories
            if self.get_retention(mid) < threshold
        ]


class AccessCounter:
    """Simple access counter for memories."""
    
    def __init__(self):
        self._counts: dict[str, int] = {}
    
    def increment(self, memory_id: str) -> int:
        """Increment and return new count."""
        self._counts[memory_id] = self._counts.get(memory_id, 0) + 1
        return self._counts[memory_id]
    
    def get(self, memory_id: str) -> int:
        """Get current count."""
        return self._counts.get(memory_id, 0)
    
    def get_top(self, n: int = 10) -> list[tuple[str, int]]:
        """Get top N accessed memories."""
        sorted_items = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]


class TimeDecayScorer:
    """Time-decay scoring for recency weighting."""
    
    def __init__(self, half_life_days: float = 7.0):
        """
        Initialize time-decay scorer.
        
        Args:
            half_life_days: Half-life for decay.
        """
        self.half_life_days = half_life_days
        self._timestamps: dict[str, datetime] = {}
    
    def update(self, memory_id: str) -> None:
        """Update timestamp for a memory."""
        self._timestamps[memory_id] = datetime.now(timezone.utc)
    
    def score(self, memory_id: str) -> float:
        """
        Get time-decay score for a memory.
        
        Args:
            memory_id: Memory to score.
            
        Returns:
            Score (0-1), higher = more recent.
        """
        if memory_id not in self._timestamps:
            return 0.0
        
        now = datetime.now(timezone.utc)
        age_days = (now - self._timestamps[memory_id]).total_seconds() / 86400.0
        
        # Exponential decay with half-life
        decay = 0.5 ** (age_days / self.half_life_days)
        
        return float(decay)


class IncrementalLearner:
    """
    Manages all incremental/online learning updates.
    
    Coordinates updates to:
    - FAISS indexes (add new vectors)
    - BM25 indexes (add new documents)
    - Bandit weights (update after outcomes)
    - Retention scores (Ebbinghaus curve)
    - Access counts
    - Time-decay scores
    
    Attributes:
        retention: Ebbinghaus retention tracker.
        access_counter: Access counter.
        time_decay: Time-decay scorer.
    """
    
    def __init__(
        self,
        faiss_add: Optional[Callable[[str, np.ndarray], None]] = None,
        bm25_add: Optional[Callable[[str, str], None]] = None,
        bandit_update: Optional[Callable[[str, np.ndarray, float], None]] = None,
    ):
        """
        Initialize incremental learner.
        
        Args:
            faiss_add: Function to add vector to FAISS.
            bm25_add: Function to add document to BM25.
            bandit_update: Function to update bandit.
        """
        self.faiss_add = faiss_add
        self.bm25_add = bm25_add
        self.bandit_update = bandit_update
        
        self.retention = EbbinghausRetention()
        self.access_counter = AccessCounter()
        self.time_decay = TimeDecayScorer()
        
        self._updates: list[IncrementalUpdate] = []
    
    def on_memory_created(
        self,
        memory_id: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        memory_type: str = "sfm",
    ) -> None:
        """
        Handle new memory creation.
        
        Args:
            memory_id: ID of new memory.
            content: Text content.
            embedding: Optional embedding vector.
            memory_type: Type of memory (sfm, lfm, etc.).
        """
        # Register in retention tracker
        self.retention.register(memory_id)
        
        # Update time-decay
        self.time_decay.update(memory_id)
        
        # Add to FAISS
        if embedding is not None and self.faiss_add:
            self.faiss_add(memory_id, embedding)
            self._log_update("faiss", memory_type, {"memory_id": memory_id})
        
        # Add to BM25
        if self.bm25_add:
            self.bm25_add(memory_id, content)
            self._log_update("bm25", memory_type, {"memory_id": memory_id})
    
    def on_memory_accessed(
        self,
        memory_id: str,
        query_embedding: Optional[np.ndarray] = None,
        outcome: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        """
        Handle memory access.
        
        Args:
            memory_id: ID of accessed memory.
            query_embedding: Query that led to this access.
            outcome: Optional outcome (for bandit update).
            
        Returns:
            Updated scores.
        """
        # Update retention
        retention = self.retention.access(memory_id)
        
        # Update access count
        access_count = self.access_counter.increment(memory_id)
        
        # Update time-decay
        self.time_decay.update(memory_id)
        
        # Update bandit if outcome provided
        if outcome and query_embedding is not None and self.bandit_update:
            reward = self._compute_reward(outcome)
            self.bandit_update(memory_id, query_embedding, reward)
            self._log_update("bandit", "retrieval", {
                "memory_id": memory_id,
                "reward": reward,
            })
        
        self._log_update("access", "memory", {
            "memory_id": memory_id,
            "retention": retention,
            "access_count": access_count,
        })
        
        return {
            "retention": retention,
            "access_count": float(access_count),
            "time_decay": self.time_decay.score(memory_id),
        }
    
    def _compute_reward(self, outcome: dict[str, Any]) -> float:
        """Compute reward from outcome."""
        reward = 0.0
        
        if outcome.get("task_success"):
            reward += 0.5
        
        sentiment = outcome.get("sentiment", "neutral")
        if sentiment == "satisfied":
            reward += 0.3
        elif sentiment == "neutral":
            reward += 0.1
        
        if outcome.get("used_in_response"):
            reward += 0.2
        
        return min(1.0, reward)
    
    def _log_update(
        self,
        update_type: str,
        target: str,
        details: dict[str, Any],
    ) -> None:
        """Log an incremental update."""
        update = IncrementalUpdate(
            update_type=update_type,
            target=target,
            details=details,
        )
        self._updates.append(update)
        
        # Keep only recent updates
        if len(self._updates) > 1000:
            self._updates = self._updates[-1000:]
    
    def get_combined_score(
        self,
        memory_id: str,
        weights: Optional[dict[str, float]] = None,
    ) -> float:
        """
        Get combined score for a memory.
        
        Args:
            memory_id: Memory to score.
            weights: Optional weights for each component.
            
        Returns:
            Combined score (0-1).
        """
        if weights is None:
            weights = {
                "retention": 0.3,
                "access": 0.3,
                "time_decay": 0.4,
            }
        
        retention = self.retention.get_retention(memory_id)
        access = min(1.0, self.access_counter.get(memory_id) / 100.0)  # Normalize
        time_decay = self.time_decay.score(memory_id)
        
        score = (
            weights.get("retention", 0.3) * retention +
            weights.get("access", 0.3) * access +
            weights.get("time_decay", 0.4) * time_decay
        )
        
        return score
    
    def get_stats(self) -> dict[str, Any]:
        """Get incremental learning statistics."""
        update_counts = {}
        for update in self._updates:
            update_counts[update.update_type] = update_counts.get(update.update_type, 0) + 1
        
        return {
            "total_updates": len(self._updates),
            "updates_by_type": update_counts,
            "tracked_memories": len(self.retention._memories),
            "top_accessed": self.access_counter.get_top(5),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_incremental_learner: Optional[IncrementalLearner] = None


def get_incremental_learner() -> IncrementalLearner:
    """Get the singleton IncrementalLearner instance."""
    global _incremental_learner
    if _incremental_learner is None:
        _incremental_learner = IncrementalLearner()
    return _incremental_learner

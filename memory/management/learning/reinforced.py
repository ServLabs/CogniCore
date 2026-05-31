"""
Reinforced Learning (Learning Type 1)

Track which retrieval paths led to good vs bad outcomes.
Strengthen successful paths, weaken failed ones.

Algorithm: LinUCB contextual bandit
Trigger: Every user interaction
LLM: No (pure computation)
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core import config


@dataclass
class BanditArm:
    """A single arm in the contextual bandit."""
    arm_id: str
    A: np.ndarray  # d x d matrix
    b: np.ndarray  # d x 1 vector
    pulls: int = 0
    total_reward: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "pulls": self.pulls,
            "total_reward": self.total_reward,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BanditArm":
        return cls(
            arm_id=data["arm_id"],
            A=np.array(data["A"]),
            b=np.array(data["b"]),
            pulls=data["pulls"],
            total_reward=data["total_reward"],
        )


class LinUCBBandit:
    """
    LinUCB contextual bandit for retrieval path optimization.
    
    Each "arm" is a retrieval strategy (e.g., "sfm_only", "lfm_then_sfm", 
    "graph_first", etc.). Context is the query embedding + metadata.
    
    Attributes:
        alpha: Exploration parameter (higher = more exploration).
        d: Feature dimension.
        arms: Dict of arm_id -> BanditArm.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        d: int = 128,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize LinUCB bandit.
        
        Args:
            alpha: Exploration parameter.
            d: Feature dimension (context vector size).
            persist_path: Path to persist weights.
        """
        self.alpha = alpha
        self.d = d
        self.persist_path = persist_path or config.paths.data_dir / "bandit_weights.json"
        self.arms: dict[str, BanditArm] = {}
        
        # Load persisted weights if available
        self._load()
    
    def _create_arm(self, arm_id: str) -> BanditArm:
        """Create a new arm with identity matrix initialization."""
        return BanditArm(
            arm_id=arm_id,
            A=np.eye(self.d),
            b=np.zeros((self.d, 1)),
        )
    
    def get_arm(self, arm_id: str) -> BanditArm:
        """Get or create an arm."""
        if arm_id not in self.arms:
            self.arms[arm_id] = self._create_arm(arm_id)
        return self.arms[arm_id]
    
    def select_arm(
        self,
        context: np.ndarray,
        available_arms: list[str],
    ) -> tuple[str, float]:
        """
        Select the best arm given context using UCB.
        
        Args:
            context: d-dimensional context vector.
            available_arms: List of arm IDs to choose from.
            
        Returns:
            Tuple of (selected_arm_id, ucb_score).
        """
        if context.shape != (self.d, 1):
            context = context.reshape((self.d, 1))
        
        best_arm = None
        best_ucb = float("-inf")
        
        for arm_id in available_arms:
            arm = self.get_arm(arm_id)
            
            # Compute theta = A^-1 * b
            A_inv = np.linalg.inv(arm.A)
            theta = A_inv @ arm.b
            
            # Compute UCB: theta^T * x + alpha * sqrt(x^T * A^-1 * x)
            exploitation = float(theta.T @ context)
            exploration = self.alpha * math.sqrt(float(context.T @ A_inv @ context))
            ucb = exploitation + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm_id
        
        return best_arm, best_ucb
    
    def update(
        self,
        arm_id: str,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """
        Update arm weights after observing reward.
        
        Args:
            arm_id: The arm that was pulled.
            context: d-dimensional context vector.
            reward: Observed reward (0-1 scale).
        """
        if context.shape != (self.d, 1):
            context = context.reshape((self.d, 1))
        
        arm = self.get_arm(arm_id)
        
        # Update A = A + x * x^T
        arm.A = arm.A + context @ context.T
        
        # Update b = b + reward * x
        arm.b = arm.b + reward * context
        
        # Update stats
        arm.pulls += 1
        arm.total_reward += reward
        
        # Persist
        self._save()
    
    def get_stats(self) -> dict[str, Any]:
        """Get bandit statistics."""
        return {
            arm_id: {
                "pulls": arm.pulls,
                "total_reward": arm.total_reward,
                "avg_reward": arm.total_reward / arm.pulls if arm.pulls > 0 else 0,
            }
            for arm_id, arm in self.arms.items()
        }
    
    def _save(self) -> None:
        """Persist weights to disk."""
        data = {
            "alpha": self.alpha,
            "d": self.d,
            "arms": {arm_id: arm.to_dict() for arm_id, arm in self.arms.items()},
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps(data))
    
    def _load(self) -> None:
        """Load weights from disk."""
        if not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            self.alpha = data.get("alpha", self.alpha)
            self.d = data.get("d", self.d)
            self.arms = {
                arm_id: BanditArm.from_dict(arm_data)
                for arm_id, arm_data in data.get("arms", {}).items()
            }
        except Exception:
            pass  # Start fresh if load fails


# ══════════════════════════════════════════════════════════════════════════════
# Reinforced Learning Manager
# ══════════════════════════════════════════════════════════════════════════════

class ReinforcedLearning:
    """
    Reinforced learning manager.
    
    Tracks retrieval outcomes and updates bandit weights.
    """
    
    # Default retrieval strategies (arms)
    RETRIEVAL_STRATEGIES = [
        "sfm_only",           # Only search SFM
        "lfm_only",           # Only search LFM
        "sfm_then_lfm",       # SFM first, LFM if needed
        "lfm_then_sfm",       # LFM first, SFM if needed
        "graph_first",        # AM graph traversal first
        "hybrid_parallel",    # All sources in parallel
        "embedding_only",     # Pure vector search
        "keyword_boost",      # BM25 + embeddings
    ]
    
    def __init__(self, bandit: Optional[LinUCBBandit] = None):
        """
        Initialize reinforced learning.
        
        Args:
            bandit: LinUCB bandit instance.
        """
        self.bandit = bandit or LinUCBBandit()
    
    def select_strategy(
        self,
        query_embedding: np.ndarray,
        available_strategies: Optional[list[str]] = None,
    ) -> str:
        """
        Select best retrieval strategy for a query.
        
        Args:
            query_embedding: Query embedding vector.
            available_strategies: Strategies to choose from.
            
        Returns:
            Selected strategy name.
        """
        strategies = available_strategies or self.RETRIEVAL_STRATEGIES
        
        # Reduce embedding to bandit dimension if needed
        context = self._prepare_context(query_embedding)
        
        strategy, _ = self.bandit.select_arm(context, strategies)
        return strategy
    
    def record_outcome(
        self,
        strategy: str,
        query_embedding: np.ndarray,
        outcome: dict[str, Any],
    ) -> None:
        """
        Record the outcome of a retrieval strategy.
        
        Args:
            strategy: Strategy that was used.
            query_embedding: Query embedding.
            outcome: Outcome dict with sentiment, task_success, etc.
        """
        context = self._prepare_context(query_embedding)
        reward = self._compute_reward(outcome)
        
        self.bandit.update(strategy, context, reward)
    
    def _prepare_context(self, embedding: np.ndarray) -> np.ndarray:
        """Prepare context vector for bandit."""
        # Reduce to bandit dimension via mean pooling if needed
        if len(embedding) > self.bandit.d:
            # Simple dimensionality reduction: chunk and mean
            chunk_size = len(embedding) // self.bandit.d
            reduced = np.array([
                embedding[i * chunk_size:(i + 1) * chunk_size].mean()
                for i in range(self.bandit.d)
            ])
            return reduced.reshape((self.bandit.d, 1))
        elif len(embedding) < self.bandit.d:
            # Pad with zeros
            padded = np.zeros(self.bandit.d)
            padded[:len(embedding)] = embedding
            return padded.reshape((self.bandit.d, 1))
        else:
            return embedding.reshape((self.bandit.d, 1))
    
    def _compute_reward(self, outcome: dict[str, Any]) -> float:
        """
        Compute reward from outcome.
        
        Reward components:
        - Sentiment: positive = 1, neutral = 0.5, negative = 0
        - Task success: True = 0.5, False = 0
        - Response time: Fast = 0.2, Slow = 0
        """
        reward = 0.0
        
        # Sentiment component (0-1)
        sentiment = outcome.get("sentiment", "neutral")
        if sentiment == "satisfied":
            reward += 0.5
        elif sentiment == "neutral":
            reward += 0.25
        # frustrated = 0
        
        # Task success component (0-0.5)
        if outcome.get("task_success", False):
            reward += 0.3
        
        # Response time component (0-0.2)
        response_ms = outcome.get("response_time_ms", 1000)
        if response_ms < 500:
            reward += 0.2
        elif response_ms < 1000:
            reward += 0.1
        
        return min(1.0, reward)
    
    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        return self.bandit.get_stats()


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_reinforced_learning: Optional[ReinforcedLearning] = None


def get_reinforced_learning() -> ReinforcedLearning:
    """Get the singleton ReinforcedLearning instance."""
    global _reinforced_learning
    if _reinforced_learning is None:
        _reinforced_learning = ReinforcedLearning()
    return _reinforced_learning

"""
NLI Connector

Natural Language Inference — contradiction & entailment detection.
Uses a local cross-encoder model (no LLM calls, ~1000 checks/second on CPU).
"""

import asyncio
from typing import Optional

from config import config
from connectors.base import BaseConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class NLIConnector(BaseConnector):
    """
    Natural Language Inference connector.
    
    Provides:
    - Contradiction detection
    - Entailment detection
    - Batch prediction
    
    Uses cross-encoder/nli-deberta-v3-small by default (~140MB).
    """
    
    LABELS = ["contradiction", "entailment", "neutral"]
    
    def __init__(self):
        """Initialize NLI connector."""
        self._model = None
    
    def is_configured(self) -> bool:
        """Check if NLI is configured."""
        return config.nli.model_name is not None
    
    async def connect(self) -> None:
        """Load NLI model."""
        from sentence_transformers import CrossEncoder
        
        path = config.nli.model_path or config.nli.model_name
        self._model = await asyncio.to_thread(CrossEncoder, path)
    
    async def disconnect(self) -> None:
        """Release NLI model."""
        self._model = None
    
    async def health_check(self) -> bool:
        """Check if NLI model is working."""
        if not self._model:
            return False
        
        try:
            self.predict("The sky is blue.", "The sky is red.")
            return True
        except Exception:
            return False
    
    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        """
        Predict NLI labels for a premise-hypothesis pair.
        
        Args:
            premise: First statement.
            hypothesis: Second statement.
            
        Returns:
            Dict with label scores: {contradiction: 0.95, entailment: 0.02, neutral: 0.03}
        """
        scores = self._model.predict([(premise, hypothesis)])[0]
        return dict(zip(self.LABELS, scores.tolist()))
    
    def is_contradicting(self, a: str, b: str) -> bool:
        """
        Quick check: do these two statements contradict?
        
        Args:
            a: First statement.
            b: Second statement.
            
        Returns:
            True if contradiction score >= threshold.
        """
        result = self.predict(a, b)
        return result["contradiction"] >= config.nli.contradiction_threshold
    
    def is_entailing(self, a: str, b: str) -> bool:
        """
        Quick check: does a entail b?
        
        Args:
            a: First statement.
            b: Second statement.
            
        Returns:
            True if entailment score >= threshold.
        """
        result = self.predict(a, b)
        return result["entailment"] >= config.nli.entailment_threshold
    
    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """
        Batch NLI prediction.
        
        Args:
            pairs: List of (premise, hypothesis) tuples.
            
        Returns:
            List of label score dicts.
        """
        if not pairs:
            return []
        
        scores = self._model.predict(pairs)
        return [dict(zip(self.LABELS, s.tolist())) for s in scores]
    
    def find_contradictions(
        self,
        statements: list[str],
        threshold: Optional[float] = None,
    ) -> list[tuple[int, int, float]]:
        """
        Find contradicting pairs in a list of statements.
        
        Args:
            statements: List of statements.
            threshold: Contradiction threshold (uses config default if None).
            
        Returns:
            List of (idx_a, idx_b, score) tuples for contradicting pairs.
        """
        threshold = threshold or config.nli.contradiction_threshold
        contradictions = []
        
        # Generate all pairs
        pairs = []
        indices = []
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                pairs.append((statements[i], statements[j]))
                indices.append((i, j))
        
        if not pairs:
            return []
        
        # Batch predict
        results = self.predict_batch(pairs)
        
        # Filter contradictions
        for (i, j), result in zip(indices, results):
            if result["contradiction"] >= threshold:
                contradictions.append((i, j, result["contradiction"]))
        
        return contradictions
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        return ConnectorInfo(
            name="nli",
            connector_type=ConnectorType.AI,
            status=ConnectorStatus.CONNECTED if self._model else ConnectorStatus.DISCONNECTED,
            metadata={"model": config.nli.model_name},
        )

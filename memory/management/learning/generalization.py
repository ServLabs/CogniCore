"""
Experience Generalization (Learning Type 2)

Detect repeated similar tasks → extract the common pattern → 
create a reusable procedure.

Algorithm: DBSCAN clustering + Longest Common Subsequence + LLM distillation
Trigger: Sleep mode
LLM: Yes (expensive)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import numpy as np

from core import config, log
from core import audit


@dataclass
class ExecutionTrace:
    """A recorded execution trace."""
    trace_id: str
    user_id: str
    timestamp: datetime
    steps: list[dict[str, Any]]  # [{action, params, result}, ...]
    query: str
    domain: str
    success: bool
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "steps": self.steps,
            "query": self.query,
            "domain": self.domain,
            "success": self.success,
        }


@dataclass
class GeneralizedProcedure:
    """A procedure extracted from similar traces."""
    name: str
    domain: str
    description: str
    steps_nl: list[str]  # Natural language steps
    parameters: list[str]  # Extracted parameters
    source_traces: list[str]  # Trace IDs used
    confidence: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "steps_nl": self.steps_nl,
            "parameters": self.parameters,
            "source_traces": self.source_traces,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


def dbscan_cluster(
    embeddings: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 3,
) -> np.ndarray:
    """
    Cluster embeddings using DBSCAN.
    
    Args:
        embeddings: N x D embedding matrix.
        eps: Maximum distance between samples.
        min_samples: Minimum samples per cluster.
        
    Returns:
        Cluster labels (-1 = noise).
    """
    try:
        from sklearn.cluster import DBSCAN
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        return clustering.fit_predict(embeddings)
    except ImportError:
        # Fallback: simple distance-based clustering
        n = len(embeddings)
        labels = np.full(n, -1)
        cluster_id = 0
        
        for i in range(n):
            if labels[i] != -1:
                continue
            
            # Find neighbors
            neighbors = []
            for j in range(n):
                if i != j:
                    dist = 1 - np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    if dist < eps:
                        neighbors.append(j)
            
            if len(neighbors) >= min_samples - 1:
                labels[i] = cluster_id
                for j in neighbors:
                    if labels[j] == -1:
                        labels[j] = cluster_id
                cluster_id += 1
        
        return labels


def longest_common_subsequence(
    seq1: list[str],
    seq2: list[str],
) -> list[str]:
    """
    Find longest common subsequence of two action sequences.
    
    Args:
        seq1: First sequence of actions.
        seq2: Second sequence of actions.
        
    Returns:
        Common subsequence.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to find LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return list(reversed(lcs))


def extract_common_pattern(traces: list[ExecutionTrace]) -> list[str]:
    """
    Extract common action pattern from multiple traces.
    
    Uses pairwise LCS and voting.
    """
    if len(traces) < 2:
        return [s["action"] for s in traces[0].steps] if traces else []
    
    # Extract action sequences
    sequences = [[s["action"] for s in t.steps] for t in traces]
    
    # Pairwise LCS
    common = sequences[0]
    for seq in sequences[1:]:
        common = longest_common_subsequence(common, seq)
    
    return common


class ExperienceGeneralizer:
    """
    Generalizes repeated experiences into reusable procedures.
    
    Attributes:
        traces: Recent execution traces.
        llm: LLM callable for distillation.
        embedder: Embedding function.
    """
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        embedder: Optional[Callable[[str], Awaitable[np.ndarray]]] = None,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize generalizer.
        
        Args:
            llm: LLM callable for procedure generation.
            embedder: Embedding function.
            min_cluster_size: Minimum traces to form a pattern.
            similarity_threshold: Minimum similarity for clustering.
        """
        self.llm = llm
        self.embedder = embedder
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.traces: list[ExecutionTrace] = []
    
    def record_trace(self, trace: ExecutionTrace) -> None:
        """Record an execution trace."""
        self.traces.append(trace)
        
        # Keep only recent traces (last 1000)
        if len(self.traces) > 1000:
            self.traces = self.traces[-1000:]
    
    async def find_patterns(self) -> list[list[ExecutionTrace]]:
        """
        Find clusters of similar traces.
        
        Returns:
            List of trace clusters.
        """
        if len(self.traces) < self.min_cluster_size:
            return []
        
        # Embed traces if needed
        for trace in self.traces:
            if trace.embedding is None and self.embedder:
                # Create trace representation
                trace_text = f"{trace.query} | " + " -> ".join(
                    s["action"] for s in trace.steps
                )
                trace.embedding = await self.embedder(trace_text)
        
        # Filter traces with embeddings
        embedded_traces = [t for t in self.traces if t.embedding is not None]
        if len(embedded_traces) < self.min_cluster_size:
            return []
        
        # Cluster
        embeddings = np.array([t.embedding for t in embedded_traces])
        labels = dbscan_cluster(
            embeddings,
            eps=1 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
        )
        
        # Group by cluster
        clusters: dict[int, list[ExecutionTrace]] = {}
        for trace, label in zip(embedded_traces, labels):
            if label >= 0:  # Ignore noise
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(trace)
        
        return list(clusters.values())
    
    async def generalize_cluster(
        self,
        traces: list[ExecutionTrace],
    ) -> Optional[GeneralizedProcedure]:
        """
        Generalize a cluster of traces into a procedure.
        
        Args:
            traces: Cluster of similar traces.
            
        Returns:
            Generalized procedure or None.
        """
        if len(traces) < self.min_cluster_size:
            return None
        
        # Extract common pattern
        common_actions = extract_common_pattern(traces)
        
        if len(common_actions) < 2:
            return None
        
        # Use LLM to distill into procedure
        if self.llm:
            prompt = self._build_distillation_prompt(traces, common_actions)
            response = await self.llm(prompt)
            procedure = self._parse_procedure_response(response, traces)
        else:
            # Fallback: simple extraction
            procedure = GeneralizedProcedure(
                name=f"auto_procedure_{traces[0].domain}",
                domain=traces[0].domain,
                description=f"Auto-generated from {len(traces)} similar traces",
                steps_nl=common_actions,
                parameters=[],
                source_traces=[t.trace_id for t in traces],
                confidence=len(traces) / 10.0,  # More traces = higher confidence
            )
        
        audit.log_raw(
            "learning",
            "generalization",
            "mml",
            "completed",
            procedure_name=procedure.name,
            source_count=len(traces),
        )
        
        return procedure
    
    def _build_distillation_prompt(
        self,
        traces: list[ExecutionTrace],
        common_actions: list[str],
    ) -> str:
        """Build prompt for LLM distillation."""
        examples = "\n\n".join([
            f"Example {i+1}:\n"
            f"Query: {t.query}\n"
            f"Steps: {' -> '.join(s['action'] for s in t.steps)}"
            for i, t in enumerate(traces[:5])  # Limit examples
        ])
        
        return f"""
Analyze these similar task executions and create a reusable procedure:

{examples}

Common action pattern: {' -> '.join(common_actions)}

Create a procedure with:
1. A descriptive name (snake_case)
2. A brief description
3. Step-by-step instructions (parameterized)
4. List of parameters needed

Format as JSON:
{{
    "name": "procedure_name",
    "description": "What this procedure does",
    "steps": ["Step 1: ...", "Step 2: ..."],
    "parameters": ["param1", "param2"]
}}
"""
    
    def _parse_procedure_response(
        self,
        response: str,
        traces: list[ExecutionTrace],
    ) -> GeneralizedProcedure:
        """Parse LLM response into procedure."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return GeneralizedProcedure(
                    name=data.get("name", "auto_procedure"),
                    domain=traces[0].domain,
                    description=data.get("description", ""),
                    steps_nl=data.get("steps", []),
                    parameters=data.get("parameters", []),
                    source_traces=[t.trace_id for t in traces],
                    confidence=len(traces) / 10.0,
                )
        except Exception as e:
            log.warning(f"Failed to parse procedure response: {e}")
        
        # Fallback
        return GeneralizedProcedure(
            name="auto_procedure",
            domain=traces[0].domain,
            description="Auto-generated procedure",
            steps_nl=[],
            parameters=[],
            source_traces=[t.trace_id for t in traces],
            confidence=0.5,
        )
    
    async def run(self) -> list[GeneralizedProcedure]:
        """
        Run generalization on all traces.
        
        Returns:
            List of generated procedures.
        """
        clusters = await self.find_patterns()
        
        procedures = []
        for cluster in clusters:
            procedure = await self.generalize_cluster(cluster)
            if procedure:
                procedures.append(procedure)
        
        return procedures


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_generalizer: Optional[ExperienceGeneralizer] = None


def get_generalizer() -> ExperienceGeneralizer:
    """Get the singleton ExperienceGeneralizer instance."""
    global _generalizer
    if _generalizer is None:
        _generalizer = ExperienceGeneralizer()
    return _generalizer

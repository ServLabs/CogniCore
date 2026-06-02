"""
Evaluation System

Dedicated module for monitoring learning effectiveness, memory quality,
and agent performance. Separate from Analytics (operational metrics) —
Evals tracks **quality**.

Eval Types:
- Fact Accuracy: Are SFM facts still true?
- Procedure Success: Do MM procedures complete without errors?
- Recall Precision: Is retrieved content relevant to query?
- Coherence Score: Do SFM facts contradict each other?
- Learning Drift: Are learned facts diverging from ground truth?
"""

from observability.evals.runner import EvalRunner, EvalResult, get_eval_runner
from observability.evals.metrics import (
    precision_at_k,
    recall_at_k,
    f1_score,
    accuracy,
)

__all__ = [
    # Runner
    "EvalRunner",
    "EvalResult",
    "get_eval_runner",
    # Metrics
    "precision_at_k",
    "recall_at_k",
    "f1_score",
    "accuracy",
]

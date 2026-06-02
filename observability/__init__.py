"""
Observability

Monitoring and quality tracking:
- analytics/  - Runtime metrics (latency, tokens, costs)
- evals/      - Quality metrics (accuracy, recall, coherence)
"""

from observability.analytics import (
    Analytics,
    get_analytics,
    record_metric,
    record_eval,
    query_metrics,
    get_dashboard_stats,
)
from observability.evals import (
    EvalRunner,
    EvalResult,
    get_eval_runner,
    precision_at_k,
    recall_at_k,
    f1_score,
    accuracy,
)

__all__ = [
    # Analytics
    "Analytics",
    "get_analytics",
    "record_metric",
    "record_eval",
    "query_metrics",
    "get_dashboard_stats",
    # Evals
    "EvalRunner",
    "EvalResult",
    "get_eval_runner",
    "precision_at_k",
    "recall_at_k",
    "f1_score",
    "accuracy",
]

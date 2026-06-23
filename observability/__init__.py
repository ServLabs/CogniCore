"""
Observability

Monitoring, auditing, and quality tracking:
- audit       - Universal audit logging (JSONL append-only)
- analytics/  - Runtime metrics (latency, tokens, costs)
- evals/      - Quality metrics (accuracy, recall, coherence)
- tracing     - Distributed tracing (traces + spans per request)
"""

from observability.audit import AuditEvent, AuditLogger, audit, audited
from observability.analytics import (
    Analytics,
    get_analytics,
    record_metric,
    record_eval,
    query_metrics,
    get_dashboard_stats,
)
from observability.analytics.analytics import record_count, record_latency
from observability.evals import (
    EvalRunner,
    EvalResult,
    get_eval_runner,
    precision_at_k,
    recall_at_k,
    f1_score,
    accuracy,
)
from observability.tracing import Tracer, Trace, Span, get_tracer

__all__ = [
    # Audit
    "AuditEvent",
    "AuditLogger",
    "audit",
    "audited",
    # Analytics
    "Analytics",
    "get_analytics",
    "record_metric",
    "record_count",
    "record_latency",
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
    # Tracing
    "Tracer",
    "Trace",
    "Span",
    "get_tracer",
]

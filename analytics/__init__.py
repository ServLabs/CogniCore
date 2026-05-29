"""
Analytics Layer

Cross-cutting metrics and performance tracking across the entire agent.
Append-only numerical data for admin/developer dashboards.
"""

from analytics.analytics import (
    Analytics,
    get_analytics,
    record_metric,
    record_eval,
    query_metrics,
    get_dashboard_stats,
)

__all__ = [
    "Analytics",
    "get_analytics",
    "record_metric",
    "record_eval",
    "query_metrics",
    "get_dashboard_stats",
]

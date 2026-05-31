"""
API Routes

All REST API route modules:
- ingestion: Data ingestion (text, files, facts)
- metrics: Analytics and performance data
- evals: Evaluation results and triggers
- admin: System health, config, management
"""

from api.routes import ingestion, metrics, evals, admin

__all__ = [
    "ingestion",
    "metrics",
    "evals",
    "admin",
]

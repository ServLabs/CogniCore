"""
Service Interface

Developer/service APIs:
- POST /ingest/text      - Ingest text
- POST /ingest/file      - Upload file
- POST /ingest/facts     - Insert facts
- GET  /ingest/status    - Job status
- GET  /metrics/*        - System metrics
- GET  /evals/*          - Evaluation results
- POST /evals/run        - Trigger eval
"""

from interfaces.service.routes import router as service_router

__all__ = ["service_router"]

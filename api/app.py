"""
API Application

FastAPI application for REST API endpoints.
Handles ingestion, metrics, evals, and admin operations.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import config, log
from audit import audit

from api.routes import ingestion, metrics, evals, admin


# ══════════════════════════════════════════════════════════════════════════════
# Application Lifecycle
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    log.info("API starting...")
    audit.log_raw("api", "startup", "app", "started")
    
    yield
    
    # Shutdown
    audit.log_raw("api", "shutdown", "app", "completed")
    log.info("API stopped")


# ══════════════════════════════════════════════════════════════════════════════
# Application Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """
    Create the FastAPI application.
    
    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="CogniCore API",
        description="REST API for ingestion, metrics, evals, and admin",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers
    app.include_router(ingestion.router, prefix="/ingest", tags=["Ingestion"])
    app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
    app.include_router(evals.router, prefix="/evals", tags=["Evaluations"])
    app.include_router(admin.router, prefix="/admin", tags=["Admin"])
    
    # Health check at root
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}
    
    return app


# ══════════════════════════════════════════════════════════════════════════════
# Singleton App
# ══════════════════════════════════════════════════════════════════════════════

_app: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """Get the singleton FastAPI app."""
    global _app
    if _app is None:
        _app = create_app()
    return _app

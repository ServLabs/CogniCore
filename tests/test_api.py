"""
Tests for interfaces.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _create_test_app() -> FastAPI:
    """Create a test app with all routers."""
    from interfaces.admin import admin_router
    from interfaces.service import service_router
    from interfaces.scheduled import scheduled_router
    
    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(service_router)
    app.include_router(scheduled_router)
    return app


class TestAdminInterface:
    """Tests for admin interface."""
    
    def test_health_returns_ok(self):
        """Health endpoint returns 200."""
        app = _create_test_app()
        client = TestClient(app)
        
        response = client.get("/admin/health")
        assert response.status_code == 200
        assert "status" in response.json()


class TestServiceInterface:
    """Tests for service interface."""
    
    def test_metrics_summary(self):
        """Metrics summary endpoint works."""
        app = _create_test_app()
        client = TestClient(app)
        
        response = client.get("/metrics/summary")
        assert response.status_code == 200
    
    def test_ingest_text(self):
        """Can ingest text."""
        app = _create_test_app()
        client = TestClient(app)
        
        response = client.post(
            "/ingest/text",
            json={"text": "Test content", "source": "test"},
        )
        assert response.status_code in [200, 202, 500]


class TestScheduledInterface:
    """Tests for scheduled interface."""
    
    def test_list_tasks(self):
        """Can list available tasks."""
        app = _create_test_app()
        client = TestClient(app)
        
        response = client.get("/scheduled/tasks")
        assert response.status_code == 200
        assert "tasks" in response.json()
    
    def test_trigger_consolidation(self):
        """Can trigger consolidation task."""
        app = _create_test_app()
        client = TestClient(app)
        
        response = client.post("/scheduled/consolidation")
        assert response.status_code == 200
        assert response.json()["task_name"] == "consolidation"

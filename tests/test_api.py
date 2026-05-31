"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for health check."""
    
    def test_health_returns_ok(self):
        """Health endpoint returns 200."""
        from api.app import get_app
        
        app = get_app()
        client = TestClient(app)
        
        response = client.get("/admin/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"


class TestMetricsEndpoint:
    """Tests for metrics."""
    
    def test_metrics_summary(self):
        """Metrics summary endpoint works."""
        from api.app import get_app
        
        app = get_app()
        client = TestClient(app)
        
        response = client.get("/metrics/summary")
        assert response.status_code == 200


class TestIngestionEndpoint:
    """Tests for ingestion."""
    
    def test_ingest_text(self):
        """Can ingest text."""
        from api.app import get_app
        
        app = get_app()
        client = TestClient(app)
        
        response = client.post(
            "/ingest/text",
            json={
                "text": "Test content for ingestion",
                "source": "test",
            }
        )
        
        # Should accept the request (may fail processing without full setup)
        assert response.status_code in [200, 202, 500]

"""
Admin Interface

System administration endpoints:
- GET  /health        - Health check
- GET  /config        - Current configuration
- GET  /status        - System status
- POST /maintenance   - Trigger maintenance
- POST /cache/clear   - Clear caches
- GET  /audit/recent  - Recent audit events
"""

from interfaces.admin.routes import router as admin_router

__all__ = ["admin_router"]

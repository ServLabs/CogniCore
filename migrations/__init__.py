"""
Migrations Module

Schema versioning and migration utilities:
- SQLite schema migrations
- FAISS index versioning
- Procedure versioning
"""

from migrations.sqlite_migrations import (
    migrate,
    get_current_version,
    MIGRATIONS,
)
from migrations.faiss_migrations import (
    check_index_compatibility,
    rebuild_index,
    get_index_metadata,
    save_index_metadata,
)

__all__ = [
    # SQLite
    "migrate",
    "get_current_version",
    "MIGRATIONS",
    # FAISS
    "check_index_compatibility",
    "rebuild_index",
    "get_index_metadata",
    "save_index_metadata",
]

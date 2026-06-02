"""
SQLite Schema Migrations

Version-controlled schema migrations for hot.db and cold.db.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from core import log


# ══════════════════════════════════════════════════════════════════════════════
# Migration Definitions
# ══════════════════════════════════════════════════════════════════════════════

MIGRATIONS = [
    # Version 1: Initial schema version table
    (1, """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT OR IGNORE INTO schema_version (version) VALUES (1);
    """),
    
    # Version 2: Add confidence to facts
    (2, """
        ALTER TABLE short_form_facts ADD COLUMN confidence REAL DEFAULT 0.8;
        INSERT OR REPLACE INTO schema_version (version) VALUES (2);
    """),
    
    # Version 3: Add source tracking
    (3, """
        ALTER TABLE short_form_facts ADD COLUMN source_doc_id TEXT;
        ALTER TABLE short_form_facts ADD COLUMN source_chunk_idx INTEGER;
        INSERT OR REPLACE INTO schema_version (version) VALUES (3);
    """),
    
    # Version 4: Add indexes for common queries
    (4, """
        CREATE INDEX IF NOT EXISTS idx_sfm_domain ON short_form_facts(domain);
        CREATE INDEX IF NOT EXISTS idx_sfm_subject ON short_form_facts(subject);
        CREATE INDEX IF NOT EXISTS idx_sfm_created ON short_form_facts(created_at);
        INSERT OR REPLACE INTO schema_version (version) VALUES (4);
    """),
    
    # Version 5: Add emotional memory sentiment index
    (5, """
        CREATE INDEX IF NOT EXISTS idx_em_sentiment ON emotional_memory(sentiment);
        CREATE INDEX IF NOT EXISTS idx_em_intensity ON emotional_memory(intensity);
        INSERT OR REPLACE INTO schema_version (version) VALUES (5);
    """),
    
    # Version 6: Add prospective memory status index
    (6, """
        CREATE INDEX IF NOT EXISTS idx_pm_status ON schedules(status);
        CREATE INDEX IF NOT EXISTS idx_pm_scheduled ON schedules(scheduled_at);
        INSERT OR REPLACE INTO schema_version (version) VALUES (6);
    """),
]


# ══════════════════════════════════════════════════════════════════════════════
# Migration Functions
# ══════════════════════════════════════════════════════════════════════════════

def get_current_version(conn: sqlite3.Connection) -> int:
    """
    Get current schema version.
    
    Args:
        conn: SQLite connection.
        
    Returns:
        Current version number (0 if no version table).
    """
    try:
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        result = cur.fetchone()
        return result[0] if result and result[0] else 0
    except sqlite3.OperationalError:
        return 0


def migrate(db_path: Path, target_version: Optional[int] = None) -> int:
    """
    Apply pending migrations.
    
    Args:
        db_path: Path to SQLite database.
        target_version: Optional target version (None = latest).
        
    Returns:
        Number of migrations applied.
    """
    conn = sqlite3.connect(db_path)
    current = get_current_version(conn)
    
    if target_version is None:
        target_version = max(v for v, _ in MIGRATIONS)
    
    applied = 0
    
    for version, sql in MIGRATIONS:
        if version > current and version <= target_version:
            log.info(f"Applying migration {version} to {db_path.name}...")
            try:
                conn.executescript(sql)
                conn.commit()
                applied += 1
                log.info(f"Migration {version} applied successfully")
            except sqlite3.Error as e:
                log.error(f"Migration {version} failed: {e}")
                conn.rollback()
                break
    
    conn.close()
    return applied


def rollback(db_path: Path, target_version: int) -> bool:
    """
    Rollback to a specific version.
    
    Note: This is a placeholder. Actual rollback requires
    storing reverse migrations.
    
    Args:
        db_path: Path to SQLite database.
        target_version: Target version to rollback to.
        
    Returns:
        True if successful.
    """
    log.warning(f"Rollback to version {target_version} requested but not implemented")
    return False


def create_migration(description: str) -> str:
    """
    Generate a new migration template.
    
    Args:
        description: Description of the migration.
        
    Returns:
        Migration SQL template.
    """
    next_version = max(v for v, _ in MIGRATIONS) + 1
    
    template = f'''
    # Version {next_version}: {description}
    ({next_version}, """
        -- Add your migration SQL here
        
        INSERT OR REPLACE INTO schema_version (version) VALUES ({next_version});
    """),
'''
    
    return template


def verify_schema(conn: sqlite3.Connection) -> dict[str, bool]:
    """
    Verify that expected tables exist.
    
    Args:
        conn: SQLite connection.
        
    Returns:
        Dict mapping table names to existence status.
    """
    expected_tables = [
        "schema_version",
        "short_form_facts",
        "emotional_memory",
        "schedules",
        "meta_memory",
    ]
    
    results = {}
    
    for table in expected_tables:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )
        results[table] = cur.fetchone() is not None
    
    return results

"""
Pytest fixtures for CogniCore tests.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config(temp_data_dir, monkeypatch):
    """Mock config with temp directories."""
    monkeypatch.setenv("COGNICORE_DATA_DIR", str(temp_data_dir))
    monkeypatch.setenv("COGNICORE_DEBUG", "true")
    
    # Re-import config to pick up new env vars
    from config import config
    return config

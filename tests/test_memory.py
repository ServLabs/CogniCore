"""
Tests for memory types.
"""

import pytest


class TestABM:
    """Tests for Autobiographical Memory."""
    
    def test_abm_is_frozen(self):
        """ABM should be immutable."""
        from memory.types.abm import AutobiographicalMemory
        
        abm = AutobiographicalMemory()
        
        with pytest.raises(Exception):  # FrozenInstanceError
            abm.name = "Changed"
    
    def test_abm_has_identity(self):
        """ABM has identity fields."""
        from memory.types.abm import AutobiographicalMemory
        
        abm = AutobiographicalMemory()
        assert abm.identity is not None
        assert abm.identity.name is not None


class TestWorkingMemory:
    """Tests for Working Memory."""
    
    def test_wm_message_role_enum(self):
        """MessageRole enum exists."""
        from memory.types.wm import MessageRole
        
        assert MessageRole.USER is not None
        assert MessageRole.ASSISTANT is not None


class TestSFM:
    """Tests for Short-Form Memory."""
    
    def test_fact_dataclass(self):
        """Fact dataclass works."""
        from memory.types.sfm import Fact
        
        fact = Fact(
            fact_id="test-1",
            content="Test fact",
            domain="general",
        )
        assert fact.fact_id == "test-1"
        assert fact.content == "Test fact"


class TestPM:
    """Tests for Prospective Memory."""
    
    def test_schedule_status_enum(self):
        """ScheduleStatus enum exists."""
        from memory.types.pm import ScheduleStatus
        
        assert ScheduleStatus.SCHEDULED is not None
        assert ScheduleStatus.COMPLETED is not None

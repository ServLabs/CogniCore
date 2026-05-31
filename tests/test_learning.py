"""
Tests for learning algorithms.
"""

import pytest
import numpy as np


class TestReinforcedLearning:
    """Tests for LinUCB bandit."""
    
    def test_bandit_creates_arms(self):
        """Bandit creates arms on demand."""
        from memory.management.learning import LinUCBBandit
        
        bandit = LinUCBBandit(d=8)
        arm = bandit.get_arm("test_arm")
        
        assert arm.arm_id == "test_arm"
        assert arm.A.shape == (8, 8)
    
    def test_bandit_selects_arm(self):
        """Bandit can select an arm."""
        from memory.management.learning import LinUCBBandit
        
        bandit = LinUCBBandit(d=8)
        context = np.random.randn(8)
        
        arm, score = bandit.select_arm(context, ["arm1", "arm2", "arm3"])
        
        assert arm in ["arm1", "arm2", "arm3"]
        assert isinstance(score, float)
    
    def test_bandit_updates(self):
        """Bandit updates after reward."""
        from memory.management.learning import LinUCBBandit
        
        bandit = LinUCBBandit(d=8)
        context = np.random.randn(8)
        
        bandit.update("arm1", context, reward=1.0)
        
        arm = bandit.get_arm("arm1")
        assert arm.pulls == 1
        assert arm.total_reward == 1.0


class TestIncrementalLearning:
    """Tests for incremental learning."""
    
    def test_retention_scoring(self):
        """Ebbinghaus retention works."""
        from memory.management.learning import EbbinghausRetention
        
        retention = EbbinghausRetention()
        retention.register("mem1")
        
        score = retention.get_retention("mem1")
        assert 0 <= score <= 1
    
    def test_access_counter(self):
        """Access counter increments."""
        from memory.management.learning import AccessCounter
        
        counter = AccessCounter()
        
        assert counter.get("mem1") == 0
        counter.increment("mem1")
        assert counter.get("mem1") == 1
        counter.increment("mem1")
        assert counter.get("mem1") == 2


class TestContrastiveLearning:
    """Tests for contrastive learning."""
    
    def test_add_correction_pair(self):
        """Can add correction-based pair."""
        from memory.management.learning import get_contrastive_learner
        
        learner = get_contrastive_learner()
        
        pair = learner.add_from_correction(
            wrong_statement="X is always Y",
            correct_statement="X can be Y or Z",
        )
        
        assert pair.positive == "X can be Y or Z"
        assert "NOT" in pair.negative


class TestMetaLearning:
    """Tests for meta-learning."""
    
    def test_log_event(self):
        """Can log learning events."""
        from memory.management.learning import get_meta_learner
        
        learner = get_meta_learner()
        
        event = learner.log_event(
            learning_type="reinforced",
            source="test",
            output_memory="sfm",
        )
        
        assert event.learning_type == "reinforced"
    
    def test_budget_allocation(self):
        """Budget allocations exist."""
        from memory.management.learning import get_meta_learner
        
        learner = get_meta_learner()
        
        budget = learner.get_budget("reinforced")
        assert 0 <= budget <= 1

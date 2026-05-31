"""
Learning Types

10 learning algorithms for memory evolution:

| # | Type          | Algorithm                    | LLM? | Trigger    |
|---|---------------|------------------------------|------|------------|
| 1 | Reinforced    | LinUCB contextual bandit     | No   | Real-time  |
| 2 | Generalization| DBSCAN + LCS + distillation  | Yes  | Sleep      |
| 3 | Abstraction   | Hierarchical clustering      | Yes  | Sleep      |
| 4 | Analogical    | Cross-domain alignment       | Yes  | Sleep      |
| 5 | Corrective    | NLI + fact versioning        | Yes  | Real-time  |
| 6 | Transfer      | Procedure adaptation         | Yes  | Sleep      |
| 7 | Meta-Learning | Effectiveness tracking       | No   | Weekly     |
| 8 | Incremental   | Online index updates         | No   | Real-time  |
| 9 | Observational | Action trace capture         | Yes  | Session end|
|10 | Contrastive   | Positive-negative pairing    | Yes  | Real-time  |
"""

# 1. Reinforced Learning
from memory.management.learning.reinforced import (
    LinUCBBandit,
    BanditArm,
    ReinforcedLearning,
    get_reinforced_learning,
)

# 2. Experience Generalization
from memory.management.learning.generalization import (
    ExecutionTrace,
    GeneralizedProcedure,
    ExperienceGeneralizer,
    get_generalizer,
)

# 3. Abstraction Learning
from memory.management.learning.abstraction import (
    AbstractionLevel,
    AbstractionResult,
    AbstractionLearner,
    get_abstraction_learner,
)

# 4. Analogical Learning
from memory.management.learning.analogical import (
    Analogy,
    AnalogicalResult,
    AnalogicalLearner,
    get_analogical_learner,
)

# 5. Corrective Learning
from memory.management.learning.corrective import (
    Correction,
    CorrectionResult,
    CorrectiveLearner,
    get_corrective_learner,
)

# 6. Transfer Learning
from memory.management.learning.transfer import (
    TransferredProcedure,
    TransferResult,
    TransferLearner,
    get_transfer_learner,
)

# 7. Meta-Learning
from memory.management.learning.meta import (
    LearningEvent,
    LearningTypeStats,
    MetaLearningResult,
    MetaLearner,
    get_meta_learner,
)

# 8. Incremental Learning
from memory.management.learning.incremental import (
    IncrementalUpdate,
    EbbinghausRetention,
    AccessCounter,
    TimeDecayScorer,
    IncrementalLearner,
    get_incremental_learner,
)

# 9. Observational Learning
from memory.management.learning.observational import (
    Action,
    ActionTrace,
    ObservedProcedure,
    ActionObserver,
    get_action_observer,
)

# 10. Contrastive Learning
from memory.management.learning.contrastive import (
    ContrastivePair,
    ConfusionEvent,
    ContrastiveLearner,
    get_contrastive_learner,
)


__all__ = [
    # 1. Reinforced
    "LinUCBBandit",
    "BanditArm",
    "ReinforcedLearning",
    "get_reinforced_learning",
    # 2. Generalization
    "ExecutionTrace",
    "GeneralizedProcedure",
    "ExperienceGeneralizer",
    "get_generalizer",
    # 3. Abstraction
    "AbstractionLevel",
    "AbstractionResult",
    "AbstractionLearner",
    "get_abstraction_learner",
    # 4. Analogical
    "Analogy",
    "AnalogicalResult",
    "AnalogicalLearner",
    "get_analogical_learner",
    # 5. Corrective
    "Correction",
    "CorrectionResult",
    "CorrectiveLearner",
    "get_corrective_learner",
    # 6. Transfer
    "TransferredProcedure",
    "TransferResult",
    "TransferLearner",
    "get_transfer_learner",
    # 7. Meta-Learning
    "LearningEvent",
    "LearningTypeStats",
    "MetaLearningResult",
    "MetaLearner",
    "get_meta_learner",
    # 8. Incremental
    "IncrementalUpdate",
    "EbbinghausRetention",
    "AccessCounter",
    "TimeDecayScorer",
    "IncrementalLearner",
    "get_incremental_learner",
    # 9. Observational
    "Action",
    "ActionTrace",
    "ObservedProcedure",
    "ActionObserver",
    "get_action_observer",
    # 10. Contrastive
    "ContrastivePair",
    "ConfusionEvent",
    "ContrastiveLearner",
    "get_contrastive_learner",
]

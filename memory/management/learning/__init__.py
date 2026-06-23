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

from memory.management.learning.reinforced import get_reinforced_learning
from memory.management.learning.generalization import get_generalizer
from memory.management.learning.abstraction import get_abstraction_learner
from memory.management.learning.analogical import get_analogical_learner
from memory.management.learning.corrective import get_corrective_learner
from memory.management.learning.transfer import get_transfer_learner
from memory.management.learning.meta import get_meta_learner
from memory.management.learning.incremental import get_incremental_learner
from memory.management.learning.observational import get_action_observer
from memory.management.learning.contrastive import get_contrastive_learner


__all__ = [
    "get_reinforced_learning",
    "get_generalizer",
    "get_abstraction_learner",
    "get_analogical_learner",
    "get_corrective_learner",
    "get_transfer_learner",
    "get_meta_learner",
    "get_incremental_learner",
    "get_action_observer",
    "get_contrastive_learner",
]

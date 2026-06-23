"""
Maintenance Daemons

7 background maintenance tasks:
- Consolidation: Process conversations into long-term memory
- Forgetting: Demote stale SFM facts to LFM
- Conflict: Detect and resolve contradictions
- Coherence: Ensure cross-type consistency
- Optimization: Index and database maintenance
- Integrity: Self-healing, repair broken references
- Linking: Auto-create cross-memory associations
"""

from memory.management.maintenance.consolidation import get_consolidation_daemon
from memory.management.maintenance.forgetting import get_forgetting_daemon
from memory.management.maintenance.conflict import get_conflict_daemon
from memory.management.maintenance.coherence import get_coherence_daemon
from memory.management.maintenance.optimization import get_optimization_daemon
from memory.management.maintenance.integrity import get_integrity_daemon
from memory.management.maintenance.linking import get_linking_daemon

__all__ = [
    "get_consolidation_daemon",
    "get_forgetting_daemon",
    "get_conflict_daemon",
    "get_coherence_daemon",
    "get_optimization_daemon",
    "get_integrity_daemon",
    "get_linking_daemon",
]

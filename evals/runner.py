"""
Eval Runner

Orchestrates all evaluation types. Runs evals, stores results,
and generates reports.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from audit import audit
from core import log


# ══════════════════════════════════════════════════════════════════════════════
# Eval Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """Result of an evaluation run."""
    eval_type: str
    score: float  # 0.0 - 1.0
    sample_size: int
    passed: int
    failed: int
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_type": self.eval_type,
            "score": self.score,
            "sample_size": self.sample_size,
            "passed": self.passed,
            "failed": self.failed,
            "details": self.details,
            "timestamp": self.timestamp,
        }
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        total = self.passed + self.failed
        return self.passed / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Eval Runner
# ══════════════════════════════════════════════════════════════════════════════

class EvalRunner:
    """
    Orchestrates all evaluation types.
    
    Provides:
    - Eval registration
    - Run all or single evals
    - Result storage
    - Reporting
    """
    
    def __init__(self):
        """Initialize eval runner."""
        self._evals: dict[str, Callable] = {}
        self._results: list[EvalResult] = []
    
    def register(self, name: str, eval_fn: Callable) -> None:
        """
        Register an eval function.
        
        Args:
            name: Eval name.
            eval_fn: Async callable that returns EvalResult.
        """
        self._evals[name] = eval_fn
        log.debug(f"Registered eval: {name}")
    
    def list_evals(self) -> list[str]:
        """List all registered eval names."""
        return list(self._evals.keys())
    
    async def run_all(self) -> list[EvalResult]:
        """
        Run all registered evals.
        
        Returns:
            List of EvalResults.
        """
        results = []
        
        for name, eval_fn in self._evals.items():
            result = await self.run_single(name)
            if result:
                results.append(result)
        
        return results
    
    async def run_single(self, name: str) -> Optional[EvalResult]:
        """
        Run a specific eval.
        
        Args:
            name: Eval name.
            
        Returns:
            EvalResult or None if failed.
        """
        if name not in self._evals:
            log.error(f"Unknown eval: {name}")
            return None
        
        audit.log_raw("evals", "eval_started", "runner", "started", target=name)
        
        try:
            result = await self._evals[name]()
            self._store_result(result)
            
            audit.log_raw(
                "evals",
                "eval_completed",
                "runner",
                "completed",
                target=name,
                details={"score": result.score, "sample_size": result.sample_size},
            )
            
            log.info(f"Eval {name}: score={result.score:.3f}, passed={result.passed}, failed={result.failed}")
            return result
        
        except Exception as e:
            audit.log_raw(
                "evals",
                "eval_failed",
                "runner",
                "failed",
                target=name,
                error=str(e),
            )
            log.error(f"Eval {name} failed: {e}", exc_info=True)
            return None
    
    async def run_by_type(self, eval_type: str) -> list[EvalResult]:
        """
        Run all evals of a specific type.
        
        Args:
            eval_type: Type prefix (e.g., "fact", "recall").
            
        Returns:
            List of EvalResults.
        """
        results = []
        
        for name in self._evals:
            if name.startswith(eval_type):
                result = await self.run_single(name)
                if result:
                    results.append(result)
        
        return results
    
    def _store_result(self, result: EvalResult) -> None:
        """Store eval result."""
        self._results.append(result)
        
        # Keep only last 1000 results in memory
        if len(self._results) > 1000:
            self._results = self._results[-1000:]
    
    def get_recent_results(
        self,
        eval_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[EvalResult]:
        """
        Get recent eval results.
        
        Args:
            eval_type: Filter by type (None = all).
            limit: Max results to return.
            
        Returns:
            List of EvalResults.
        """
        results = self._results
        
        if eval_type:
            results = [r for r in results if r.eval_type == eval_type]
        
        return results[-limit:]
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of all eval results.
        
        Returns:
            Dict with summary stats per eval type.
        """
        summary = {}
        
        for result in self._results:
            if result.eval_type not in summary:
                summary[result.eval_type] = {
                    "count": 0,
                    "total_score": 0.0,
                    "total_passed": 0,
                    "total_failed": 0,
                }
            
            s = summary[result.eval_type]
            s["count"] += 1
            s["total_score"] += result.score
            s["total_passed"] += result.passed
            s["total_failed"] += result.failed
        
        # Calculate averages
        for eval_type, stats in summary.items():
            if stats["count"] > 0:
                stats["avg_score"] = stats["total_score"] / stats["count"]
        
        return summary


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_eval_runner: Optional[EvalRunner] = None


def get_eval_runner() -> EvalRunner:
    """Get the singleton EvalRunner instance."""
    global _eval_runner
    if _eval_runner is None:
        _eval_runner = EvalRunner()
    return _eval_runner

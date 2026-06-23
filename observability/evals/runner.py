"""
Eval Runner

Orchestrates all evaluation types. Runs evals, stores results,
and generates reports. Persists results to DuckDB.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from config import config
from logger import log
from observability.audit import audit


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
        """Store eval result in memory and persist to DuckDB."""
        self._results.append(result)
        
        # Keep only last 1000 results in memory
        if len(self._results) > 1000:
            self._results = self._results[-1000:]
        
        # Persist to DuckDB
        try:
            self._persist_to_db(result)
        except Exception as e:
            log.warning("Failed to persist eval result to DuckDB: %s", e)
    
    def _persist_to_db(self, result: EvalResult) -> None:
        """Write eval result to DuckDB eval_results table."""
        import duckdb
        
        db_path = str(config.paths.analytics_db)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = duckdb.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_results (
                    ts TIMESTAMP NOT NULL,
                    eval_type VARCHAR NOT NULL,
                    score DOUBLE NOT NULL,
                    sample_size INTEGER NOT NULL,
                    passed INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    details JSON
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval_results_ts
                ON eval_results(ts)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval_results_type
                ON eval_results(eval_type, ts)
            """)
            conn.execute(
                "INSERT INTO eval_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.fromisoformat(result.timestamp) if result.timestamp else datetime.now(timezone.utc),
                    result.eval_type,
                    result.score,
                    result.sample_size,
                    result.passed,
                    result.failed,
                    json.dumps(result.details, default=str) if result.details else None,
                ),
            )
        finally:
            conn.close()
    
    def get_recent_results(
        self,
        eval_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[EvalResult]:
        """
        Get recent eval results from DuckDB (falls back to in-memory).
        
        Args:
            eval_type: Filter by type (None = all).
            limit: Max results to return.
            
        Returns:
            List of EvalResults.
        """
        try:
            return self._query_results_from_db(eval_type, limit)
        except Exception:
            # Fallback to in-memory
            results = self._results
            if eval_type:
                results = [r for r in results if r.eval_type == eval_type]
            return results[-limit:]
    
    def _query_results_from_db(
        self,
        eval_type: Optional[str],
        limit: int,
    ) -> list[EvalResult]:
        """Query eval results from DuckDB."""
        import duckdb
        
        db_path = str(config.paths.analytics_db)
        conn = duckdb.connect(db_path, read_only=True)
        try:
            where = f"WHERE eval_type = '{eval_type}'" if eval_type else ""
            rows = conn.execute(f"""
                SELECT ts, eval_type, score, sample_size, passed, failed, details
                FROM eval_results
                {where}
                ORDER BY ts DESC
                LIMIT {limit}
            """).fetchall()
        finally:
            conn.close()
        
        results = []
        for row in rows:
            results.append(EvalResult(
                eval_type=row[1],
                score=row[2],
                sample_size=row[3],
                passed=row[4],
                failed=row[5],
                details=json.loads(row[6]) if row[6] else {},
                timestamp=row[0].isoformat() if row[0] else None,
            ))
        return results
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of all eval results from DuckDB.
        
        Returns:
            Dict with summary stats per eval type.
        """
        try:
            return self._query_summary_from_db()
        except Exception:
            return self._in_memory_summary()
    
    def _query_summary_from_db(self) -> dict[str, Any]:
        """Query eval summary from DuckDB."""
        import duckdb
        
        db_path = str(config.paths.analytics_db)
        conn = duckdb.connect(db_path, read_only=True)
        try:
            rows = conn.execute("""
                SELECT 
                    eval_type,
                    COUNT(*) as count,
                    AVG(score) as avg_score,
                    MIN(score) as min_score,
                    MAX(score) as max_score,
                    SUM(passed) as total_passed,
                    SUM(failed) as total_failed,
                    MAX(ts) as last_run
                FROM eval_results
                GROUP BY eval_type
                ORDER BY eval_type
            """).fetchall()
        finally:
            conn.close()
        
        summary = {}
        for row in rows:
            summary[row[0]] = {
                "count": row[1],
                "avg_score": row[2],
                "min_score": row[3],
                "max_score": row[4],
                "total_passed": row[5],
                "total_failed": row[6],
                "last_run": row[7].isoformat() if row[7] else None,
            }
        return summary
    
    def _in_memory_summary(self) -> dict[str, Any]:
        """Fallback: compute summary from in-memory results."""
        summary: dict[str, Any] = {}
        
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
        
        for eval_type, stats in summary.items():
            if stats["count"] > 0:
                stats["avg_score"] = stats["total_score"] / stats["count"]
        
        return summary


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_eval_runner: Optional[EvalRunner] = None


def get_eval_runner() -> EvalRunner:
    """Get the singleton EvalRunner instance with all evals registered."""
    global _eval_runner
    if _eval_runner is None:
        _eval_runner = EvalRunner()
        _register_all_evals(_eval_runner)
    return _eval_runner


def _register_all_evals(runner: EvalRunner) -> None:
    """Register all built-in eval types."""
    from observability.evals.types.fact_accuracy import eval_fact_accuracy
    from observability.evals.types.recall_quality import eval_recall_quality, eval_recall_coverage
    from observability.evals.types.coherence import eval_coherence
    from observability.evals.types.learning_drift import eval_learning_drift

    runner.register("fact_accuracy", eval_fact_accuracy)
    runner.register("recall_precision", eval_recall_quality)
    runner.register("recall_coverage", eval_recall_coverage)
    runner.register("coherence", eval_coherence)
    runner.register("learning_drift", eval_learning_drift)

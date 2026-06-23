"""
Learning Drift Eval

Monitor if learned facts are diverging from ground truth over time.
Compares current knowledge against baseline snapshots.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from observability.evals.runner import EvalResult


async def eval_learning_drift(
    sfm=None,
    baseline_snapshot: Optional[dict[str, Any]] = None,
    sample_size: int = 100,
) -> EvalResult:
    """
    Evaluate learning drift by comparing current facts to baseline.
    
    Drift types:
    - Value drift: Fact values have changed
    - Confidence drift: Confidence scores have changed significantly
    - Missing facts: Facts from baseline no longer exist
    - New contradictions: New facts contradict baseline
    
    Args:
        sfm: Short-form memory instance.
        baseline_snapshot: Baseline facts to compare against.
        sample_size: Number of facts to sample.
        
    Returns:
        EvalResult with drift score.
    """
    if baseline_snapshot is None:
        baseline_snapshot = await _load_baseline_snapshot()
    
    if not baseline_snapshot:
        return EvalResult(
            eval_type="learning_drift",
            score=1.0,
            sample_size=0,
            passed=0,
            failed=0,
            details={"message": "No baseline snapshot available"},
        )
    
    # Get current facts
    current_facts = await _get_current_facts(sfm, sample_size)
    
    if not current_facts:
        return EvalResult(
            eval_type="learning_drift",
            score=0.0,
            sample_size=0,
            passed=0,
            failed=0,
            details={"message": "No current facts to evaluate"},
        )
    
    # Compare against baseline
    drift_analysis = _analyze_drift(current_facts, baseline_snapshot)
    
    # Calculate drift score (1.0 = no drift, 0.0 = complete drift)
    total_checked = drift_analysis["total_checked"]
    total_drifted = (
        drift_analysis["value_drift_count"] +
        drift_analysis["missing_count"] +
        drift_analysis["contradiction_count"]
    )
    
    score = 1.0 - (total_drifted / total_checked) if total_checked > 0 else 1.0
    
    passed = total_checked - total_drifted
    failed = total_drifted
    
    return EvalResult(
        eval_type="learning_drift",
        score=score,
        sample_size=total_checked,
        passed=passed,
        failed=failed,
        details=drift_analysis,
    )


async def _load_baseline_snapshot() -> dict[str, Any]:
    """
    Load baseline snapshot from storage.
    
    Returns:
        Dict mapping fact_id to fact data.
    """
    # Would load from evals/datasets/baseline_snapshot.json
    # or from a previous eval run stored in DuckDB
    return {}


async def _get_current_facts(
    sfm,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Get current facts from SFM."""
    if sfm is None:
        return []
    
    try:
        # Would call sfm.sample() in production
        return []
    except Exception:
        return []


def _analyze_drift(
    current_facts: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze drift between current facts and baseline.
    
    Args:
        current_facts: Current facts from SFM.
        baseline: Baseline snapshot.
        
    Returns:
        Drift analysis dict.
    """
    value_drifts = []
    confidence_drifts = []
    missing = []
    
    current_by_id = {f.get("id"): f for f in current_facts}
    baseline_facts = baseline.get("facts", {})
    
    # Check each baseline fact
    for fact_id, baseline_fact in baseline_facts.items():
        current_fact = current_by_id.get(fact_id)
        
        if current_fact is None:
            missing.append(fact_id)
            continue
        
        # Check value drift
        if _has_value_drift(current_fact, baseline_fact):
            value_drifts.append({
                "fact_id": fact_id,
                "baseline": baseline_fact,
                "current": current_fact,
            })
        
        # Check confidence drift
        conf_drift = _confidence_drift(current_fact, baseline_fact)
        if abs(conf_drift) > 0.2:  # >20% change
            confidence_drifts.append({
                "fact_id": fact_id,
                "drift": conf_drift,
            })
    
    return {
        "total_checked": len(baseline_facts),
        "value_drift_count": len(value_drifts),
        "confidence_drift_count": len(confidence_drifts),
        "missing_count": len(missing),
        "contradiction_count": 0,  # Would use NLI to detect
        "value_drifts": value_drifts[:10],
        "confidence_drifts": confidence_drifts[:10],
        "missing": missing[:10],
    }


def _has_value_drift(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    """Check if fact value has drifted."""
    return (
        current.get("subject") != baseline.get("subject") or
        current.get("predicate") != baseline.get("predicate") or
        current.get("object") != baseline.get("object")
    )


def _confidence_drift(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> float:
    """Calculate confidence drift."""
    current_conf = current.get("confidence", 0.8)
    baseline_conf = baseline.get("confidence", 0.8)
    return current_conf - baseline_conf


async def create_baseline_snapshot(sfm) -> dict[str, Any]:
    """
    Create a baseline snapshot from current SFM state.
    
    Args:
        sfm: Short-form memory instance.
        
    Returns:
        Baseline snapshot dict.
    """
    facts = await _get_current_facts(sfm, sample_size=1000)
    
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fact_count": len(facts),
        "facts": {f.get("id"): f for f in facts if f.get("id")},
    }

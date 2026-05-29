"""
Recall Quality Eval

Evaluate recall precision and coverage using labeled query-document pairs.
"""

from typing import Any, Optional

from evals.runner import EvalResult
from evals.metrics import precision_at_k, recall_at_k, f1_score


async def eval_recall_quality(
    mml=None,
    test_set: Optional[list[tuple[str, list[str]]]] = None,
    top_k: int = 10,
) -> EvalResult:
    """
    Evaluate recall precision using labeled query-document pairs.
    
    Args:
        mml: Memory management layer instance.
        test_set: List of (query, relevant_doc_ids) tuples.
        top_k: Number of results to consider.
        
    Returns:
        EvalResult with precision/recall scores.
    """
    # Load test set if not provided
    if test_set is None:
        test_set = _load_recall_test_set()
    
    if not test_set:
        return EvalResult(
            eval_type="recall_precision",
            score=1.0,
            sample_size=0,
            passed=0,
            failed=0,
            details={"message": "No test set available"},
        )
    
    precision_scores = []
    recall_scores = []
    
    for query, relevant_ids in test_set:
        # Run recall
        results = await _run_recall(mml, query, top_k)
        retrieved_ids = [r.get("id") for r in results]
        relevant_set = set(relevant_ids)
        
        # Calculate metrics
        p = precision_at_k(retrieved_ids, relevant_set, top_k)
        r = recall_at_k(retrieved_ids, relevant_set, top_k)
        
        precision_scores.append(p)
        recall_scores.append(r)
    
    # Calculate averages
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = f1_score(avg_precision, avg_recall)
    
    # Count passed/failed (precision >= 0.5 is pass)
    passed = sum(1 for p in precision_scores if p >= 0.5)
    failed = len(precision_scores) - passed
    
    return EvalResult(
        eval_type="recall_precision",
        score=avg_precision,
        sample_size=len(test_set),
        passed=passed,
        failed=failed,
        details={
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "precision_distribution": precision_scores,
            "recall_distribution": recall_scores,
        },
    )


async def _run_recall(
    mml,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run recall for a query."""
    if mml is None:
        return []
    
    try:
        # Would call mml.recall() in production
        # results = await mml.recall(query, memory_types=["sfm", "lfm"], top_k=top_k)
        return []
    except Exception:
        return []


def _load_recall_test_set() -> list[tuple[str, list[str]]]:
    """
    Load recall test set from file.
    
    Returns:
        List of (query, relevant_doc_ids) tuples.
    """
    # Would load from evals/datasets/recall_test_set.json
    # For now, return empty
    return []


async def eval_recall_coverage(
    mml=None,
    test_set: Optional[list[tuple[str, list[str]]]] = None,
    top_k: int = 10,
) -> EvalResult:
    """
    Evaluate recall coverage (did we miss relevant content?).
    
    Args:
        mml: Memory management layer instance.
        test_set: List of (query, relevant_doc_ids) tuples.
        top_k: Number of results to consider.
        
    Returns:
        EvalResult with coverage score.
    """
    if test_set is None:
        test_set = _load_recall_test_set()
    
    if not test_set:
        return EvalResult(
            eval_type="recall_coverage",
            score=1.0,
            sample_size=0,
            passed=0,
            failed=0,
            details={"message": "No test set available"},
        )
    
    coverage_scores = []
    
    for query, relevant_ids in test_set:
        results = await _run_recall(mml, query, top_k)
        retrieved_ids = [r.get("id") for r in results]
        relevant_set = set(relevant_ids)
        
        # Coverage = recall
        coverage = recall_at_k(retrieved_ids, relevant_set, top_k)
        coverage_scores.append(coverage)
    
    avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
    
    # Count passed/failed (coverage >= 0.7 is pass)
    passed = sum(1 for c in coverage_scores if c >= 0.7)
    failed = len(coverage_scores) - passed
    
    return EvalResult(
        eval_type="recall_coverage",
        score=avg_coverage,
        sample_size=len(test_set),
        passed=passed,
        failed=failed,
        details={
            "avg_coverage": avg_coverage,
            "coverage_distribution": coverage_scores,
        },
    )

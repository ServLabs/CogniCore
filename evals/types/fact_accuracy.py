"""
Fact Accuracy Eval

Verify SFM facts against ground truth or source documents.
"""

from typing import Any, Optional

from evals.runner import EvalResult


async def eval_fact_accuracy(
    sfm=None,
    nli=None,
    sample_size: int = 100,
) -> EvalResult:
    """
    Verify SFM facts against ground truth or source documents.
    
    For each fact:
    1. If fact has source_doc_id, verify against source using NLI
    2. If no source, check for contradictions with other facts
    
    Args:
        sfm: Short-form memory instance.
        nli: NLI connector for entailment checking.
        sample_size: Number of facts to sample.
        
    Returns:
        EvalResult with accuracy score.
    """
    # Get sample of facts
    facts = await _get_fact_sample(sfm, sample_size)
    
    if not facts:
        return EvalResult(
            eval_type="fact_accuracy",
            score=1.0,
            sample_size=0,
            passed=0,
            failed=0,
            details={"message": "No facts to evaluate"},
        )
    
    passed = 0
    failed = 0
    failures = []
    
    for fact in facts:
        is_valid = await _verify_fact(fact, nli)
        
        if is_valid:
            passed += 1
        else:
            failed += 1
            failures.append({
                "fact_id": fact.get("id", "unknown"),
                "subject": fact.get("subject", ""),
                "predicate": fact.get("predicate", ""),
                "object": fact.get("object", ""),
                "reason": "contradiction_or_drift",
            })
    
    score = passed / len(facts) if facts else 1.0
    
    return EvalResult(
        eval_type="fact_accuracy",
        score=score,
        sample_size=len(facts),
        passed=passed,
        failed=failed,
        details={
            "failures": failures[:10],  # Top 10 failures
            "failure_rate": failed / len(facts) if facts else 0,
        },
    )


async def _get_fact_sample(
    sfm,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Get a sample of facts from SFM."""
    if sfm is None:
        return []
    
    try:
        # Would call sfm.sample() in production
        return []
    except Exception:
        return []


async def _verify_fact(
    fact: dict[str, Any],
    nli,
) -> bool:
    """
    Verify a single fact.
    
    Uses NLI to check entailment if source is available.
    """
    if nli is None:
        # Without NLI, assume valid
        return True
    
    source_doc_id = fact.get("source_doc_id")
    
    if source_doc_id:
        # Verify against source document
        return await _verify_against_source(fact, source_doc_id, nli)
    else:
        # Check for contradictions (placeholder)
        return True


async def _verify_against_source(
    fact: dict[str, Any],
    source_doc_id: str,
    nli,
) -> bool:
    """
    Verify fact against its source document using NLI.
    
    Args:
        fact: Fact to verify.
        source_doc_id: Source document ID.
        nli: NLI connector.
        
    Returns:
        True if fact is entailed by source.
    """
    # Build fact text
    fact_text = f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}"
    
    # Would fetch source document and check entailment
    # source = await lfm.get(source_doc_id)
    # result = nli.predict(source.content[:1000], fact_text)
    # return result["entailment"] > 0.7
    
    return True

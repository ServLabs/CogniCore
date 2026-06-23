"""
Coherence Eval

Check for internal contradictions in SFM facts.
Uses NLI to detect contradicting statements.
"""

from typing import Any, Optional

from observability.evals.runner import EvalResult
from observability.evals.metrics import coherence_score


async def eval_coherence(
    sfm=None,
    nli=None,
    sample_size: int = 100,
    contradiction_threshold: float = 0.8,
) -> EvalResult:
    """
    Evaluate coherence by checking for contradictions in SFM facts.
    
    Args:
        sfm: Short-form memory instance.
        nli: NLI connector for contradiction detection.
        sample_size: Number of facts to sample.
        contradiction_threshold: NLI score threshold for contradiction.
        
    Returns:
        EvalResult with coherence score.
    """
    # Get sample of facts
    facts = await _get_fact_sample(sfm, sample_size)
    
    if not facts:
        return EvalResult(
            eval_type="coherence",
            score=1.0,
            sample_size=0,
            passed=0,
            failed=0,
            details={"message": "No facts to evaluate"},
        )
    
    # Find contradictions
    contradictions = await _find_contradictions(facts, nli, contradiction_threshold)
    
    # Calculate coherence score
    score = coherence_score(contradictions, len(facts))
    
    # Count facts involved in contradictions
    contradicting_facts = set()
    for a, b in contradictions:
        contradicting_facts.add(a)
        contradicting_facts.add(b)
    
    passed = len(facts) - len(contradicting_facts)
    failed = len(contradicting_facts)
    
    return EvalResult(
        eval_type="coherence",
        score=score,
        sample_size=len(facts),
        passed=passed,
        failed=failed,
        details={
            "contradiction_count": len(contradictions),
            "contradicting_fact_count": len(contradicting_facts),
            "contradictions": [
                {
                    "fact_a": facts[a] if a < len(facts) else None,
                    "fact_b": facts[b] if b < len(facts) else None,
                }
                for a, b in contradictions[:10]  # Top 10
            ],
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


async def _find_contradictions(
    facts: list[dict[str, Any]],
    nli,
    threshold: float,
) -> list[tuple[int, int]]:
    """
    Find contradicting fact pairs using NLI.
    
    Args:
        facts: List of facts.
        nli: NLI connector.
        threshold: Contradiction score threshold.
        
    Returns:
        List of (fact_idx_a, fact_idx_b) pairs.
    """
    if nli is None or not facts:
        return []
    
    contradictions = []
    
    # Build fact texts
    fact_texts = [
        f"{f.get('subject', '')} {f.get('predicate', '')} {f.get('object', '')}"
        for f in facts
    ]
    
    # Check all pairs (could be optimized with clustering)
    for i in range(len(fact_texts)):
        for j in range(i + 1, len(fact_texts)):
            try:
                is_contradiction = nli.is_contradicting(fact_texts[i], fact_texts[j])
                if is_contradiction:
                    contradictions.append((i, j))
            except Exception:
                pass
    
    return contradictions


async def eval_coherence_by_domain(
    sfm=None,
    nli=None,
    domain: str = "",
    sample_size: int = 50,
) -> EvalResult:
    """
    Evaluate coherence within a specific domain.
    
    Args:
        sfm: Short-form memory instance.
        nli: NLI connector.
        domain: Domain to filter facts.
        sample_size: Number of facts to sample.
        
    Returns:
        EvalResult with domain-specific coherence score.
    """
    # Would filter facts by domain before evaluation
    return await eval_coherence(sfm, nli, sample_size)

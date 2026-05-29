"""
Eval Metrics

Common metric calculations for evaluations.
"""

from typing import Any, Sequence


def precision_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any],
    k: int = 10,
) -> float:
    """
    Calculate precision@k.
    
    Precision = relevant_retrieved / total_retrieved
    
    Args:
        retrieved: List of retrieved items (in order).
        relevant: Set of relevant items.
        k: Number of top results to consider.
        
    Returns:
        Precision score (0.0 - 1.0).
    """
    if not retrieved or k <= 0:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    return relevant_in_top_k / len(top_k)


def recall_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any],
    k: int = 10,
) -> float:
    """
    Calculate recall@k.
    
    Recall = relevant_retrieved / total_relevant
    
    Args:
        retrieved: List of retrieved items (in order).
        relevant: Set of relevant items.
        k: Number of top results to consider.
        
    Returns:
        Recall score (0.0 - 1.0).
    """
    if not relevant:
        return 1.0  # No relevant items = perfect recall
    
    if not retrieved or k <= 0:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    return relevant_in_top_k / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision: Precision score.
        recall: Recall score.
        
    Returns:
        F1 score (0.0 - 1.0).
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def accuracy(
    predictions: Sequence[Any],
    ground_truth: Sequence[Any],
) -> float:
    """
    Calculate accuracy.
    
    Accuracy = correct / total
    
    Args:
        predictions: Predicted values.
        ground_truth: True values.
        
    Returns:
        Accuracy score (0.0 - 1.0).
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def mean_reciprocal_rank(
    retrieved_lists: list[Sequence[Any]],
    relevant_sets: list[set[Any]],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = average of 1/rank for first relevant item in each query.
    
    Args:
        retrieved_lists: List of retrieved item lists (one per query).
        relevant_sets: List of relevant item sets (one per query).
        
    Returns:
        MRR score (0.0 - 1.0).
    """
    if not retrieved_lists:
        return 0.0
    
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def normalized_discounted_cumulative_gain(
    retrieved: Sequence[Any],
    relevance_scores: dict[Any, float],
    k: int = 10,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        retrieved: List of retrieved items (in order).
        relevance_scores: Dict mapping items to relevance scores.
        k: Number of top results to consider.
        
    Returns:
        NDCG score (0.0 - 1.0).
    """
    import math
    
    if not retrieved or k <= 0:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        rel = relevance_scores.get(item, 0.0)
        dcg += rel / math.log2(i + 2)  # +2 because log2(1) = 0
    
    # Calculate ideal DCG
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_scores))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def coherence_score(
    contradiction_pairs: list[tuple[int, int]],
    total_facts: int,
) -> float:
    """
    Calculate coherence score based on contradictions.
    
    Coherence = 1 - (contradicting_facts / total_facts)
    
    Args:
        contradiction_pairs: List of (fact_idx_a, fact_idx_b) pairs.
        total_facts: Total number of facts.
        
    Returns:
        Coherence score (0.0 - 1.0).
    """
    if total_facts == 0:
        return 1.0
    
    # Count unique facts involved in contradictions
    contradicting_facts = set()
    for a, b in contradiction_pairs:
        contradicting_facts.add(a)
        contradicting_facts.add(b)
    
    return 1.0 - (len(contradicting_facts) / total_facts)

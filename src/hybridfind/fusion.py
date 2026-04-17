"""Rank fusion utilities."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using weighted Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(rankings)

    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (doc_idx, _score) in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

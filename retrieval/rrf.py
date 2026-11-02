"""Reciprocal Rank Fusion for combining ranked retrieval results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .bm25 import RetrievalResult


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[RetrievalResult | str]],
    *,
    k: int = 60,
    weights: Iterable[float] | None = None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Combine ranked passage lists using weighted Reciprocal Rank Fusion.

    Input ranks are one-based. Raw retrieval scores are intentionally ignored:
    RRF uses rank positions so dense and sparse scores can be combined safely.
    """

    if k < 1:
        raise ValueError("k must be at least 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided")

    ranking_lists = [list(ranking) for ranking in rankings]
    if weights is None:
        ranking_weights = [1.0] * len(ranking_lists)
    else:
        ranking_weights = list(weights)
        if len(ranking_weights) != len(ranking_lists):
            raise ValueError("weights must match the number of rankings")
        if any(weight < 0 for weight in ranking_weights):
            raise ValueError("weights must be non-negative")

    fused_scores: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(ranking_lists, ranking_weights):
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            passage_id = item.passage_id if isinstance(item, RetrievalResult) else item
            if not isinstance(passage_id, str):
                raise TypeError("ranking items must be passage IDs or RetrievalResult")
            if passage_id in seen:
                continue
            seen.add(passage_id)
            fused_scores[passage_id] += weight / (k + rank)

    fused = [
        RetrievalResult(passage_id, score)
        for passage_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda result: (-result.score, result.passage_id))
    return fused[:top_k] if top_k is not None else fused

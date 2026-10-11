from collections import defaultdict

from retrieval.types import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    top_k: int = 5,
    rrf_k: int = 60,
    source: str = "hybrid",
) -> list[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")
    if not ranked_lists:
        return []

    scores: dict[str, float] = defaultdict(float)
    results_by_id: dict[str, RetrievalResult] = {}

    for results in ranked_lists:
        for rank, hit in enumerate(results, start=1):
            scores[hit.chunk_id] += 1.0 / (rrf_k + rank)
            results_by_id[hit.chunk_id] = hit

    ranked_ids = sorted(
        scores.keys(),
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )[:top_k]

    fused: list[RetrievalResult] = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        base = results_by_id[chunk_id]
        fused.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=base.text,
                metadata=base.metadata,
                score=scores[chunk_id],
                source=source,
                chunk=base.chunk,
                rank=rank,
            )
        )
    return fused

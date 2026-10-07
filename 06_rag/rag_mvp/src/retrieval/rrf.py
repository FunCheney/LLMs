from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """
    将多路检索结果按排名融合。

    ranked_lists: {"dense": [chunk_id, ...], "sparse": [chunk_id, ...]}
    返回: [(chunk_id, rrf_score), ...] 按分数降序
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    scores: dict[str, float] = defaultdict(float)

    for chunk_ids in ranked_lists.values():
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ordered[:top_k]

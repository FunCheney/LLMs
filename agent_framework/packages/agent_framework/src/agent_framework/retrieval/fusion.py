"""多路结果的融合（W5 D2）：RRF。

为什么必须是 RRF 而不是"把分数加起来"：

- dense 的分数是余弦相似度（0–1，随向量归一化固定）；
- sparse 的分数是 BM25 原始分（无上界，随语料规模与平均文档长度浮动）；
- rerank 的分数是 cross-encoder logit（可正可负）。

三者放在一起比较，等于让**语料规模**决定排序结果：语料翻倍后 BM25 分的量级
会变，而余弦分不变，于是同一个查询的排序会莫名其妙地翻转。RRF 只看名次，
名次在所有检索器里同义。

框架侧保留一份 RRF 的理由不是"重复造轮子"，而是 multi-query 需要在**已映射为
`ScoredChunk` 之后**再融合一次：rag_mvp 的 RRF 只处理它自己的 `RetrievalResult`，
且只融两路（dense/sparse）。改写开启后是 N 路，只能在这里融。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..knowledge.base import ScoredChunk

# RRF 平滑常数。60 是原论文的经验值：它让第 1 名与第 2 名的差距
# （1/61 vs 1/62）小于"被两路同时召回"的加成（多一个 1/61），
# 所以"两路都中"比"单路第一"更值钱——这正是混合检索想要的效果。
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """按名次融合多路结果。

    - 去重键是 `chunk_id`：同一段文本从两路召回，必须合成一条，
      否则 context 里会出现两份相同内容，浪费 token 还稀释注意力。
    - 融合后 `retriever` 记录"被哪几路召回"（如 `dense+sparse`）。
      全丢成 `hybrid` 会抹掉"混合到底贡献了什么"这个归因信息。
    - `score` 是 RRF 分，**不可**与任何单路分数比较，只能用于本次排序。
    """
    if k <= 0:
        raise ValueError("k 必须大于 0，否则第一名的权重变成 1/0")

    fused: dict[str, float] = {}
    names: dict[str, list[str]] = {}
    first_seen: dict[str, ScoredChunk] = {}

    for ranking in rankings:
        for position, scored in enumerate(ranking, start=1):
            chunk_id = scored.chunk.chunk_id
            # 用列表位置而不是 scored.rank：上游实现可能没重排名次（或填了旧值），
            # 而 fusion 要的是"在这次召回里的名次"。
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + position)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = scored
            if scored.retriever not in names.setdefault(chunk_id, []):
                names[chunk_id].append(scored.retriever)

    ordered = sorted(
        fused.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        ScoredChunk(
            chunk=first_seen[chunk_id].chunk,
            score=score,
            retriever="+".join(names[chunk_id]),
            rank=rank,
        )
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]


def renumber(chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """按当前顺序重排名次。

    精排/过滤/治理都会改变顺序，但 `rank` 是被复制过来的旧值。
    不重编的话，下游拿 `rank` 做截断或展示时用的是**改动前**的名次——
    典型症状是"引用编号与排序看起来对不上"，而排查时每个环节都"没错"。
    """
    return [
        ScoredChunk(
            chunk=scored.chunk,
            score=scored.score,
            retriever=scored.retriever,
            rank=index,
        )
        for index, scored in enumerate(chunks, start=1)
    ]

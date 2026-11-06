"""检索流水线与 RRF 的契约（W5 D2/D3）。

测的是**降级语义**，不是排序算法本身：

- 有人把 RRF 写成"分数加权"，两条 score 量级不同的结果一融合就翻车；
- 有人把精排超时写成"抛错"，于是精排一卡整个请求 502；
- 有人 ACL 过滤后忘了重排名次，引用编号与排序对不上。

每条用例对应一个具体的线上后果，而不是"函数能跑"。
"""

from __future__ import annotations

import asyncio

import pytest
from agent_framework.knowledge import Chunk, ScoredChunk
from agent_framework.retrieval import (
    DEGRADE_BUDGET,
    DEGRADE_RERANK,
    DEGRADE_REWRITE,
    RetrievalPipeline,
    reciprocal_rank_fusion,
    renumber,
)


def _chunk(chunk_id: str, *, acl: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=f"正文 {chunk_id}",
        source_uri="docs/brand.md",
        metadata={} if acl is None else {"acl": acl},
    )


def _scored(chunk_id: str, *, score: float = 1.0, retriever: str = "dense") -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(chunk_id), score=score, retriever=retriever, rank=1)


class FakeRetriever:
    name = "fake"

    def __init__(
        self,
        hits: list[ScoredChunk],
        *,
        per_query: dict[str, list[ScoredChunk]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._hits = hits
        self._per_query = per_query or {}
        self._error = error
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, *, top_k: int = 10, filters=None) -> list[ScoredChunk]:
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        source = self._per_query.get(query, self._hits)
        return source[:top_k]


class FakeReranker:
    name = "fake-rerank"

    def __init__(self, *, mode: str = "ok", order: list[str] | None = None) -> None:
        self._mode = mode
        self._order = order or []
        self.calls = 0

    async def rerank(self, query, chunks, *, top_k=10):
        self.calls += 1
        if self._mode == "timeout":
            await asyncio.sleep(5)
        if self._mode == "error":
            raise RuntimeError("模型爆了")
        by_id = {s.chunk.chunk_id: s for s in chunks}
        picked = [by_id[cid] for cid in self._order if cid in by_id]
        picked += [s for s in chunks if s.chunk.chunk_id not in self._order]
        return [
            ScoredChunk(chunk=s.chunk, score=9.0 - i, retriever="rerank", rank=i + 1)
            for i, s in enumerate(picked[:top_k])
        ]


class FakeRewriter:
    name = "fake-rewrite"

    def __init__(self, queries: list[str] | None = None, *, error: Exception | None = None) -> None:
        self._queries = queries or ["查询二", "查询三"]
        self._error = error

    async def rewrite(self, query: str, *, n: int = 3) -> list[str]:
        if self._error is not None:
            raise self._error
        # 故意把原查询放在第二条：协议说第一条必须是原查询，
        # pipeline 必须自己补上，不能指望实现守规矩。
        return [self._queries[0], query, *self._queries[1:]]


# --- RRF --------------------------------------------------------------------


def test_rrf_uses_rank_not_raw_score() -> None:
    """两路分数量纲不同（余弦 0–1 vs BM25 几十），融合必须只看名次。

    若按分数加权，sparse 的第 5 名会靠"分数大"压过 dense 的第 1 名，
    而这两个分数根本没有可比性。
    """
    dense = [_scored("a", score=0.83), _scored("b", score=0.81)]
    sparse = [_scored("b", score=41.0), _scored("a", score=39.0)]

    fused = reciprocal_rank_fusion([dense, sparse])

    # b 在 dense 第 2、sparse 第 1；a 在 dense 第 1、sparse 第 2。
    # 两者名次和相同 → 分数相同，顺序由首次出现决定（稳定），
    # 关键是不能因为 sparse 把 b 排在前面就让 b 分高出一大截。
    assert {s.chunk.chunk_id for s in fused} == {"a", "b"}
    assert fused[0].score == pytest.approx(fused[1].score)


def test_rrf_prefers_hit_by_both_channels() -> None:
    """被两路同时召回 > 单路第一：这是混合检索的收益来源。"""
    dense = [
        _scored("only-dense", retriever="dense"),
        _scored("both", score=0.7, retriever="dense"),
    ]
    sparse = [
        _scored("both", score=41.0, retriever="sparse"),
        _scored("only-sparse", retriever="sparse"),
    ]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert fused[0].chunk.chunk_id == "both"
    assert fused[0].retriever == "dense+sparse"


def test_rrf_dedupes_and_records_provenance() -> None:
    """同一个 chunk 两路都中，只能留一条，且要记下是哪几路召回的。"""
    fused = reciprocal_rank_fusion(
        [[_scored("x", retriever="dense")], [_scored("x", retriever="sparse")]]
    )

    assert len(fused) == 1
    assert fused[0].retriever == "dense+sparse"
    assert fused[0].rank == 1


def test_rrf_rejects_zero_k() -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([[_scored("x")]], k=0)


def test_rrf_truncates_after_fusion() -> None:
    fused = reciprocal_rank_fusion([[_scored(f"c{i}") for i in range(5)]], top_k=2)

    assert [s.chunk.chunk_id for s in fused] == ["c0", "c1"]


def test_renumber_fixes_rank_after_resorting() -> None:
    """精排/过滤改顺序后，rank 必须跟着改，否则下游按旧名次截断。"""
    reordered = renumber([_scored("b"), _scored("a"), _scored("c")])

    assert [s.rank for s in reordered] == [1, 2, 3]
    assert reordered[0].chunk.chunk_id == "b"


# --- Pipeline ---------------------------------------------------------------


async def test_pipeline_without_reranker_keeps_recall_order() -> None:
    retriever = FakeRetriever([_scored("a"), _scored("b")])
    pipeline = RetrievalPipeline(retriever, budget_ms=500)

    outcome = await pipeline.run("品牌语气", top_k=2)

    assert [s.chunk.chunk_id for s in outcome.chunks] == ["a", "b"]
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.reranked is False
    assert outcome.diagnostics.degraded == ()


async def test_pipeline_overfetches_for_later_filtering() -> None:
    """召回要放大：后面 ACL/去重会砍，只取 top_k 会永远给不满。"""
    retriever = FakeRetriever([_scored(f"c{i}") for i in range(30)])
    pipeline = RetrievalPipeline(retriever, budget_ms=500, recall_multiplier=3.0)

    await pipeline.run("查询", top_k=5)

    assert retriever.calls[0][1] == 15


async def test_rerank_timeout_degrades_to_recall_order() -> None:
    """精排超时必须回退到召回顺序，而不是抛错——这是生产与 demo 的分水岭。"""
    retriever = FakeRetriever([_scored("a"), _scored("b")])
    pipeline = RetrievalPipeline(
        retriever,
        reranker=FakeReranker(mode="timeout"),
        budget_ms=400,
        rerank_timeout_ms=50,
    )

    outcome = await pipeline.run("查询", top_k=2)

    assert [s.chunk.chunk_id for s in outcome.chunks] == ["a", "b"]
    assert outcome.diagnostics is not None
    assert DEGRADE_RERANK in outcome.diagnostics.degraded
    assert outcome.diagnostics.reranked is False


async def test_rerank_error_degrades_to_recall_order() -> None:
    """模型报错和超时按同一套处理：结果可用比排序最优重要。"""
    retriever = FakeRetriever([_scored("a"), _scored("b")])
    pipeline = RetrievalPipeline(
        retriever, reranker=FakeReranker(mode="error"), budget_ms=400
    )

    outcome = await pipeline.run("查询", top_k=2)

    assert [s.chunk.chunk_id for s in outcome.chunks] == ["a", "b"]
    assert outcome.diagnostics is not None
    assert DEGRADE_RERANK in outcome.diagnostics.degraded


async def test_rerank_reorders_when_it_succeeds() -> None:
    retriever = FakeRetriever([_scored("a"), _scored("b"), _scored("c")])
    pipeline = RetrievalPipeline(
        retriever,
        reranker=FakeReranker(order=["c", "a", "b"]),
        budget_ms=500,
    )

    outcome = await pipeline.run("查询", top_k=3)

    assert [s.chunk.chunk_id for s in outcome.chunks] == ["c", "a", "b"]
    assert [s.retriever for s in outcome.chunks] == ["rerank"] * 3
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.reranked is True


async def test_rewrite_failure_falls_back_to_original_query() -> None:
    """改写是锦上添花：失败了要用原查询继续，不能让整个请求失败。"""
    retriever = FakeRetriever([_scored("a")])
    pipeline = RetrievalPipeline(
        retriever,
        rewriter=FakeRewriter(error=RuntimeError("网关超时")),
        budget_ms=500,
    )

    outcome = await pipeline.run("品牌语气", top_k=1)

    assert outcome.diagnostics is not None
    assert outcome.diagnostics.queries == ("品牌语气",)
    assert DEGRADE_REWRITE in outcome.diagnostics.degraded
    assert retriever.calls == [("品牌语气", 3)]


async def test_rewrite_always_keeps_original_query_first() -> None:
    """实现把原查询排在第二条时，pipeline 必须补到第一位。"""
    retriever = FakeRetriever([_scored("a")])
    pipeline = RetrievalPipeline(
        retriever, rewriter=FakeRewriter(["查询二"]), budget_ms=500
    )

    outcome = await pipeline.run("原查询", top_k=1)

    assert outcome.diagnostics is not None
    assert outcome.diagnostics.queries[0] == "原查询"
    assert len(retriever.calls) == 2


async def test_rewrite_timeout_marks_degraded_and_keeps_original() -> None:
    """改写超时（一次真实 LLM 调用的典型失败）必须降级，不能拖死请求。

    2026-11-05 实测：默认 `rewrite_timeout_ms=2500` 在本机跑 qwen 是每次超时，
    整条链路 28/28 全走降级。降级本身是对的，但**报告里必须看出来**——否则
    "改写档"测出来的数字其实等于"不改写档"，会被读成"改写没用"。
    """

    class _SlowRewriter:
        name = "slow"

        async def rewrite(self, query: str, *, n: int = 3) -> list[str]:
            await asyncio.sleep(5)
            return [query, "迟到的改写"]

    retriever = FakeRetriever([_scored("a")])
    pipeline = RetrievalPipeline(
        retriever,
        rewriter=_SlowRewriter(),
        budget_ms=5000,
        rewrite_timeout_ms=50,
    )

    outcome = await pipeline.run("原查询", top_k=1)

    assert outcome.diagnostics is not None
    assert DEGRADE_REWRITE in outcome.diagnostics.degraded
    assert outcome.diagnostics.queries == ("原查询",)
    # 改写耗时单独可读：没有这个字段，"总耗时长"只能归因到"检索慢"
    assert outcome.diagnostics.rewrite_ms >= 50
    assert [s.chunk.chunk_id for s in outcome.chunks] == ["a"]


async def test_rewrite_ms_is_zero_when_rewrite_disabled() -> None:
    """没开改写时不能记成"很快"，要记 0——否则均值里混进一批不存在的样本。"""

    pipeline = RetrievalPipeline(FakeRetriever([_scored("a")]), budget_ms=500)
    outcome = await pipeline.run("原查询", top_k=1)

    assert outcome.diagnostics is not None
    assert outcome.diagnostics.rewrite_ms == 0.0


async def test_multi_query_results_are_fused_and_deduped() -> None:
    retriever = FakeRetriever(
        [],
        per_query={
            "原查询": [_scored("a"), _scored("b", score=0.5)],
            "查询二": [_scored("b", score=0.9), _scored("c", score=0.1)],
            "查询三": [_scored("a"), _scored("c", score=0.2)],
        },
    )
    pipeline = RetrievalPipeline(
        retriever, rewriter=FakeRewriter(["查询二", "查询三"]), budget_ms=500
    )

    outcome = await pipeline.run("原查询", top_k=10)

    ids = [s.chunk.chunk_id for s in outcome.chunks]
    assert sorted(ids) == ["a", "b", "c"]
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.candidates == 3


async def test_acl_filter_drops_chunks_and_counts_without_leaking_text() -> None:
    """越权 chunk 不进结果；诊断里只留数量，不留内容。"""
    retriever = FakeRetriever([_scored("secret"), _scored("public")])
    pipeline = RetrievalPipeline(
        retriever,
        acl_check=lambda chunk: chunk.metadata.get("acl") != "private",
        budget_ms=500,
    )

    outcome = await pipeline.run("查询", top_k=2)

    # FakeRetriever 的 chunk 没有 acl 字段 → 全部放行，验证"默认放行"是显式选择
    assert len(outcome.chunks) == 2

    retriever_private = FakeRetriever(
        [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="secret",
                    document_id="d1",
                    text="机密正文",
                    source_uri="s.md",
                    metadata={"acl": "private"},
                ),
                score=1.0,
                retriever="dense",
                rank=1,
            ),
            _scored("public"),
        ]
    )
    pipeline_private = RetrievalPipeline(
        retriever_private,
        acl_check=lambda chunk: chunk.metadata.get("acl") != "private",
        budget_ms=500,
    )

    filtered = await pipeline_private.run("查询", top_k=2)

    assert [s.chunk.chunk_id for s in filtered.chunks] == ["public"]
    assert filtered.diagnostics is not None
    assert filtered.diagnostics.filtered_out == 1


async def test_budget_exhausted_skips_rerank() -> None:
    """预算耗尽时不该再尝试精排，否则延迟成倍超出配置值。"""
    retriever = FakeRetriever([_scored("a"), _scored("b")])
    pipeline = RetrievalPipeline(
        retriever,
        reranker=FakeReranker(),
        budget_ms=0.001,
    )

    outcome = await pipeline.run("查询", top_k=2)

    assert outcome.diagnostics is not None
    assert DEGRADE_BUDGET in outcome.diagnostics.degraded
    assert outcome.diagnostics.reranked is False


async def test_recall_error_propagates() -> None:
    """召回失败是系统故障，必须抛出去——伪装成空结果会被归因成"语料缺失"。"""
    pipeline = RetrievalPipeline(
        FakeRetriever([], error=FileNotFoundError("索引不存在")), budget_ms=500
    )

    with pytest.raises(FileNotFoundError):
        await pipeline.run("查询", top_k=1)


async def test_empty_recall_is_not_an_error() -> None:
    """没召回 ≠ 失败：链路正常就给空列表，让归因走检索层内容问题。"""
    pipeline = RetrievalPipeline(FakeRetriever([]), budget_ms=500)

    outcome = await pipeline.run("查询", top_k=3)

    assert outcome.chunks == []
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.ok


async def test_top_k_must_be_positive() -> None:
    pipeline = RetrievalPipeline(FakeRetriever([]), budget_ms=500)

    with pytest.raises(ValueError, match="top_k"):
        await pipeline.run("查询", top_k=0)


def test_budget_must_be_positive() -> None:
    """0 不能表示"不限"：那样一次精排就能吃满整条链路而没人发现。"""
    with pytest.raises(ValueError, match="budget_ms"):
        RetrievalPipeline(FakeRetriever([]), budget_ms=0)

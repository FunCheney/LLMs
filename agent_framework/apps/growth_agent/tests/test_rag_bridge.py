"""rag_mvp → agent_framework 桥接层的契约（W5 D1）。

这里测的不是 rag_mvp 的检索算法（那有它自己的 35 个测试），而是**接缝**：
字段改名、名次重编、线程隔离、超时语义、过滤拒绝。

接缝出错的特点是"能跑但结果悄悄变错"——引用指向检索器名、超时被当成没召回、
ACL 被静默忽略。所以每条用例都对应一个具体的错误后果。
"""

from __future__ import annotations

import asyncio
import time

import pytest
from agent_framework.retrieval import Reranker, Retriever
from rag_mvp.contrib.agent_framework import (
    AsyncReranker,
    AsyncRetriever,
    RetrievalTimeoutError,
    to_scored_chunk,
    to_scored_chunks,
)
from rag_mvp.retrieval.types import RetrievalResult
from rag_mvp.schemas import DocumentChunk


def _hit(
    *,
    chunk_id: str = "doc1-p003-c001",
    retriever_name: str = "dense",
    score: float = 0.91,
    rank: int | None = 1,
    metadata: dict | None = None,
    **chunk_overrides: object,
) -> RetrievalResult:
    fields: dict[str, object] = {
        "chunk_id": chunk_id,
        "document_id": "doc1",
        "document_title": "品牌手册",
        "section_title": "调性",
        "section_path": ["第二章", "2.1 语气"],
        "page_start": 3,
        "page_end": 4,
        "content": "品牌语气应克制、具体，避免形容词堆砌。",
        "source_uri": "docs/brand.md",
        "source_type": "md",
    }
    fields.update(chunk_overrides)
    chunk = DocumentChunk(**fields)  # type: ignore[arg-type]
    return RetrievalResult(
        chunk_id=chunk_id,
        text=chunk.content,
        metadata=dict(metadata or {"retrieval_latency_ms": 7}),
        score=score,
        source=retriever_name,
        chunk=chunk,
        rank=rank,
    )


class _SyncRetriever:
    """模拟 rag_mvp 的同步检索器（鸭子类型，不需要继承 ABC）。"""

    SOURCE = "fake"

    def __init__(
        self,
        hits: list[RetrievalResult] | None = None,
        *,
        delay_s: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._hits = hits if hits is not None else [_hit()]
        self._delay_s = delay_s
        self._error = error

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._delay_s:
            time.sleep(self._delay_s)
        if self._error is not None:
            raise self._error
        return self._hits[:top_k]


# --- 字段映射 ---------------------------------------------------------------


def test_source_becomes_retriever_not_source_uri() -> None:
    """rag_mvp 的 `source` 是检索器名，必须落到 `retriever`。

    搞反的后果：`chunk.source_uri` 变成 "[hybrid]"，W6 的引用准确率
    全部指向一个不存在的文档，而且看起来"有引用"。
    """
    scored = to_scored_chunk(_hit(retriever_name="hybrid"))

    assert scored.retriever == "hybrid"
    assert scored.chunk.source_uri == "docs/brand.md"


def test_document_fields_are_carried_over() -> None:
    scored = to_scored_chunk(_hit())

    assert scored.chunk.chunk_id == "doc1-p003-c001"
    assert scored.chunk.document_id == "doc1"
    assert scored.chunk.text == "品牌语气应克制、具体，避免形容词堆砌。"
    assert (scored.chunk.page_start, scored.chunk.page_end) == (3, 4)
    assert scored.chunk.section_path == ("第二章", "2.1 语气")
    assert scored.chunk.metadata["document_title"] == "品牌手册"
    assert scored.chunk.metadata["section_title"] == "调性"
    assert scored.chunk.metadata["source_type"] == "md"


def test_retriever_metadata_is_preserved() -> None:
    """检索器自己写进 metadata 的东西（如 prerank_score）不能丢。"""
    scored = to_scored_chunk(_hit(metadata={"prerank_score": 0.42}))

    assert scored.chunk.metadata["prerank_score"] == 0.42


def test_unknown_page_zero_becomes_none() -> None:
    """rag_mvp 用 0 表示"页码未知"，core 用 None。

    直接透传会写出"第 0 页"这种无法定位的引用，也会被下游当成有效页码。
    """
    scored = to_scored_chunk(_hit(page_start=0, page_end=0))

    assert scored.chunk.page_start is None
    assert scored.chunk.page_end is None


def test_missing_source_uri_is_flagged_not_faked() -> None:
    """缺 source_uri 时不编一个合法 URI，而是打标记。

    编造会让这类脏数据混进"引用正确"的分子；打标后 W6 会把它算作失败，
    问题才能暴露出来。
    """
    scored = to_scored_chunk(_hit(source_uri=""))

    assert scored.chunk.metadata["source_uri_missing"] is True
    assert scored.chunk.source_uri == "missing-source:doc1"


def test_rank_follows_list_position() -> None:
    """列表顺序就是实际名次；不能沿用可能为空/错位的 hit.rank。"""
    hits = [_hit(chunk_id=f"doc1-p00{i}-c001", rank=None) for i in range(1, 4)]

    scored = to_scored_chunks(hits)

    assert [s.rank for s in scored] == [1, 2, 3]


def test_single_hit_without_rank_falls_back_to_one() -> None:
    """兜底 1 而不是 0：RRF 里 0 会让第一名拿到 1/0 的权重。"""
    assert to_scored_chunk(_hit(rank=None)).rank == 1


def test_mapped_result_satisfies_core_retriever_contract() -> None:
    """映射结果必须能被 core 的协议接受（字段齐全、类型正确）。"""
    async def _run() -> list:
        return await AsyncRetriever(_SyncRetriever()).retrieve("品牌语气")

    assert isinstance(AsyncRetriever(_SyncRetriever()), Retriever)
    assert len(asyncio.run(_run())) == 1


# --- AsyncRetriever 的行为 --------------------------------------------------


async def test_empty_query_returns_empty_list() -> None:
    """空查询不是故障：返回空列表，与其他实现一致。"""
    retriever = AsyncRetriever(_SyncRetriever())
    try:
        assert await retriever.retrieve("   ") == []
    finally:
        retriever.close()


async def test_filters_are_rejected_loudly() -> None:
    """检索器层不转发 filters，必须报错而不是忽略。

    静默忽略等于把"只检索有权限的文档"变成"检索全部"——这是越权，
    而且从返回结果上看不出来。所以必须报错。

    注意断言匹配的是"尚未转发"而不是"不支持"：pgvector 后端已经能按
    metadata 过滤，缺的是 DenseRetriever → store 这段接线。两者混为一谈
    会让后来的人以为"后端没这个能力"，从而去重复实现一遍。
    """
    retriever = AsyncRetriever(_SyncRetriever())
    try:
        with pytest.raises(NotImplementedError, match="尚未转发"):
            await retriever.retrieve("品牌语气", filters={"acl": ["team-a"]})
    finally:
        retriever.close()


async def test_top_k_must_be_positive() -> None:
    retriever = AsyncRetriever(_SyncRetriever())
    try:
        with pytest.raises(ValueError, match="top_k"):
            await retriever.retrieve("品牌语气", top_k=0)
    finally:
        retriever.close()


async def test_timeout_raises_instead_of_returning_empty() -> None:
    """超时必须抛错。

    返回空列表会被 W4 归因当作"链路正常但没召回"，于是人跑去补语料、
    改查询，而真实原因是精排卡住。这个误判每次要烧半天。
    """
    retriever = AsyncRetriever(_SyncRetriever(delay_s=0.5), timeout_s=0.05)
    try:
        with pytest.raises(RetrievalTimeoutError, match="0.05s"):
            await retriever.retrieve("品牌语气")
    finally:
        retriever.close()


async def test_blocking_retriever_runs_off_the_event_loop() -> None:
    """两个并发的阻塞检索应当并行，而不是串行。

    串行说明代码在事件循环里直接调了同步函数——那时 p95 会随并发数线性恶化，
    而且 CPU 明明闲着也加不上去。这里 2 路 × 0.2s：并行约 0.2s，串行 0.4s。
    """
    retriever = AsyncRetriever(_SyncRetriever(delay_s=0.2))
    try:
        started = time.perf_counter()
        await asyncio.gather(
            retriever.retrieve("查询一"),
            retriever.retrieve("查询二"),
        )
        elapsed = time.perf_counter() - started
    finally:
        retriever.close()

    assert elapsed < 0.35, f"看起来在事件循环里串行执行了（耗时 {elapsed:.2f}s）"


async def test_inner_error_is_not_swallowed() -> None:
    """索引缺失这类错误要透出去，不能变成"没召回"。"""
    retriever = AsyncRetriever(_SyncRetriever(error=FileNotFoundError("索引不存在")))
    try:
        with pytest.raises(FileNotFoundError):
            await retriever.retrieve("品牌语气")
    finally:
        retriever.close()


async def test_name_defaults_to_inner_source() -> None:
    retriever = AsyncRetriever(_SyncRetriever())
    try:
        assert retriever.name == "fake"
    finally:
        retriever.close()


# --- AsyncReranker：精排必须能被单独持有 ------------------------------------


class _SyncReranker:
    """模拟 rag_mvp 的 BGEReranker（鸭子类型）。

    真实实现会修改传入的 hit（score/rank）并返回同一个对象列表，
    这里必须保持同样的行为——桥接层依赖"返回的是原对象"来保留 chunk。
    """

    def __init__(self, *, order: list[str] | None = None) -> None:
        self._order = order or []
        self.seen: list[list[str]] = []

    def rerank(self, query: str, hits, top_k=None):
        self.seen.append([hit.chunk_id for hit in hits])
        by_id = {hit.chunk_id: hit for hit in hits}
        picked = [by_id[cid] for cid in self._order if cid in by_id]
        picked += [hit for hit in hits if hit.chunk_id not in self._order]
        for rank, hit in enumerate(picked, start=1):
            hit.metadata["prerank_score"] = hit.score
            hit.score = 10.0 - rank
            hit.rank = rank
        return picked[:top_k] if top_k is not None else picked


async def test_reranker_maps_back_to_scored_chunks() -> None:
    """精排结果要回到 core 类型，且名次/来源标记正确。

    `retriever` 必须变成 `rerank`：否则归因时分不清"混合召回的"和
    "被精排抬上来的"，也就无法回答"精排到底有没有用"。
    """
    scored = to_scored_chunks([_hit(chunk_id="c1"), _hit(chunk_id="c2")])
    reranker = AsyncReranker(_SyncReranker(order=["c2", "c1"]))

    try:
        assert isinstance(reranker, Reranker)
        out = await reranker.rerank("品牌语气", scored, top_k=2)
    finally:
        reranker.close()

    assert [s.chunk.chunk_id for s in out] == ["c2", "c1"]
    assert [s.rank for s in out] == [1, 2]
    assert all(s.retriever == "rerank" for s in out)


async def test_reranker_preserves_chunk_provenance() -> None:
    """精排只能改顺序与分数，不能把 source_uri / 页码改掉。"""
    scored = to_scored_chunks([_hit()])
    reranker = AsyncReranker(_SyncReranker())

    try:
        out = await reranker.rerank("品牌语气", scored, top_k=1)
    finally:
        reranker.close()

    assert out[0].chunk.source_uri == "docs/brand.md"
    assert (out[0].chunk.page_start, out[0].chunk.page_end) == (3, 4)
    assert out[0].chunk.section_path == ("第二章", "2.1 语气")


async def test_reranker_round_trip_fills_unknown_pages_with_zero() -> None:
    """core 用 None 表示页码未知，rag_mvp 侧必须变回 0。

    直接传 None 会让 rag_mvp 的格式化输出 "None-None"，而它的类型契约是 int。
    """
    scored = to_scored_chunks([_hit(page_start=0, page_end=0)])
    inner = _SyncReranker()
    reranker = AsyncReranker(inner)

    try:
        await reranker.rerank("品牌语气", scored, top_k=1)
    finally:
        reranker.close()

    assert scored[0].chunk.page_start is None
    assert len(inner.seen) == 1


async def test_reranker_empty_input_short_circuits() -> None:
    inner = _SyncReranker()
    reranker = AsyncReranker(inner)

    try:
        assert await reranker.rerank("品牌语气", []) == []
    finally:
        reranker.close()

    assert inner.seen == []

"""知识层与检索层的契约不变量（W5 D1）。

这些用例锁的是**以后改不动**的部分（字段语义、协议形状），不是当前实现细节。
每条失败都对应一次真实的踩坑，注释里写了是哪个。
"""

from __future__ import annotations

import inspect

import pytest
from agent_framework.knowledge import Chunk, EmbeddingProvider, ScoredChunk, VectorStore
from agent_framework.retrieval import QueryRewriter, Reranker, Retriever
from pydantic import ValidationError


def _chunk(**kwargs: object) -> Chunk:
    base: dict[str, object] = {
        "chunk_id": "doc1-p001-c001",
        "document_id": "doc1",
        "text": "检索增强生成把外部证据带进上下文",
        "source_uri": "docs/rag.md",
    }
    base.update(kwargs)
    return Chunk(**base)  # type: ignore[arg-type]


class _DummyRetriever:
    @property
    def name(self) -> str:
        return "dummy"

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: object = None,
    ) -> list[ScoredChunk]:
        return []


class _DummyEmbedder:
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def dimension(self) -> int:
        return 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]


# --- Chunk 的字段语义 -------------------------------------------------------


def test_chunk_is_immutable() -> None:
    """Chunk 在检索链路上被多处共享，可变会导致"下游改了上游的引用出处"。"""
    chunk = _chunk()
    with pytest.raises(ValidationError):
        chunk.text = "被改掉了"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["chunk_id", "document_id", "source_uri"])
def test_blank_identifiers_rejected(field: str) -> None:
    """空值不是"暂时没填"：引用会指向空字符串，删除传播也找不到目标。"""
    with pytest.raises(ValidationError):
        _chunk(**{field: "   "})


def test_page_end_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _chunk(page_end=0)


def test_cross_page_chunk_keeps_both_pages() -> None:
    """跨页块必须保留起止页，引用才能写成"第 3–4 页"。"""
    chunk = _chunk(page_start=3, page_end=4)
    assert (chunk.page_start, chunk.page_end) == (3, 4)


# --- 两个"source"必须分开 ---------------------------------------------------


def test_source_uri_and_retriever_are_distinct_fields() -> None:
    """rag_mvp 用 `source` 一个字段表达两件事：文档出处 + 检索器名。

    直接搬过来时，检索器的名字会覆盖文档出处，引用变成 `"[hybrid]"`。
    这条用例锁住"它们是两个字段"。
    """
    chunk = _chunk(source_uri="docs/rag.md")
    scored = ScoredChunk(chunk=chunk, score=0.8, retriever="hybrid", rank=1)

    assert scored.chunk.source_uri == "docs/rag.md"
    assert scored.retriever == "hybrid"
    assert scored.chunk.source_uri != scored.retriever


def test_rank_is_one_based() -> None:
    """RRF 直接用名次计算，0-based 会让第一名拿到 1/0 的权重。"""
    with pytest.raises(ValidationError):
        ScoredChunk(chunk=_chunk(), score=0.5, retriever="dense", rank=0)


def test_scored_chunk_requires_retriever() -> None:
    """同一个 chunk 会被多个检索器召回，不记来源就看不出混合召回是否真贡献了新命中。"""
    with pytest.raises(ValidationError):
        ScoredChunk(chunk=_chunk(), score=0.5, rank=1)  # type: ignore[call-arg]


# --- 协议形状 ---------------------------------------------------------------


def test_retriever_protocol_is_runtime_checkable() -> None:
    assert isinstance(_DummyRetriever(), Retriever)


def test_partial_implementation_is_not_a_retriever() -> None:
    class _OnlyName:
        @property
        def name(self) -> str:
            return "half"

    assert not isinstance(_OnlyName(), Retriever)


def test_retriever_contract_is_async() -> None:
    """协议只保证"可以 await"：实现写同步方法会把阻塞带进事件循环。"""
    assert inspect.iscoroutinefunction(_DummyRetriever.retrieve)


def test_embedding_provider_protocol() -> None:
    assert isinstance(_DummyEmbedder(), EmbeddingProvider)


def test_embedding_provider_exposes_dimension() -> None:
    """维度必须是协议的一部分：靠 len(vectors[0]) 猜，遇到空输入就 IndexError。"""
    assert _DummyEmbedder().dimension == 3


async def test_empty_recall_returns_empty_list_not_none() -> None:
    """空结果 ≠ 失败。

    返回 [] 表示"链路正常但没召回"（内容问题），抛异常表示"链路坏了"
    （系统问题）。两者混成 None，W4 归因会把 rerank 超时误判成语料缺失。
    """
    result = await _DummyRetriever().retrieve("一个查不到的问题")
    assert result == []


# --- 向量库必须能删 ---------------------------------------------------------


def test_vector_store_requires_delete() -> None:
    """没有 delete 就只能全量重建，增量索引的意义被抹掉（W6）。"""

    class _NoDelete:
        @property
        def name(self) -> str:
            return "no-delete"

        async def upsert(
            self, chunks: object, vectors: object, *, metadata: object = None
        ) -> int:
            return 0

        async def search(
            self, vector: object, *, top_k: int, filters: object = None
        ) -> list[ScoredChunk]:
            return []

    assert not isinstance(_NoDelete(), VectorStore)


def test_full_vector_store_satisfies_protocol() -> None:
    class _Store:
        @property
        def name(self) -> str:
            return "memory"

        async def upsert(
            self, chunks: object, vectors: object, *, metadata: object = None
        ) -> int:
            return 0

        async def search(
            self, vector: object, *, top_k: int, filters: object = None
        ) -> list[ScoredChunk]:
            return []

        async def delete(self, *, document_id: str) -> int:
            return 0

    assert isinstance(_Store(), VectorStore)


# --- 其余两条协议 -----------------------------------------------------------


def test_reranker_protocol_is_runtime_checkable() -> None:
    class _Reranker:
        @property
        def name(self) -> str:
            return "cross-encoder"

        async def rerank(
            self, query: str, chunks: list[ScoredChunk], *, top_k: int = 10
        ) -> list[ScoredChunk]:
            return chunks[:top_k]

    assert isinstance(_Reranker(), Reranker)


def test_query_rewriter_protocol_is_runtime_checkable() -> None:
    class _Rewriter:
        @property
        def name(self) -> str:
            return "multi-query"

        async def rewrite(self, query: str, *, n: int = 3) -> list[str]:
            return [query]

    assert isinstance(_Rewriter(), QueryRewriter)

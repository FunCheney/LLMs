"""pgvector 后端接到框架检索链路上的端到端测试。

这是"unit test 都绿但一上线就挂"的那类接线验证：单独的 store 测试只证明 SQL
写得对，`RetrievalPipeline → AsyncRetriever → DenseRetriever → PgVectorStore`
任何一环的字段名对不上，都要等到真实请求才暴露。

不需要 torch：用 `FakeEmbeddingProvider` 生成确定性向量。默认跳过，
跑法见 `06_rag/rag_mvp/test/test_pgvector_integration.py` 的 docstring。
"""

from __future__ import annotations

import os
import uuid

import pytest
from agent_framework.retrieval import RetrievalPipeline
from rag_mvp.contrib.agent_framework import AsyncRetriever
from rag_mvp.embedding.fake import FakeEmbeddingProvider
from rag_mvp.retrieval.retriever import DenseRetriever
from rag_mvp.vector_store.pgvector import PgVectorStore
from rag_mvp.vector_store.vector_store_types import VectorRecord

DSN = os.getenv("PGVECTOR_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="未配置 PGVECTOR_DSN，跳过真库测试")

# 维度取 64 而不是 4：假向量按 `(char_sum % 97 + i) % dim` 生成，
# 维度太低时不同文本会撞成同一个向量，"该命中哪条"就无从断言了。
DIM = 64
TABLE = "rag_chunks_e2e"


@pytest.fixture()
def seeded() -> tuple[PgVectorStore, FakeEmbeddingProvider]:
    embedder = FakeEmbeddingProvider(dimension=DIM)
    store = PgVectorStore(DSN, table=TABLE, dimension=DIM)
    store.ensure_schema()
    store.drop()

    for index, text in enumerate(
        [
            "品牌语气应克制，避免夸张的形容词堆叠。",
            "关键词布局优先覆盖长尾词，再逐周向核心词靠拢。",
            "社媒发帖节奏按渠道分层：高频短内容 + 周更长文。",
        ]
    ):
        store.upsert(
            VectorRecord(
                id=f"c{index}",
                text=text,
                metadata={
                    "chunk_id": f"c{index}",
                    "document_id": f"doc{index}",
                    "source_uri": f"s3://kb/doc{index}.md",
                    "section_title": "章节",
                    "page_start": 3,
                },
                embedding=embedder.embed_text(text).vector,
            )
        )
    try:
        yield store, embedder
    finally:
        store._connection().execute(f"DROP TABLE IF EXISTS {TABLE}")
        store.close()


async def test_pipeline_retrieves_from_pgvector(
    seeded: tuple[PgVectorStore, FakeEmbeddingProvider],
) -> None:
    """整条链路的冒烟：能查出结果、字段没串、诊断信息齐全。"""
    store, embedder = seeded
    pipeline = RetrievalPipeline(
        AsyncRetriever(DenseRetriever(store=store, embedder=embedder), name="pg-dense"),
        budget_ms=2000,
    )

    outcome = await pipeline.run("品牌语气怎么定", top_k=2)

    assert len(outcome.chunks) == 2
    top = outcome.chunks[0]
    # document_id / source_uri 必须在穿透四层后仍然是"文档出处"，而不是检索器名
    assert top.chunk.document_id.startswith("doc")
    assert top.chunk.source_uri.startswith("s3://kb/")
    assert top.retriever == "dense"
    assert top.chunk.page_start == 3
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.recall_ms > 0


async def test_exact_text_ranks_first(
    seeded: tuple[PgVectorStore, FakeEmbeddingProvider],
) -> None:
    """用某条 chunk 的原文当查询，它必须是第一名。

    这条断言检的是 pgvector 的 `<=>` 方向：写反成"距离升序当成相关度升序"时，
    结果永远是最不相关的那条，而分数看起来仍然正常，非常难发现。
    """
    store, embedder = seeded
    pipeline = RetrievalPipeline(
        AsyncRetriever(DenseRetriever(store=store, embedder=embedder), name="pg-dense"),
        budget_ms=2000,
    )
    target = "关键词布局优先覆盖长尾词，再逐周向核心词靠拢。"

    outcome = await pipeline.run(target, top_k=3)

    assert outcome.chunks[0].chunk.text == target
    scores = [chunk.score for chunk in outcome.chunks]
    assert scores == sorted(scores, reverse=True)


async def test_pipeline_honours_top_k_across_backend(
    seeded: tuple[PgVectorStore, FakeEmbeddingProvider],
) -> None:
    """`top_k` 必须在 SQL 里生效，而不是取回全部再在 Python 里截断。

    取回全部再截断在 3 条数据上看不出差别，到 100 万条时就是"检索一次传 100 万行"。
    """
    store, embedder = seeded
    pipeline = RetrievalPipeline(
        AsyncRetriever(DenseRetriever(store=store, embedder=embedder), name="pg-dense"),
        budget_ms=2000,
    )

    outcome = await pipeline.run("发帖节奏", top_k=1)

    assert len(outcome.chunks) == 1


async def test_upsert_updates_visible_result(
    seeded: tuple[PgVectorStore, FakeEmbeddingProvider],
) -> None:
    """改一条正文后重新检索必须看到新内容——索引没有"缓存旧值"的余地。"""
    store, embedder = seeded
    new_text = "关键词布局改为按购买意图分层，先做对比词。"
    store.upsert(
        VectorRecord(
            id="c1",
            text=new_text,
            metadata={
                "chunk_id": "c1",
                "document_id": "doc1",
                "source_uri": "s3://kb/doc1.md",
            },
            embedding=embedder.embed_text(new_text).vector,
        )
    )
    pipeline = RetrievalPipeline(
        AsyncRetriever(DenseRetriever(store=store, embedder=embedder), name="pg-dense"),
        budget_ms=2000,
    )

    outcome = await pipeline.run(new_text, top_k=1)

    assert outcome.chunks[0].chunk.text == new_text


async def test_empty_corpus_returns_empty_not_error(
    seeded: tuple[PgVectorStore, FakeEmbeddingProvider],
) -> None:
    """空结果 ≠ 失败：语料为空时返回空列表，让 W4 的归因把它算成"语料缺失"。"""
    store, embedder = seeded
    store.drop()
    pipeline = RetrievalPipeline(
        AsyncRetriever(DenseRetriever(store=store, embedder=embedder), name="pg-dense"),
        budget_ms=2000,
    )

    outcome = await pipeline.run("任意问题", top_k=5)

    assert outcome.chunks == []
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.ok


def test_table_name_is_validated_before_sql() -> None:
    """表名会被拼进 SQL，必须在构造期挡住注入式输入。"""
    with pytest.raises(ValueError, match="表名"):
        PgVectorStore(DSN, table=f"x; DROP TABLE {uuid.uuid4().hex}", ensure_schema=False)

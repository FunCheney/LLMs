"""增量索引在真库上的验收（W6 D2 / Gate 6 #3 #4）。

单测用内存 store 证明逻辑对；这一组证明**SQL 与事务在 PostgreSQL 上也对**。
两者缺一不可：`replace_document` 的删旧写新如果不在同一事务里，内存 store 永远
发现不了——它的"删"和"写"是同一行 Python 执行下来的。

验收四件事（对应任务书 D2 自检）：

1. 新增一份文档 → 另一份文档的 chunk 数与向量**没变**（不是全量重建）；
2. 内容未变再同步 → `SKIPPED`，且 embedder **一次都没被调用**（成本真的省了）；
3. 内容变了 → 旧 chunk 全部消失、新 chunk 全部写入，不留半新半旧；
4. 删除 → 该文档清零，重复删除幂等；别的文档不动。

外加两条 W6 特有的：`model_snapshot` 变了必须重建；切分结果 document_id 写错必须拒绝
（否则删除传播会漏删）。

默认跳过（不配 `PGVECTOR_DSN` 时）。跑法：

    cd ops && docker compose up -d pgvector
    cd .. && PGVECTOR_DSN=postgresql://rag:rag@localhost:5433/rag \
      .venv/bin/python -m pytest apps/growth_agent/tests/test_incremental_e2e.py -q -o addopts=""
"""

from __future__ import annotations

import os

import pytest
from agent_framework.knowledge import (
    Chunk,
    IncrementalIndexer,
    SyncAction,
    content_hash,
)
from rag_mvp.contrib.agent_framework import AsyncDocumentStore
from rag_mvp.vector_store.pgvector import PgVectorStore

DSN = os.getenv("PGVECTOR_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="未配置 PGVECTOR_DSN，跳过真库测试")

TABLE = "rag_chunks_incremental_e2e"
DIM = 16


class CountingEmbedder:
    """确定性向量 + 调用计数。

    计数是这个测试的核心断言之一：`SKIPPED` 只说明返回了什么，不代表真的没干活。
    只有 embedder 的调用次数为 0，才能证明"没变就不重嵌"。
    """

    def __init__(self, dim: int = DIM, name: str = "counting") -> None:
        self._dim = dim
        self._name = name
        self.calls = 0
        self.texts_embedded = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [
            [((sum(map(ord, text)) + i) % 89) / 89.0 for i in range(self._dim)]
            for text in texts
        ]


def chunk_by_blank_line(
    content: str, *, document_id: str, metadata: dict[str, object] | None = None
) -> list[Chunk]:
    """最笨但确定的切分：空行分段。测试不关心切分质量，只关心条数稳定。"""

    blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
    return [
        Chunk(
            chunk_id=f"{document_id}-c{index}",
            document_id=document_id,
            text=block,
            source_uri=f"docs/{document_id}.md",
            metadata=dict(metadata or {}),
        )
        for index, block in enumerate(blocks)
    ]


DOC_A_V1 = "第一段：品牌语气要克制。\n\n第二段：避免夸张形容词。\n\n第三段：结论先行。"
DOC_A_V2 = (
    "第一段：品牌语气要克制。\n\n第二段：避免夸张形容词。\n\n"
    "第三段：结论先行。\n\n第四段：新增的一段，用于验证替换。"
)
DOC_B = "另一份文档的第一段。\n\n另一份文档的第二段。"


@pytest.fixture()
def store() -> AsyncDocumentStore:
    inner = PgVectorStore(DSN, table=TABLE, dimension=DIM)
    inner.ensure_schema()
    inner.drop()
    # 注册表不会被 drop() 清掉（它只 TRUNCATE chunk 表），这里显式清，
    # 否则上一次跑留下的 hash 会让本次首轮 sync 被判成 SKIPPED。
    inner._connection().execute(f"DELETE FROM {TABLE}_documents")
    store = AsyncDocumentStore(inner)
    yield store
    inner.drop()
    inner._connection().execute(f"DELETE FROM {TABLE}_documents")
    store.close()


def _ids(store: AsyncDocumentStore, document_id: str) -> list[str]:
    rows = store._store._connection().execute(  # noqa: SLF001 - 验收要看真库内容
        f"SELECT chunk_id FROM {TABLE} WHERE document_id = %s ORDER BY chunk_id",
        (document_id,),
    ).fetchall()
    return [row[0] for row in rows]


async def test_create_update_delete_without_full_rebuild(store: AsyncDocumentStore) -> None:
    embedder = CountingEmbedder()
    indexer = IncrementalIndexer(store, embedder, model_snapshot="m1")

    # 1) 首次入库
    created = await indexer.sync("doc-a", DOC_A_V1, chunker=chunk_by_blank_line)
    assert created.action is SyncAction.CREATED
    assert created.chunks_written == 3
    assert len(_ids(store, "doc-a")) == 3

    # 2) 新增第二份文档：doc-a 的块**一条都不能少**（这就是"免全量重建"）
    other = await indexer.sync("doc-b", DOC_B, chunker=chunk_by_blank_line)
    assert other.action is SyncAction.CREATED
    assert len(_ids(store, "doc-a")) == 3
    assert len(_ids(store, "doc-b")) == 2

    # 3) 内容未变再同步：SKIPPED，且 embedder 没被调用
    calls_before = embedder.calls
    skipped = await indexer.sync("doc-a", DOC_A_V1, chunker=chunk_by_blank_line)
    assert skipped.action is SyncAction.SKIPPED
    assert embedder.calls == calls_before, "SKIPPED 却仍在做 embedding，成本没省"
    # 换行符不同不算改（归一化只统一换行）
    assert content_hash(DOC_A_V1) == content_hash(DOC_A_V1.replace("\n", "\r\n"))

    # 4) 内容变了：旧块全消失、新块全在，版本号 +1
    updated = await indexer.sync("doc-a", DOC_A_V2, chunker=chunk_by_blank_line)
    assert updated.action is SyncAction.UPDATED
    assert updated.version == 2
    assert updated.chunks_written == 4
    assert updated.chunks_deleted == 3
    assert _ids(store, "doc-a") == ["doc-a-c0", "doc-a-c1", "doc-a-c2", "doc-a-c3"]
    # 另一份文档完全不受影响
    assert len(_ids(store, "doc-b")) == 2

    # 5) 删除：清零；重复删除幂等
    deleted = await indexer.delete("doc-a")
    assert deleted.action is SyncAction.DELETED
    assert deleted.chunks_deleted == 4
    assert _ids(store, "doc-a") == []
    again = await indexer.delete("doc-a")
    assert again.chunks_deleted == 0
    assert len(_ids(store, "doc-b")) == 2


async def test_model_snapshot_change_forces_rebuild(store: AsyncDocumentStore) -> None:
    """内容没变但模型换了 → 必须重建。

    不重建的话，新 query 向量与旧库向量不在同一空间，相似度全是噪声，
    而检索本身不会报任何错——只在"召回质量莫名变差"时才被发现。
    """

    first = IncrementalIndexer(store, CountingEmbedder(name="m"), model_snapshot="m1")
    await first.sync("doc-a", DOC_A_V1, chunker=chunk_by_blank_line)

    second = IncrementalIndexer(store, CountingEmbedder(name="m"), model_snapshot="m2")
    rebuilt = await second.sync("doc-a", DOC_A_V1, chunker=chunk_by_blank_line)

    assert rebuilt.action is SyncAction.UPDATED
    assert rebuilt.version == 2
    assert "模型变了" in rebuilt.note


async def test_document_id_mismatch_is_refused(store: AsyncDocumentStore) -> None:
    """切分器把 document_id 写错 → 拒绝写入。

    放过去的后果是"删了文档但 chunk 还在，检索照样召回"——删除传播静默失效。
    """

    def wrong_chunker(
        content: str, *, document_id: str, metadata: dict[str, object] | None = None
    ) -> list[Chunk]:
        return [
            Chunk(
                chunk_id="x-c0",
                document_id="someone-else",
                text=content,
                source_uri="docs/x.md",
            )
        ]

    indexer = IncrementalIndexer(store, CountingEmbedder(), model_snapshot="m1")

    with pytest.raises(ValueError, match="document_id"):
        await indexer.sync("doc-x", "正文", chunker=wrong_chunker)
    assert _ids(store, "doc-x") == []

"""增量索引的契约（W6 D2）。

测的是"省下来的那次 embedding"与"失败时不会留下半新半旧"，
而不是"能写进库"。四条用例对应四种真实返工：

- 不比对 hash → 每次同步都重嵌，成本随语料线性增长；
- 更新非事务 → 中途失败后引用指向已删 chunk；
- 删除不按文档成组 → 删完还能召回；
- 模型换了却跳过 → 新旧向量混在一个空间，检索静默变差。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from agent_framework.knowledge import (
    Chunk,
    DocVersion,
    IncrementalIndexer,
    SyncAction,
    content_hash,
)


class FakeEmbedder:
    name = "fake-embed"
    dimension = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        return [[float(len(text) % 11)] * self.dimension for text in batch]

    @property
    def embedded_chunks(self) -> int:
        return sum(len(batch) for batch in self.calls)


class FakeStore:
    """内存版 DocumentStore。行为对齐 pgvector 实现：replace 内删旧写新。"""

    def __init__(self) -> None:
        self.versions: dict[str, DocVersion] = {}
        self.rows: dict[str, list[str]] = {}
        self.replaces: list[tuple[str, int, int]] = []

    async def get_version(self, document_id: str) -> DocVersion | None:
        return self.versions.get(document_id)

    async def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        version: int,
        content_hash: str,
        model_snapshot: str,
    ) -> tuple[int, int]:
        assert len(chunks) == len(vectors)
        deleted = len(self.rows.get(document_id, []))
        self.rows[document_id] = [chunk.chunk_id for chunk in chunks]
        self.versions[document_id] = DocVersion(
            document_id=document_id,
            content_hash=content_hash,
            version=version,
            model_snapshot=model_snapshot,
        )
        self.replaces.append((document_id, len(chunks), deleted))
        return len(chunks), deleted

    async def delete_document(self, document_id: str) -> int:
        removed = len(self.rows.pop(document_id, []))
        self.versions.pop(document_id, None)
        return removed


def _chunker(
    content: str, *, document_id: str, metadata: dict | None = None
) -> list[Chunk]:
    """按空行切段，模拟"页感知切分"的最小版本（真实实现由检索侧注入）。"""

    parts = [part.strip() for part in content.split("\n\n") if part.strip()]
    return [
        Chunk(
            chunk_id=f"{document_id}-{index:03d}",
            document_id=document_id,
            text=part,
            source_uri=str((metadata or {}).get("source_uri", "docs/a.md")),
            metadata={"document_title": (metadata or {}).get("title", "")},
        )
        for index, part in enumerate(parts)
    ]


@pytest.fixture()
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def indexer(store: FakeStore, embedder: FakeEmbedder) -> IncrementalIndexer:
    return IncrementalIndexer(store, embedder)


# --- 新增 --------------------------------------------------------------------


async def test_first_sync_creates_version_one(
    indexer: IncrementalIndexer, store: FakeStore, embedder: FakeEmbedder
) -> None:
    result = await indexer.sync("doc-a", "第一段\n\n第二段", chunker=_chunker)

    assert result.action is SyncAction.CREATED
    assert result.version == 1
    assert result.chunks_written == 2
    assert result.chunks_deleted == 0
    assert store.rows["doc-a"] == ["doc-a-000", "doc-a-001"]
    assert embedder.embedded_chunks == 2


async def test_syncing_a_new_document_does_not_touch_the_old_one(
    indexer: IncrementalIndexer, store: FakeStore
) -> None:
    """"新增不改旧"是 ROADMAP 检查点的字面要求：不是全量重建。"""

    await indexer.sync("doc-a", "A 的第一段", chunker=_chunker)
    before = dict(store.rows)

    await indexer.sync("doc-b", "B 的第一段\n\nB 的第二段", chunker=_chunker)

    assert store.rows["doc-a"] == before["doc-a"]
    assert store.replaces == [("doc-a", 1, 0), ("doc-b", 2, 0)]


# --- 跳过 --------------------------------------------------------------------


async def test_unchanged_content_skips_embedding(
    indexer: IncrementalIndexer, embedder: FakeEmbedder, store: FakeStore
) -> None:
    """跳过是这里最值钱的分支：不跳过就等于每次同步都按字符数付一次 embedding。"""

    await indexer.sync("doc-a", "同一段内容", chunker=_chunker)
    calls_after_first = len(embedder.calls)

    result = await indexer.sync("doc-a", "同一段内容", chunker=_chunker)

    assert result.action is SyncAction.SKIPPED
    assert result.chunks_written == 0
    assert len(embedder.calls) == calls_after_first
    assert len(store.replaces) == 1


async def test_line_ending_change_is_not_a_content_change(
    indexer: IncrementalIndexer, embedder: FakeEmbedder
) -> None:
    """CRLF 与 LF 是同一个文档。不归一化就会把"换了编辑器"当成内容更新。"""

    await indexer.sync("doc-a", "第一段\n第二段", chunker=_chunker)
    result = await indexer.sync("doc-a", "第一段\r\n第二段", chunker=_chunker)

    assert result.action is SyncAction.SKIPPED
    assert embedder.embedded_chunks == 1  # 只有第一次嵌过


async def test_model_change_forces_rebuild_even_with_same_content(
    store: FakeStore,
) -> None:
    """换 embedding 模型后旧向量与新查询不在同一空间，必须重建（ADR-009）。"""

    old_embedder = FakeEmbedder()
    await IncrementalIndexer(store, old_embedder, model_snapshot="bge-m3:1024").sync(
        "doc-a", "同一段内容", chunker=_chunker
    )

    new_embedder = FakeEmbedder()
    result = await IncrementalIndexer(
        store, new_embedder, model_snapshot="bge-large:1024"
    ).sync("doc-a", "同一段内容", chunker=_chunker)

    assert result.action is SyncAction.UPDATED
    assert "模型变了" in result.note
    assert new_embedder.embedded_chunks == 1


# --- 更新 --------------------------------------------------------------------


async def test_update_replaces_all_chunks_and_bumps_version(
    indexer: IncrementalIndexer, store: FakeStore
) -> None:
    await indexer.sync("doc-a", "旧第一段\n\n旧第二段", chunker=_chunker)

    result = await indexer.sync("doc-a", "新第一段", chunker=_chunker)

    assert result.action is SyncAction.UPDATED
    assert result.version == 2
    assert result.chunks_written == 1
    assert result.chunks_deleted == 2  # 旧的两段被清掉，不留残留
    assert store.rows["doc-a"] == ["doc-a-000"]


async def test_repeated_upsert_is_idempotent(
    indexer: IncrementalIndexer, store: FakeStore
) -> None:
    await indexer.sync("doc-a", "同一段内容", chunker=_chunker)
    first = dict(store.rows)

    await indexer.sync("doc-a", "同一段内容", chunker=_chunker)
    await indexer.sync("doc-a", "同一段内容", chunker=_chunker)

    assert store.rows == first


# --- 删除 --------------------------------------------------------------------


async def test_delete_removes_all_chunks_and_resets_version(
    indexer: IncrementalIndexer, store: FakeStore
) -> None:
    await indexer.sync("doc-a", "第一段\n\n第二段", chunker=_chunker)

    result = await indexer.delete("doc-a")

    assert result.action is SyncAction.DELETED
    assert result.chunks_deleted == 2
    assert "doc-a" not in store.rows
    assert store.versions.get("doc-a") is None


async def test_delete_is_idempotent(indexer: IncrementalIndexer) -> None:
    await indexer.sync("doc-a", "第一段", chunker=_chunker)
    await indexer.delete("doc-a")

    again = await indexer.delete("doc-a")

    assert again.chunks_deleted == 0
    assert "幂等" in again.note


# --- 防御 --------------------------------------------------------------------


async def test_empty_content_is_rejected(indexer: IncrementalIndexer) -> None:
    """空内容不能"更新成 0 块"——那会静默清空已有索引且返回成功。"""

    with pytest.raises(ValueError, match="内容为空"):
        await indexer.sync("doc-a", "   \n\n  ", chunker=_chunker)


async def test_chunker_returning_nothing_is_rejected(indexer: IncrementalIndexer) -> None:
    with pytest.raises(ValueError, match="0 个块"):
        await indexer.sync("doc-a", "有内容", chunker=lambda *a, **k: [])


async def test_chunker_must_not_rewrite_document_id(
    indexer: IncrementalIndexer,
) -> None:
    """切分器写错 document_id 时，删除传播会漏删——必须当场拦住。"""

    def bad_chunker(content: str, *, document_id: str, metadata: dict | None = None):
        return [
            Chunk(
                chunk_id="x-1",
                document_id="wrong-id",
                text="正文",
                source_uri="docs/a.md",
            )
        ]

    with pytest.raises(ValueError, match="document_id"):
        await indexer.sync("doc-a", "有内容", chunker=bad_chunker)


async def test_embedding_count_mismatch_is_rejected(store: FakeStore) -> None:
    """条数不齐时按序配对会静默错位（第 3 块的向量是第 4 段的）。"""

    class ShortEmbedder(FakeEmbedder):
        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[0.0] * self.dimension]

    indexer = IncrementalIndexer(store, ShortEmbedder())

    with pytest.raises(ValueError, match="embedding 条数"):
        await indexer.sync("doc-a", "第一段\n\n第二段", chunker=_chunker)


async def test_dry_run_does_not_write(indexer: IncrementalIndexer, store: FakeStore) -> None:
    result = await indexer.sync("doc-a", "第一段", chunker=_chunker, dry_run=True)

    assert result.chunks_written == 1
    assert "dry_run" in result.note
    assert store.rows == {}


def test_content_hash_ignores_line_endings_only() -> None:
    assert content_hash("a\nb") == content_hash("a\r\nb")
    # 行内空格是真实变化（会改变切分与检索），不能被归一化吃掉
    assert content_hash("a\nb") != content_hash("a \nb")
    # 全角/半角差异同理
    assert content_hash("（a）") != content_hash("(a)")

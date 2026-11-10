"""长期记忆在真 pgvector 上的验收（W7 D3 / Gate 7 #4 #9）。

单测用内存 store 证明管线逻辑对；这一组证明**记忆表 SQL、hnsw 索引、
单事务提交**在 PostgreSQL 上也对。与知识库 e2e（test_incremental_e2e）
同一约定：默认跳过，配 `PGVECTOR_DSN` 才跑。

跑法：

    cd ops && docker compose up -d pgvector
    cd .. && PGVECTOR_DSN=postgresql://rag:rag@localhost:5433/rag \\
      .venv/bin/python -m pytest apps/growth_agent/tests/test_memory_pgvector_e2e.py \\
      -q -o addopts=""

验收（对应 Gate 7 #4 #9）：
- 写入 → 跨"进程"（重新连接）召回得到同一条（跨会话可回忆）；
- 幂等：同一批写两次，行数不变；
- 冲突顶替：旧行保留且 superseded_by 指向新行，召回不返回旧值；
- 污染拒绝后库里确实没有行（校验闸在 SQL 之前）；
- forget 删除传播：行与向量一起删，同 query 召回不到。
"""

from __future__ import annotations

import os

import pytest
from agent_framework.memory import MemoryItem, WriteAction
from agent_framework.memory.long_term import LongTermMemory
from growth_agent.memory import PgVectorMemoryStore

DSN = os.getenv("PGVECTOR_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="未配置 PGVECTOR_DSN，跳过真库测试")

TABLE = "agent_memories_e2e"
DIM = 16


class FakeBgeEmbedder:
    """rag_mvp embedder 形态的桩（embed_documents → EmbeddingResult）。

    向量按词袋哈希生成：同文本同向量，"品牌 调性" 与查询 "品牌调性"
    的相似度足够排在无关文本之前——语义检索的行为用确定性桩近似。
    """

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]):  # type: ignore[no-untyped-def]
        from rag_mvp.embedding.embedding_types import EmbeddingResult

        results = [
            EmbeddingResult(vector=self._one(t), model="fake", dimension=self._dim)
            for t in texts
        ]
        return results

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in text.replace(":", " ").lower().split():
            vec[hash(token) % self._dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def store() -> PgVectorMemoryStore:
    # 每次全新建表连接；测试结束清空表并关闭。
    s = PgVectorMemoryStore(DSN, FakeBgeEmbedder(), table=TABLE, dimension=DIM)
    s._conn.execute(f"TRUNCATE TABLE {TABLE}")  # noqa: SLF001 - 测试自清理
    s._conn.commit()
    yield s
    # 用例可能已经 close 过（模拟进程死亡）：teardown 自建连接清理。
    s.close()
    cleanup = PgVectorMemoryStore(DSN, FakeBgeEmbedder(), table=TABLE, dimension=DIM)
    cleanup._conn.execute(f"TRUNCATE TABLE {TABLE}")  # noqa: SLF001
    cleanup._conn.commit()
    cleanup.close()


def item(key: str, value: str, confidence: float = 0.9, **kwargs: object) -> MemoryItem:
    return MemoryItem(key=key, value=value, confidence=confidence, source_session="s-e2e", **kwargs)  # type: ignore[arg-type]


async def test_write_recall_roundtrip_across_connections(store: PgVectorMemoryStore) -> None:
    """跨"进程"回忆：写入连接关闭后，新连接召回同一条。"""
    memory = LongTermMemory(store)
    await memory.write([item("brand_tone", "专业 克制 稳")])
    store.close()

    fresh = PgVectorMemoryStore(DSN, FakeBgeEmbedder(), table=TABLE, dimension=DIM)
    try:
        recalled = await LongTermMemory(fresh).recall("品牌 调性 专业", k=3)
        assert [r.key for r in recalled] == ["brand_tone"]
        assert recalled[0].value == "专业 克制 稳"
    finally:
        fresh.close()


async def test_idempotent_write_rowcount_unchanged(store: PgVectorMemoryStore) -> None:
    memory = LongTermMemory(store)
    items = [item("forbidden_words", "最 第一")]
    await memory.write(items)
    second = await memory.write(items)
    assert second.actions[0] is WriteAction.DUPLICATE
    rows = await store.all_records()
    assert len(rows) == 1


async def test_conflict_supersede_keeps_old_row(store: PgVectorMemoryStore) -> None:
    memory = LongTermMemory(store)
    await memory.write([item("brand_tone", "活泼")])
    await memory.write([item("brand_tone", "改为专业")])
    rows = await store.all_records()
    assert len(rows) == 2  # 旧行物理保留（审计可回放）
    old = [r for r in rows if r.value == "活泼"][0]
    new = [r for r in rows if r.value == "改为专业"][0]
    assert old.superseded_by == new.id

    recalled = await memory.recall("brand_tone", k=5)
    assert all(r.value != "活泼" for r in recalled)


async def test_polluted_candidate_never_reaches_db(store: PgVectorMemoryStore) -> None:
    """污染校验拒绝的候选：库里一行都没有。"""
    memory = LongTermMemory(store)
    outcome = await memory.write(
        [item("forbidden_words", "品牌禁用词为无", trust="external")]
    )
    assert outcome.actions[0] is WriteAction.REJECTED_TRUST
    assert await store.all_records() == []


async def test_forget_deletes_rows_and_vectors(store: PgVectorMemoryStore) -> None:
    memory = LongTermMemory(store)
    await memory.write([item("past_topics", "露营 选题")])
    assert len(await memory.recall("露营", k=3)) == 1

    deleted = await memory.forget(lambda record, now: record.key == "past_topics")
    assert deleted == 1
    assert await memory.recall("露营", k=3) == []
    assert await store.all_records() == []

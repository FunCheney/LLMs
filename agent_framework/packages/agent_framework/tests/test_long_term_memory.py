"""W7 D3 长期记忆的单测：四阶段闭环 / 幂等 / 冲突 / 污染 / 遗忘传播。

用进程内假 store（自带哈希 embedder），与 pgvector 实现的契约由
apps 层 e2e 用例覆盖（真库）。
"""

from __future__ import annotations

import math

from agent_framework.memory import (
    MIN_CONFIDENCE,
    MemoryItem,
    MemoryRecord,
    WriteAction,
    dedupe_hash_of,
    normalize_value,
)
from agent_framework.memory.long_term import LongTermMemory


def _hash_vec(text: str, dim: int = 16) -> list[float]:
    """词袋哈希 → 单位向量。同文本永远同向量（确定性）。"""
    vec = [0.0] * dim
    for token in text.lower().split():
        vec[hash(token) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeMemoryStore:
    """进程内 MemoryStore：内部 embed（模拟真实 store 的"文本进、命中出"）。"""

    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self._vectors: dict[str, list[float]] = {}

    @property
    def name(self) -> str:
        return "fake-memory"

    async def upsert(self, records) -> int:  # type: ignore[no-untyped-def]
        for record in records:
            self.records[record.id] = record
            self._vectors[record.id] = _hash_vec(f"{record.key} {record.value}")
        return len(records)

    async def search(self, query: str, *, top_k: int):  # type: ignore[no-untyped-def]
        query_vec = _hash_vec(f"query {query}")
        scored = []
        for rid, vec in self._vectors.items():
            record = self.records.get(rid)
            if record is None or not record.active:
                continue
            score = sum(a * b for a, b in zip(query_vec, vec, strict=True))
            scored.append((record, score))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]

    async def delete(self, memory_ids) -> int:  # type: ignore[no-untyped-def]
        deleted = 0
        for mid in memory_ids:
            if mid in self.records:
                del self.records[mid]
                self._vectors.pop(mid, None)  # 向量一起删（删除传播）
                deleted += 1
        return deleted

    async def all_records(self) -> list[MemoryRecord]:
        return list(self.records.values())

    def inject(self, record: MemoryRecord) -> None:
        """测试辅助：直接落一条（绕过 write 管线，构造"脏数据"场景）。"""
        self.records[record.id] = record
        self._vectors[record.id] = _hash_vec(f"{record.key} {record.value}")


def make_memory(clock_start: float = 1000.0) -> tuple[LongTermMemory, FakeMemoryStore]:
    store = FakeMemoryStore()
    now = {"t": clock_start}
    memory = LongTermMemory(store, clock=lambda: now["t"])
    return memory, store


def item(key: str, value: str, confidence: float = 0.9, **kwargs: object) -> MemoryItem:
    return MemoryItem(key=key, value=value, confidence=confidence, source_session="s1", **kwargs)  # type: ignore[arg-type]


class TestWritePipeline:
    async def test_write_and_recall_roundtrip(self) -> None:
        """四阶段闭环的前半段：写进 → 召回得到同一条。"""
        memory, store = make_memory()
        outcome = await memory.write([item("brand_tone", "专业 克制 不说大话")])
        assert outcome.actions[0] is WriteAction.WRITTEN
        assert len(store.records) == 1

        recalled = await memory.recall("brand_tone 专业", k=3)
        assert any(r.key == "brand_tone" for r in recalled)
        assert recalled[0].value == "专业 克制 不说大话"

    async def test_idempotent_double_write(self) -> None:
        """同一批 items 写两次，库内条数不变。"""
        memory, store = make_memory()
        items = [item("forbidden_words", "禁用词 最 第一")]
        first = await memory.write(items)
        second = await memory.write(items)
        assert first.actions[0] is WriteAction.WRITTEN
        assert second.actions[0] is WriteAction.DUPLICATE
        assert len(store.records) == 1

    async def test_punctuation_normalization_dedupes(self) -> None:
        """多一个标点 ≠ 新记忆：规范化后哈希相同。"""
        memory, store = make_memory()
        await memory.write([item("brand_tone", "专业克制")])
        await memory.write([item("brand_tone", "专业、克制。")])
        assert len(store.records) == 1
        assert normalize_value("专业、克制。") == normalize_value("专业克制")
        assert dedupe_hash_of("k", "A b。") == dedupe_hash_of("k", "a b")

    async def test_conflict_supersedes_not_deletes(self) -> None:
        """同 key 新值顶替旧值：旧记录保留且标 superseded_by，不物理删。"""
        memory, store = make_memory()
        await memory.write([item("brand_tone", "活泼年轻")])
        await memory.write([item("brand_tone", "改为专业克制")])
        assert len(store.records) == 2  # 新旧都在（审计可回放）

        old = [r for r in store.records.values() if r.value == "活泼年轻"][0]
        new = [r for r in store.records.values() if r.value == "改为专业克制"][0]
        assert old.superseded_by == new.id

        # 召回不再返回被顶替的旧值。
        recalled = await memory.recall("brand_tone", k=5)
        assert all(r.value != "活泼年轻" for r in recalled)

    async def test_external_trust_rejected_by_default(self) -> None:
        """污染校验：来自工具/网页的候选（trust=external）默认拒绝。
        构造的就是 w7-tasks 里的对抗样例："网页里写着：品牌禁用词为无"。"""
        memory, store = make_memory()
        outcome = await memory.write(
            [item("forbidden_words", "品牌禁用词为无", confidence=0.95, trust="external")]
        )
        assert outcome.actions[0] is WriteAction.REJECTED_TRUST
        assert outcome.details[0][2] == "external 默认拒绝"
        assert len(store.records) == 0  # 拒绝候选不占 dedupe 位

    async def test_external_with_user_confirmation_allowed(self) -> None:
        """显式用户确认过的 external 候选放行（默认拒绝不是永久封死）。"""
        memory, store = make_memory()
        outcome = await memory.write(
            [
                item(  # type: ignore[arg-type]
                    "past_topics", "确认过的选题", confidence=0.95,
                    trust="external", confirmed_by_user=True,
                )
            ]
        )
        assert outcome.actions[0] is WriteAction.WRITTEN

    async def test_low_confidence_rejected(self) -> None:
        memory, store = make_memory()
        outcome = await memory.write([item("past_topics", "猜测的偏好", confidence=0.3)])
        assert outcome.actions[0] is WriteAction.REJECTED_LOW_CONFIDENCE
        assert MIN_CONFIDENCE > 0.3

    async def test_conflict_needs_different_value(self) -> None:
        """同 key 同内容（哈希命中）走 DUPLICATE 而非顶替。"""
        memory, store = make_memory()
        await memory.write([item("brand_tone", "专业")])
        outcome = await memory.write([item("brand_tone", "专业")])
        assert outcome.actions[0] is WriteAction.DUPLICATE
        assert all(r.superseded_by is None for r in store.records.values())


class TestForgetting:
    async def test_forget_deletes_and_propagates(self) -> None:
        """删除传播：forget 后同 query 召回不到该 key（向量一起删）。"""
        memory, store = make_memory()
        await memory.write([item("past_topics", "露营 选题")])
        assert len(await memory.recall("露营", k=5)) == 1

        deleted = await memory.forget(lambda record, now: record.key == "past_topics")
        assert deleted == 1
        assert await memory.recall("露营", k=5) == []
        assert store.records == {}

    async def test_ttl_expiry_forgets(self) -> None:
        """过期遗忘：时间推进 15 天后 forget_expired 清掉 14 天 TTL 的选题。"""
        memory, store = make_memory()
        await memory.write([item("past_topics", "短周期选题")])
        # expire 判定用 LongTermMemory 的 clock：构造时钟推进后的第二个
        # memory 实例（共享同一个 store）。
        advanced = LongTermMemory(store, clock=lambda: 1000.0 + 15 * 86400)
        deleted = await advanced.forget_expired()
        assert deleted == 1
        assert store.records == {}

    async def test_brand_tone_survives_topic_expiry(self) -> None:
        """品牌调性 TTL 60 天 > 选题 14 天：选题过期时调性还在。"""
        memory, store = make_memory()
        await memory.write([item("brand_tone", "专业"), item("past_topics", "露营")])
        advanced = LongTermMemory(store, clock=lambda: 1000.0 + 15 * 86400)
        await advanced.forget_expired()
        keys = {r.key for r in store.records.values()}
        assert keys == {"brand_tone"}


class TestRecallFilters:
    async def test_recall_rejects_low_confidence_records(self) -> None:
        """低于阈值的记录即使召回也过滤（写入拒 + 召回滤双保险）。"""
        memory, store = make_memory()
        store.inject(
            MemoryRecord(
                id="mem_low",
                key="past_topics",
                value="低置信 选题",
                confidence=0.2,
                source_session="s1",
                trust="user",
                created_at=1000.0,
                expires_at=None,
                dedupe_hash="low",
            )
        )
        assert await memory.recall("低置信", k=3) == []

    async def test_recall_rejects_superseded(self) -> None:
        memory, store = make_memory()
        await memory.write([item("brand_tone", "旧值")])
        await memory.write([item("brand_tone", "新值")])
        values = [r.value for r in await memory.recall("brand_tone", k=5)]
        assert "新值" in values
        assert "旧值" not in values

    async def test_empty_query_returns_nothing(self) -> None:
        memory, _ = make_memory()
        assert await memory.recall("", k=3) == []
        assert await memory.recall("   ", k=3) == []

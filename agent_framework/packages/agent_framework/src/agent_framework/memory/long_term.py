"""长期记忆（W7 D3）：提取 / 写入 / 检索 / 遗忘 四阶段。

**知识库 vs 记忆的边界**（docs/w7/memory-vs-knowledge.md 的代码侧执行）：
记忆与文档检索共用 pgvector 底座但**不同表**（namespace 隔离）。混存的
后果是双向的：事实被当成偏好遗忘掉，偏好被当成事实参与引用。框架侧
用 `MemoryStore` 协议隔离——实现方必须用独立表，这是接入契约而不是
建议。

写入管线（提取 → 规范化 → 校验 → 策略判断 → 提交）五道闸，
每道都有"不做会怎样"的对应故障（见 w7-tasks Day3 表格）：

    规范化：去空白/统一标点（hash 用），同内容不因多一个句号存两条
    幂等：  dedupe_hash upsert，重试/重复消费不写重复记忆
    冲突：  同 key 新值 → 旧值 superseded_by 指向新值，不物理删
    污染：  只收 trust in (system, user)；external 默认拒绝
            （按**来源信任级**拦，不是关键词黑名单——换说法就绕过的
            东西不叫防线）
    提交：  单次 upsert（新记录 + 被顶替的旧记录一起写），
            失败不留半条

最贵的错误不是"没记住"，是**记错并且一直用错**——它不报错，只会让
后面每一次生成都带着同一个错误偏好。所以校验闸宁可误拒（宁可少记）。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_TTL_S",
    "MIN_CONFIDENCE",
    "MemoryItem",
    "MemoryRecord",
    "MemoryStore",
    "WriteAction",
    "WriteOutcome",
    "dedupe_hash_of",
    "normalize_value",
]

# 各类 key 的默认 TTL（秒）。品牌调性/禁用词是长期项；选题类更新快，
# 记太久反而会在下次选题时召回过时偏好。
DEFAULT_TTL_S: dict[str, float] = {
    "brand_tone": 60 * 86400,
    "forbidden_words": 60 * 86400,
    "past_topics": 14 * 86400,
}
UNKNOWN_KEY_TTL_S = 30 * 86400

# 低置信度记忆不参与召回（遗忘闸的前半段；后半段是 forget 清库）。
MIN_CONFIDENCE = 0.6

# 规范化时视作等价的空白与标点。转小写比较、存原文——
# "Apple " 与 "apple、" 必须命中同一条记忆。
_STRIP_CHARS = " \t\r\n　，。！？；：、,.!?;:"


class WriteAction(StrEnum):
    """一条候选在写入管线里的最终去向。"""

    WRITTEN = "written"
    # 同 dedupe_hash 已存在：幂等命中，什么都不写。
    DUPLICATE = "duplicate"
    # 同 key 旧值被新值顶替（旧记录不删，标 superseded_by）。
    SUPERSEDED = "superseded"
    # 污染校验拒绝（external 且未显式确认）。
    REJECTED_TRUST = "rejected_trust"
    # 置信度低于阈值。
    REJECTED_LOW_CONFIDENCE = "rejected_low_confidence"


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """记忆的候选/召回形态。key 是受控词表（brand_tone 等），
    value 存原文；比较与去重都走 normalize_value。"""

    key: str
    value: str
    # 提取方给出的置信度 0–1。低于 MIN_CONFIDENCE 的候选在写入时被拒。
    confidence: float
    # 这条记忆来自哪个会话（审计与"用户改主意"的溯源）。
    source_session: str
    # 信任级：system（我们配置的）/ user（用户会话原文）/ external
    # （工具返回、网页）。external 默认拒绝写入，除非 confirmed_by_user。
    trust: str = "user"
    # 用户显式确认过（"记住：..."）。external 内容只有带它才放行。
    confirmed_by_user: bool = False
    created_at: float = field(default_factory=time.time)
    # 由 write() 计算，调用方不填。
    dedupe_hash: str = ""
    # 被新值顶替时指向新记录 id（只在 MemoryRecord 上有意义）。
    superseded_by: str | None = None
    # 写入时按 key 的 TTL 推出来的过期时间戳（epoch 秒）。
    expires_at: float | None = None


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """存储形态（MemoryStore 读写它）。与 MemoryItem 分开：
    记录多两个治理字段（superseded_by 的目标、内容哈希），且永不回传给
    提取层——提取层只该看到"活跃记忆"。"""

    id: str
    key: str
    value: str
    confidence: float
    source_session: str
    trust: str
    created_at: float
    expires_at: float | None
    dedupe_hash: str
    superseded_by: str | None = None

    @property
    def active(self) -> bool:
        return self.superseded_by is None

    def alive_at(self, now: float) -> bool:
        return self.active and (self.expires_at is None or self.expires_at > now)

    def to_item(self) -> MemoryItem:
        return MemoryItem(
            key=self.key,
            value=self.value,
            confidence=self.confidence,
            source_session=self.source_session,
            trust=self.trust,
            created_at=self.created_at,
            dedupe_hash=self.dedupe_hash,
            superseded_by=self.superseded_by,
            expires_at=self.expires_at,
        )


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    """一次 write 的逐条结果 + 汇总。审计与实验报告直接读它。"""

    actions: tuple[WriteAction, ...] = ()
    # 各动作对应的（key, id/原因）。REJECTED_* 的第三段是原因。
    details: tuple[tuple[str, str, str], ...] = ()
    # 实际写入库的新记录数。
    written: int = 0
    # 被顶替的旧记录数（它们也被重写了一次——写入 superseded_by）。
    superseded: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "superseded": self.superseded,
            "actions": [a.value for a in self.actions],
            "details": [list(d) for d in self.details],
        }


@runtime_checkable
class MemoryStore(Protocol):
    """记忆存储协议。与知识库的 `VectorStore` 平行但**不共用**：
    记忆按 id upsert / 按语义检索 / 按 id 删除，实现必须用独立表
    （namespace 隔离是本协议的接入前提，不是可选优化）。

    **向量是 store 的私事**：接口是"文本进、命中出"（与 Retriever 协议
    同一形态）。调用方（LongTermMemory）不需要知道 embedding 模型是谁，
    换模型只换 store 实现——记忆与知识库甚至可以用不同 embedder。
    """

    @property
    def name(self) -> str: ...

    async def upsert(self, records: Sequence[MemoryRecord]) -> int:
        """按 id 写入或覆盖（store 自己 embed `key: value`），返回写入条数。
        **单次调用是一个提交单元**：中途失败不能留半条
        （调用方靠它做"新记录 + 顶替记录"的原子写）。"""
        ...

    async def search(self, query: str, *, top_k: int) -> list[tuple[MemoryRecord, float]]:
        """按语义召回，返回 (记录, 相似度)。实现侧应过滤或过取（over-fetch）
        让调用方能过滤活跃记录——两边都过滤没有坏处。"""
        ...

    async def delete(self, memory_ids: Sequence[str]) -> int:
        """按 id 删除（向量行一起删——删除传播）。返回条数。"""
        ...

    async def all_records(self) -> list[MemoryRecord]:
        """全量记录（含已顶替的）。forget/衰减扫描用；数据量大时实现
        可以换游标分页，接口不变。"""
        ...


def normalize_value(value: str) -> str:
    """去空白与常见标点、转小写。只用于比较与哈希，**不回写原文**——
    原文是审计对象，规范化形态是匹配形态。"""
    lowered = (value or "").strip().lower()
    return "".join(ch for ch in lowered if ch not in _STRIP_CHARS)


def dedupe_hash_of(key: str, value: str) -> str:
    """hash(key + normalize(value))：同内容重复写入只留一条。"""
    payload = f"{key.strip().lower()}\u0000{normalize_value(value)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class LongTermMemory:
    """四阶段的编排器。store 由调用方注入（框架零 ML 依赖：
    embedding 在 store 实现里，LLM 提取器是独立的 MemoryExtractor）。"""

    def __init__(
        self,
        store: MemoryStore,
        *,
        min_confidence: float = MIN_CONFIDENCE,
        ttl_s: dict[str, float] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._min_confidence = min_confidence
        self._ttl = dict(DEFAULT_TTL_S)
        if ttl_s:
            self._ttl.update(ttl_s)
        self._clock = clock

    # -- 写入 -------------------------------------------------------------- #

    async def write(self, items: Sequence[MemoryItem]) -> WriteOutcome:
        """五道闸全部通过的候选才落库。返回逐条去向（审计用）。"""
        if not items:
            return WriteOutcome()

        now = self._clock()
        existing = {r.dedupe_hash: r for r in await self._store.all_records() if r.active}
        by_key: dict[str, MemoryRecord] = {}
        for record in existing.values():
            by_key.setdefault(record.key, record)

        actions: list[WriteAction] = []
        details: list[tuple[str, str, str]] = []
        to_write: list[MemoryRecord] = []
        superseded_ids: list[str] = []

        for item in items:
            digest = item.dedupe_hash or dedupe_hash_of(item.key, item.value)

            # 闸 1：污染校验（先于幂等：被拒绝的候选不该占用 dedupe 位）。
            if item.trust == "external" and not item.confirmed_by_user:
                actions.append(WriteAction.REJECTED_TRUST)
                details.append((item.key, "rejected", "external 默认拒绝"))
                continue
            # 闸 2：置信度。
            if item.confidence < self._min_confidence:
                actions.append(WriteAction.REJECTED_LOW_CONFIDENCE)
                details.append((item.key, "rejected", f"confidence={item.confidence:.2f}"))
                continue
            # 闸 3：幂等。
            if digest in existing:
                actions.append(WriteAction.DUPLICATE)
                details.append((item.key, existing[digest].id, "dedupe_hash 命中"))
                continue

            record = self._to_record(item, digest, now)
            # 闸 4：同 key 冲突 → 旧记录顶替（不物理删，审计要能回放）。
            old = by_key.get(item.key)
            if old is not None and old.dedupe_hash != digest:
                superseded_ids.append(old.id)
                to_write.append(
                    MemoryRecord(
                        id=old.id,
                        key=old.key,
                        value=old.value,
                        confidence=old.confidence,
                        source_session=old.source_session,
                        trust=old.trust,
                        created_at=old.created_at,
                        expires_at=old.expires_at,
                        dedupe_hash=old.dedupe_hash,
                        superseded_by=record.id,
                    )
                )
                actions.append(WriteAction.SUPERSEDED)
                details.append((item.key, record.id, f"顶替 {old.id}"))
            else:
                actions.append(WriteAction.WRITTEN)
                details.append((item.key, record.id, "新写入"))

            to_write.append(record)
            existing[digest] = record
            by_key[item.key] = record

        # 闸 5：提交（单次 upsert = 原子单元，见 MemoryStore 协议注释）。
        written = await self._store.upsert(to_write) if to_write else 0
        return WriteOutcome(
            actions=tuple(actions),
            details=tuple(details),
            written=max(written - len(superseded_ids), 0),
            superseded=len(superseded_ids),
        )

    # -- 检索 -------------------------------------------------------------- #

    async def recall(self, query: str, k: int = 3) -> list[MemoryItem]:
        """按语义召回活跃记忆。过期/被顶替/低置信的记录在这里过滤——
        过滤放调用方而不是 store 侧：过滤规则（TTL/置信度）属于记忆
        策略，属于框架，store 只管存取。"""
        if k <= 0 or not query.strip():
            return []

        hits = await self._store.search(query, top_k=max(k * 3, k))
        now = self._clock()
        picked: list[tuple[float, MemoryRecord]] = []
        for record, score in hits:
            if record.alive_at(now) and record.confidence >= self._min_confidence:
                picked.append((score, record))
            if len(picked) >= k:
                break
        return [record.to_item() for _, record in picked]

    # -- 遗忘 -------------------------------------------------------------- #

    async def forget(
        self, predicate: Callable[[MemoryRecord, float], bool]
    ) -> int:
        """按谓词删除记忆（TTL 清理 / 显式删除 / 置信衰减）。

        删除传播：store.delete 连向量一起删。**不**用 superseded（那是
        "被顶替但保留审计"）；forget 是"用户/策略要求消失"，两者语义
        相反——见 docs/w7/memory-vs-knowledge.md 的生命周期一节。
        """
        now = self._clock()
        targets = [
            record.id
            for record in await self._store.all_records()
            if record.active and predicate(record, now)
        ]
        if not targets:
            return 0
        return await self._store.delete(targets)

    async def forget_expired(self) -> int:
        """TTL 清理的入口（衰减/过期是"遗忘"最常见的一种）。"""

        async def is_dead(record: MemoryRecord, now: float) -> bool:
            return not record.alive_at(now)

        return await self.forget(lambda r, now: not r.alive_at(now))

    # -- 内部 -------------------------------------------------------------- #

    def _to_record(self, item: MemoryItem, digest: str, now: float) -> MemoryRecord:
        ttl = self._ttl.get(item.key, UNKNOWN_KEY_TTL_S)
        record_id = f"mem_{digest}"
        return MemoryRecord(
            id=record_id,
            key=item.key,
            value=item.value,
            confidence=item.confidence,
            source_session=item.source_session,
            trust=item.trust,
            created_at=now,
            expires_at=now + ttl,
            dedupe_hash=digest,
        )


def ttl_of(key: str, ttl_s: dict[str, float] | None = None) -> timedelta:
    """key → 生存期。给报告与配置校验用（人类可读形态）。"""
    table = dict(DEFAULT_TTL_S)
    if ttl_s:
        table.update(ttl_s)
    return timedelta(seconds=table.get(key, UNKNOWN_KEY_TTL_S))

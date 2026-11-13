"""语义缓存（W11 D5）：相似的 query 复用上一次的结果。

## 为什么"相似"这件事要先说清风险

语义缓存省的是钱，赔的是**正确性**，而且赔起来毫无征兆：
用户问"2026 年 Q3 的返点政策"，命中了一条"2026 年 Q2 的返点政策"的答案，
返回的文本读起来完全正常——**这是它比"报错"危险的地方**。

所以本模块有三道闸，缺一不可：

1. **相似度阈值**（默认 0.92）。低阈值等于把"差不多的问题"当"同一个问题"；
2. **TTL**。过期就是过期，不因为"昨天刚算过"而留用；
3. **资产版本**。prompt 或模型换了，旧答案就不再是"这套系统会给出的答案"。
   版本不一致的条目命中即作废，并单独计数（`invalidated`）——
   这不是"未命中"，混在一起会让人以为缓存没配好。

## 哪些查询**不该**进缓存

含时效词的查询（今天/最新/刚刚/now）不该缓存：它们的答案在语义上就要求"此刻"。
`is_timeable` 默认拦掉一批中文与英文词，并且**可注入**——
把判断写死在缓存里，等于把业务口径藏进基础设施。

## 命中要留标记，成本要从报告里消失但在账上分开

命中返回的 `ChatOutcome` 会带 `cache_hit=True`，`degraded` 保持原样。
成本报告里它必须与 `usage_estimated` 区分开：命中是"没花钱"，估算是"花了多少不确定"，
两者混在一起，成本数字就再也说不清了。

## 离线也能跑

`HashingEmbedder` 是确定性的字符 bigram 哈希向量：没有外部 embedding 服务也能测。
它不是"好"的向量，只是**可复现**的向量——测试要的是后者。
生产上换成真 embedding（同一个 `Embedder` 协议），缓存层不用改一行。
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..llm.client import ChatOutcome

__all__ = [
    "CacheStats",
    "Embedder",
    "HashingEmbedder",
    "SemanticCache",
    "cosine",
    "has_time_sensitive_term",
    "last_user_text",
]

# 时效词。命中任一就不缓存——**默认拦截**，因为"缓存了一个过时答案"没人会发现。
_TIME_SENSITIVE: tuple[str, ...] = (
    "今天",
    "今日",
    "现在",
    "最新",
    "刚刚",
    "本周",
    "本月",
    "实时",
    "today",
    "now",
    "latest",
    "current",
    "this week",
)


# 英文词要按词边界匹配：`"now" in "known"` 是真——那会把一个普通问题判成时效问题。
_ASCII_TERM = re.compile(r"\b(" + "|".join(t for t in _TIME_SENSITIVE if t.isascii()) + r")\b")


def has_time_sensitive_term(text: str) -> bool:
    """中文按子串、英文按词边界。混着用一个 `in` 判断，会误伤 `known` / `snow`。"""

    lowered = text.lower()
    if any(term in lowered for term in _TIME_SENSITIVE if not term.isascii()):
        return True
    return _ASCII_TERM.search(lowered) is not None


class Embedder(Protocol):
    """把文本变成向量。协议只要求这一个方法，方便换成真 embedding。"""

    def embed(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class HashingEmbedder:
    """确定性的字符 bigram 哈希向量。离线可复现，不连任何服务。

    `dim` 小一点（256）是刻意的：它逼着相似度靠"词面重合"而不是"高维巧合"，
    于是测试里的阈值行为可解释。
    """

    dim: int = 256

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dim
        cleaned = "".join(text.split()).lower()

        if len(cleaned) < 2:
            cleaned = cleaned or " "

        for index in range(len(cleaned) - 1):
            gram = cleaned[index : index + 2]
            slot = int(hashlib.sha256(gram.encode()).hexdigest()[:8], 16) % self.dim
            vector[slot] += 1.0

        norm = math.sqrt(sum(item * item for item in vector))
        return tuple(item / norm for item in vector) if norm else tuple(vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度。任一侧为零向量时返回 0——不是 1，也不是 NaN。"""

    if len(left) != len(right):
        raise ValueError(f"向量维度不一致：{len(left)} vs {len(right)}")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot  # 两侧都已归一化


@dataclass
class CacheStats:
    """命中的每一档都要分开数。

    `invalidated`（版本不一致）与 `expired`（TTL 过期）如果混进 `misses`，
    看板上就只剩"命中率低"，而不知道该调 TTL 还是该查版本漂移。
    """

    hits: int = 0
    misses: int = 0
    expired: int = 0
    invalidated: int = 0
    blocked: int = 0
    evicted: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses + self.expired + self.invalidated + self.blocked

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "expired": self.expired,
            "invalidated": self.invalidated,
            "blocked": self.blocked,
            "evicted": self.evicted,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass(slots=True)
class _Entry:
    text: str
    vector: tuple[float, ...]
    outcome: ChatOutcome
    stored_at: float
    asset_digest: str


@dataclass
class SemanticCache:
    """进程内语义缓存。多副本之间不共享（各自命中率不同，这是已知缺口）。"""

    embedder: Embedder = field(default_factory=HashingEmbedder)
    threshold: float = 0.92
    ttl_s: float = 86_400.0
    max_entries: int = 512
    # 资产快照的 digest（`runtime.assets.AssetSnapshot.digest`）。
    # 换 prompt / 换模型之后，旧条目必须作废——否则会拿"上一套系统"的答案回放。
    asset_digest: str = ""
    # 可注入的"这条查询能不能缓存"判断。默认拦时效词。
    is_cacheable: Callable[[str], bool] | None = None
    clock: Callable[[], float] = time.monotonic
    stats: CacheStats = field(default_factory=CacheStats, init=False)
    _entries: list[_Entry] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold 必须在 (0, 1]，收到 {self.threshold}")

    # -- 读写 --------------------------------------------------------------- #

    async def get(self, query: str) -> ChatOutcome | None:
        """命中返回带 `cache_hit=True` 的 outcome（**复制**一份，不改原件）。"""

        if not self._cacheable(query):
            self.stats.blocked += 1
            return None

        vector = tuple(self.embedder.embed(query))
        now = self.clock()
        best: tuple[float, _Entry] | None = None
        # 这一轮清掉了几条（过期或版本不符）。它决定"没命中"该记成哪一档：
        # 清掉过条目就记 expired/invalidated，**不再记 miss**——
        # 两档都记会让 `lookups` 把一次查询算两次，命中率分母直接翻倍。
        dropped = 0

        for entry in list(self._entries):
            if now - entry.stored_at > self.ttl_s:
                self._entries.remove(entry)
                self.stats.expired += 1
                dropped += 1
                continue

            if self.asset_digest and entry.asset_digest != self.asset_digest:
                self._entries.remove(entry)
                self.stats.invalidated += 1
                dropped += 1
                continue

            score = cosine(vector, entry.vector)
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, entry)

        if best is None:
            if not dropped:
                self.stats.misses += 1
            return None

        self.stats.hits += 1
        return _mark_hit(best[1].outcome)

    async def put(self, query: str, outcome: ChatOutcome) -> None:
        """写入。降级答案**不进缓存**——把"这不是真答案"缓存下来，会传播一整天。"""

        if not self._cacheable(query) or outcome.degraded:
            return

        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)  # 简单的 FIFO：语义缓存的命中本来就该集中在近期
            self.stats.evicted += 1

        self._entries.append(
            _Entry(
                text=query,
                vector=tuple(self.embedder.embed(query)),
                outcome=outcome,
                stored_at=self.clock(),
                asset_digest=self.asset_digest,
            )
        )

    # -- 给 DegradationPolicy 用 ------------------------------------------- #

    def as_cache_lookup(self) -> Callable[[Any], Awaitable[ChatOutcome | None]]:
        """适配 `DegradationPolicy(cache=...)` 的签名：`(call) -> awaitable`。"""

        async def _lookup(call: Any) -> ChatOutcome | None:
            return await self.get(last_user_text(call))

        return _lookup

    # -- 观测 --------------------------------------------------------------- #

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def _cacheable(self, query: str) -> bool:
        if self.is_cacheable is not None:
            return self.is_cacheable(query)
        return not has_time_sensitive_term(query)


def last_user_text(call: Any) -> str:
    """取最后一条 user 消息作为缓存键的原料。

    用整段 messages 做键会把"同一问题在不同上下文里"当成不同条目——
    那一半的命中就白丢了。用最后一问是取舍：**上下文变了仍可能命中**，
    所以 `asset_digest` 与 TTL 才必须生效。
    """

    messages: Iterable[Any] = getattr(call, "messages", ()) or ()
    for message in reversed(list(messages)):
        if str(getattr(message, "role", "")) in {"user", "MessageRole.USER"}:
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _mark_hit(outcome: ChatOutcome) -> ChatOutcome:
    """复制一份并打 `cache_hit`。**不改原件**：原件还在缓存里给后续请求用。"""

    return ChatOutcome(
        response=outcome.response,
        alias=outcome.alias,
        retry_count=outcome.retry_count,
        fallback_from=outcome.fallback_from,
        degraded=outcome.degraded,
        cache_hit=True,
    )

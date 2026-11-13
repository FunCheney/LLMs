"""幂等键（W11 D3）：同一个请求，只执行一次。

## 键是什么

`(principal, 规范化后的请求内容哈希)`。两个部分缺一不可：

- 只有内容哈希：A 和 B 提了完全一样的请求，B 会拿到 A 的结果——**跨用户串数据**；
- 只有 principal：同一个人连点两次"发布"，第二次会拿到第一次的结果（对），
  但同一个人写了两篇不同的稿子，第二篇会被判成重复（错）。

## 规范化那一步最容易写错

`json.dumps({"a":1,"b":2})` 与 `json.dumps({"b":2,"a":1})` 是两个不同的字符串，
但它们是同一个请求。所以必须 `sort_keys=True`。

**列表顺序不收**：`tags=["a","b"]` 与 `tags=["b","a"]` 在业务上可能是同一件事，
但"帮我把这三章按这个顺序写"里，顺序就是语义。宁可少去重一次，也不要把不同请求合并。

## 三种状态

| begin() 结果 | 含义 | 调用方该做什么 |
|---|---|---|
| `new` | 第一次见 | 执行，完成后 `complete(value)` |
| `in_flight` | 有人正在执行 | **409**，不要排队等（等=把并发堆在入口） |
| `completed` | 已经做完 | 直接返回 `entry.value`，不重跑 |

第三种是"重复提交同一 `POST /v1/tasks` 只执行一次、返回同一 task_id"的实现；
第二种是它最容易漏的一半——只判"完成过"，那么**并发**两次提交会各跑一遍，
而这两遍在日志里都合法。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

__all__ = [
    "DuplicateInFlight",
    "EntryState",
    "IdempotencyEntry",
    "IdempotencyKey",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "canonical_payload",
    "digest_of",
    "normalize_principal",
]


# --------------------------------------------------------------------------- #
# 规范化
# --------------------------------------------------------------------------- #


def canonical_payload(payload: Any) -> str:
    """把请求体压成唯一字符串。

    `sort_keys=True` 处理"字段顺序不同"；`separators` 去掉空格差异
    （`{"a": 1}` 与 `{"a":1}` 是同一个请求）；
    `ensure_ascii=False` 是必要的——中文被转成 `\\uXXXX` 之后，
    同一个请求用两种编码写会算出两个哈希。
    """

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        # `default=str` 只影响不可 JSON 化的对象（datetime/UUID），
        # 让它们退化成字符串而不是在哈希阶段炸掉整个请求。
        default=str,
    )


def digest_of(payload: Any) -> str:
    return hashlib.sha256(canonical_payload(payload).encode("utf-8")).hexdigest()[:32]


def normalize_principal(principal: str | None) -> str:
    """主体名规范化。匿名统一成 `anonymous`——不然 `None` 与 `""` 会算出两条键。"""

    cleaned = (principal or "").strip().lower()
    return cleaned or "anonymous"


# --------------------------------------------------------------------------- #
# 键
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class IdempotencyKey:
    principal: str
    digest: str

    @classmethod
    def for_request(cls, principal: str | None, payload: Any) -> IdempotencyKey:
        return cls(principal=normalize_principal(principal), digest=digest_of(payload))

    @classmethod
    def from_raw(cls, principal: str | None, raw: str) -> IdempotencyKey:
        """调用方自己带 `Idempotency-Key` 头时走这条。

        自带的键也**必须**拼上主体：客户端生成的 UUID 撞车概率极低但不是零，
        而一旦撞上，A 会拿到 B 的结果。
        """

        return cls(principal=normalize_principal(principal), digest=digest_of(raw))

    def as_str(self) -> str:
        return f"{self.principal}:{self.digest}"

    def __str__(self) -> str:  # pragma: no cover - 只为日志好看
        return self.as_str()


# --------------------------------------------------------------------------- #
# 存储
# --------------------------------------------------------------------------- #


class DuplicateInFlight(RuntimeError):
    """同一个请求正在被执行。**快速拒绝**，不排队等。

    排队等意味着并发全部堆在入口，而入口是最不该堆东西的地方；
    调用方拿到它就该退避（或者干脆什么都不做，因为这个请求已经在做了）。
    """


class EntryState(StrEnum):
    # `NEW` 只作为 `begin()` 的**返回语义**出现：表示"这次归你执行"，
    # 存储里对应的状态是 `IN_FLIGHT`。两者分开是因为"我拿到了执行权"
    # 与"有人正在执行"是调用方视角的两种结论，合成一个会让第一个调用方
    # 以为自己撞车了。
    NEW = "new"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    # 执行失败：**必须放行重试**。把失败也锁死的话，一次上游抖动会让这个请求
    # 永远不能重提——而"重提"正是用户唯一的恢复手段。
    FAILED = "failed"


@dataclass(slots=True)
class IdempotencyEntry:
    key: IdempotencyKey
    state: EntryState
    value: Any = None
    error: str = ""


class IdempotencyStore(Protocol):
    async def begin(self, key: IdempotencyKey) -> IdempotencyEntry:
        """占位。返回 `new` 表示归你执行；`in_flight`/`completed` 表示别执行。"""
        ...

    async def complete(self, key: IdempotencyKey, value: Any) -> None: ...

    async def fail(self, key: IdempotencyKey, error: str) -> None: ...


@dataclass
class InMemoryIdempotencyStore:
    """进程内实现。

    **它只保证单进程幂等。** 多副本部署时必须换成 Redis/DB，
    否则同一个请求打到两个副本就是两次执行——而这件事在单机测试里永远看不到。
    这个限制写在 docstring 里而不是等出事时才发现。
    """

    _entries: dict[str, IdempotencyEntry] = field(default_factory=dict, repr=False)

    async def begin(self, key: IdempotencyKey) -> IdempotencyEntry:
        existing = self._entries.get(key.as_str())

        if existing is not None and existing.state is not EntryState.FAILED:
            return existing

        # 首次，或者上一次失败了（失败不占用幂等位，否则一次上游抖动会让
        # 这个请求永远不能重提——而"重提"正是用户唯一的恢复手段）。
        self._entries[key.as_str()] = IdempotencyEntry(key, EntryState.IN_FLIGHT)
        return IdempotencyEntry(key, EntryState.NEW)

    async def begin_or_raise(self, key: IdempotencyKey) -> IdempotencyEntry:
        """给 HTTP 层用：`in_flight` 直接抛 409。"""

        entry = await self.begin(key)
        if entry.state is EntryState.IN_FLIGHT:
            raise DuplicateInFlight(f"请求 {key.as_str()} 正在执行中，不要重复提交")
        return entry

    async def complete(self, key: IdempotencyKey, value: Any) -> None:
        self._entries[key.as_str()] = IdempotencyEntry(key, EntryState.COMPLETED, value=value)

    async def fail(self, key: IdempotencyKey, error: str) -> None:
        self._entries[key.as_str()] = IdempotencyEntry(key, EntryState.FAILED, error=error)

    def snapshot(self) -> Mapping[str, EntryState]:
        return {key: entry.state for key, entry in self._entries.items()}

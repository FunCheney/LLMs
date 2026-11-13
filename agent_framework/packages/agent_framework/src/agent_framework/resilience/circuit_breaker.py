"""熔断器（W11 D2）。放在 **`LLMGateway` 之外**，不重复实现重试。

## 它不做什么（比"它做什么"更重要）

| 职责 | 归谁 | 为什么不能挪 |
|---|---|---|
| 重试、退避、换候选 | `LLMGateway`（ADR-003） | 那里有 `retryable` 的唯一判定 |
| 并发准入 | `ConcurrencyLimiter`（W2） | 那里有背压与 503 |
| **连续失败后停止调用上游** | 本文件 | 这是唯一一个"决定不再尝试"的地方 |

如果熔断器自己重试，一次故障会被放大成 `threshold × max_attempts` 次调用——
压测第一分钟就能把上游打死，而日志里看起来只是"重试有点多"。
所以 `call()` 的失败计数单位是**一次 `gateway.chat()`**，而那次调用内部可能已经重试了 3 次。

## 三个状态与一个必须说清的细节

```
CLOSED --连续失败达阈值--> OPEN --超过 reset_after_s--> HALF_OPEN --探测成功--> CLOSED
                                    ^                        |
                                    └──── 探测失败 ───────────┘
```

**HALF_OPEN 只放行一个探测。** 放行多个的话，上游刚恢复就被 N 个并发探测同时打，
很容易再次全失败 → 立刻回到 OPEN，形成"永远探测不成功"的抖动。
这一点用 `_probing` 布尔而不是计数实现：计数在异常路径上容易漏减，漏减一次就永久卡住。

## 哪些错误算"上游失败"

只统计**表明上游不健康**的：超时、5xx、限流、上游鉴权失败。
`BadRequestError` / `ContentFilteredError` 不算——那是我方请求或内容的问题，
把它们算进去会让"我们自己写错参数"变成"上游被熔断"，故障域直接跑偏。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from ..llm.errors import (
    CircuitOpenError,
    LLMError,
    RateLimitError,
    UpstreamAuthError,
    UpstreamServerError,
    UpstreamTimeoutError,
)

__all__ = [
    "CLOSED",
    "HALF_OPEN",
    "OPEN",
    "CircuitBreaker",
    "CircuitState",
    "CircuitStats",
    "counts_as_failure",
]

T = TypeVar("T")

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# 只统计这些。**白名单而不是黑名单**：新增一种上游错误时，
# 默认不计数比默认计数安全——"少熔断一次"的代价是慢，"多熔断一次"的代价是全挂。
_FAILURE_TYPES: tuple[type[LLMError], ...] = (
    UpstreamTimeoutError,
    UpstreamServerError,
    RateLimitError,
    UpstreamAuthError,
)


def counts_as_failure(exc: BaseException) -> bool:
    """这次失败是否说明"上游不健康"。"""

    return isinstance(exc, _FAILURE_TYPES)


@dataclass
class CircuitStats:
    """给看板/日志用的快照。`rejected` 是**被快速拒绝**的次数——

    它与"失败次数"必须分开看：被拒绝说明熔断在起作用（省了上游调用），
    而失败次数说明上游还没好。合成一个数会让"熔断生效"看起来像"故障变多"。
    """

    state: str = CLOSED
    failures: int = 0
    successes: int = 0
    rejected: int = 0
    trips: int = 0
    last_error: str = ""
    opened_at: float | None = None


@dataclass
class CircuitBreaker:
    """连续失败 N 次打开，冷却 `reset_after_s` 后放一个探测。

    `clock` 可注入：测试要构造"冷却 30 秒后"，靠真 sleep 会让整个测试套变慢，
    而一旦有人给测试加 sleep，下一步一定是把它 skip 掉。
    """

    failure_threshold: int = 5
    reset_after_s: float = 30.0
    clock: Callable[[], float] = field(default=time.monotonic)

    # `init=False`：这些是运行时状态，不是配置。允许从构造参数传进去的话，
    # 会出现"构造时指定成 open"这种既没用又让审计说不清的用法。
    _state: str = field(default=CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _probing: bool = field(default=False, init=False)
    _stats: CircuitStats = field(default_factory=CircuitStats, init=False, repr=False)

    # -- 观测 --------------------------------------------------------------- #

    @property
    def state(self) -> str:
        """读状态时会**顺带**检查冷却是否结束。

        不这样做的话，"冷却结束后仍显示 open"要等到下一次调用才纠正，
        而看板上那条 open 会多挂几十秒，值班会以为冷却没生效。
        """

        if (
            self._state == OPEN
            and self._opened_at is not None
            and self.clock() - self._opened_at >= self.reset_after_s
        ):
            self._state = HALF_OPEN
        return self._state

    @property
    def stats(self) -> CircuitStats:
        self._stats.state = self.state
        self._stats.failures = self._failures
        self._stats.opened_at = self._opened_at
        return self._stats

    # -- 主入口 ------------------------------------------------------------- #

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """执行一次调用。熔断打开时**不调用 `fn`**，直接抛 `CircuitOpenError`。

        注意 `fn` 是零参可调用对象而不是 `(fn, *args)`：省掉参数转发这层，
        就不会出现"熔断器把参数吞了/改了"这类问题——它没有能力改请求。
        """

        state = self.state

        if state == OPEN:
            self._stats.rejected += 1
            raise CircuitOpenError(
                f"熔断打开中（连续失败 {self._failures} 次），"
                f"冷却 {self.reset_after_s:.0f}s 后放行探测"
            )

        if state == HALF_OPEN:
            if self._probing:
                # 探测名额已被占用：这一次快速拒绝，而不是排队等。
                # 排队会让流量在上游刚恢复时瞬间堆起来。
                self._stats.rejected += 1
                raise CircuitOpenError("半开探测进行中，其余请求快速拒绝")
            self._probing = True

        try:
            result = await fn()
        except BaseException as exc:
            self._on_failure(exc)
            raise
        else:
            self._on_success()
            return result
        finally:
            self._probing = False

    # -- 状态迁移 ----------------------------------------------------------- #

    def _on_success(self) -> None:
        self._stats.successes += 1
        self._failures = 0
        self._state = CLOSED
        self._opened_at = None

    def _on_failure(self, exc: BaseException) -> None:
        if not counts_as_failure(exc):
            # 我方的问题不改变上游的健康判断，也不清零已有的失败计数：
            # 清零会让"每 4 次 5xx 夹 1 次参数错"永远熔不断。
            return

        self._failures += 1
        self._stats.last_error = type(exc).__name__

        if self._state == HALF_OPEN:
            # 探测失败 → 立刻回到 OPEN 并重新计时。这里**不**等阈值：
            # 半开阶段本来就是"上游可能还没好"，一次失败就够证明还没好。
            self._trip()
        elif self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = OPEN
        self._opened_at = self.clock()
        self._stats.trips += 1

    # -- 手动干预 ----------------------------------------------------------- #

    def reset(self) -> None:
        """运维手动复位。**要写进审计**：手动复位掩盖了一次真实故障，
        不留痕的话下次复盘会以为它自己恢复的。"""

        self._state = CLOSED
        self._failures = 0
        self._opened_at = None

    def force_open(self) -> None:
        """配合 Kill Switch（W11 D5）：主动断开某条上游，用于演练与止损。"""

        self._trip()

"""按 alias 分信号量的并发闸门（W2 D2，Gate「100 并发错误率 < 1%」的第一道闸）。

为什么不是"限制 QPS"
-------------------
QPS 限制解决的是"上游说我打太快"，那靠 429 + 退避；这里解决的是"我自己的
连接池/内存扛不住"——它只跟在飞请求数有关。两者混成一个数字，压测时会发现
"限了 QPS 仍然 OOM"。

三层覆盖（数字全部来自 models.yaml，代码里不出现常量）
-----------------------------------------------------
alias 覆盖 > provider 覆盖 > 全局默认，且再叠一层"全局总闸"。
两层都要的原因：单模型限额挡住"某一家被打爆"，全局限额挡住"每家都没超、加起来超"。

拿不到就 503
------------
`acquire()` 超时抛 `ServerBusyError`（`retryable=False`），**不会被 LLMGateway 再重试**。
如果这里返回的是可重试错误，过载时 100 并发会被放大成 300 并发。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from ..llm.errors import ServerBusyError

__all__ = ["ConcurrencyLimiter"]

_LOG = logging.getLogger("agent_framework.resilience")


class ConcurrencyLimiter:
    """按 key（alias）计数的信号量组。"""

    def __init__(
        self,
        limits: Mapping[str, int],
        *,
        max_inflight: int | None = None,
        default_limit: int = 8,
        default_timeout_s: float = 0.25,
        retry_after_s: float = 1.0,
    ) -> None:
        if default_limit < 1:
            raise ValueError("default_limit 必须 >= 1")

        self._limits = {key: value for key, value in limits.items() if value > 0}
        self._default_limit = default_limit
        self._default_timeout_s = default_timeout_s
        self._retry_after_s = retry_after_s

        # 全局总闸：None 表示不限（只在测试里用）。
        self._global: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_inflight) if max_inflight else None
        )

        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._in_flight: dict[str, int] = {}
        self._build_guard = asyncio.Lock()

    # -- 查询 ------------------------------------------------------------ #

    def limit_for(self, key: str) -> int:
        return self._limits.get(key, self._default_limit)

    def in_flight(self, key: str | None = None) -> int | dict[str, int]:
        """在飞请求数。给压测脚本与将来的 /metrics 用，别自己再数一遍。"""

        if key is None:
            return dict(self._in_flight)

        return self._in_flight.get(key, 0)

    # -- 获取 / 释放 ----------------------------------------------------- #

    async def acquire(self, key: str, timeout_s: float | None = None) -> None:
        """拿不到（超时）就抛 `ServerBusyError`，并把已拿到的全局槽位还回去。"""

        timeout = self._default_timeout_s if timeout_s is None else timeout_s
        took_global = False

        if self._global is not None:
            await self._take(self._global, key, timeout, scope="global")
            took_global = True

        try:
            semaphore = await self._semaphore(key)
            await self._take(semaphore, key, timeout, scope=key)
        except ServerBusyError:
            # 只拿到总闸、没拿到 key 时必须归还：少还一次，全局槽位就会永久泄漏，
            # 表现是"跑了半小时后所有请求都 503"，而且重启才能恢复。
            if took_global and self._global is not None:
                self._global.release()

            raise

        self._in_flight[key] = self._in_flight.get(key, 0) + 1

    async def release(self, key: str) -> None:
        current = self._in_flight.get(key, 0)

        if current <= 0:
            # 多还一次会让信号量上限悄悄变大，压测结论随之失真；这里宁可吵一点。
            _LOG.warning("release without acquire key=%s", key)
            return

        self._in_flight[key] = current - 1
        semaphore = self._semaphores.get(key)

        if semaphore is not None:
            semaphore.release()

        if self._global is not None:
            self._global.release()

    @asynccontextmanager
    async def slot(self, key: str, timeout_s: float | None = None) -> AsyncIterator[None]:
        """`async with limiter.slot(alias): ...` —— 异常路径也保证释放。"""

        await self.acquire(key, timeout_s)

        try:
            yield
        finally:
            await self.release(key)

    # -- 内部 ------------------------------------------------------------ #

    async def _semaphore(self, key: str) -> asyncio.Semaphore:
        # 建信号量本身要 await（asyncio.Lock），但只有在首次见到该 key 时才需要。
        semaphore = self._semaphores.get(key)

        if semaphore is not None:
            return semaphore

        async with self._build_guard:
            semaphore = self._semaphores.get(key)

            if semaphore is None:
                semaphore = asyncio.Semaphore(self.limit_for(key))
                self._semaphores[key] = semaphore

            return semaphore

    async def _take(
        self,
        semaphore: asyncio.Semaphore,
        key: str,
        timeout: float,
        *,
        scope: str,
    ) -> None:
        started = time.monotonic()

        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        except TimeoutError as exc:
            waited_ms = round((time.monotonic() - started) * 1000, 3)
            _LOG.warning(
                "overloaded key=%s scope=%s in_flight=%s waited_ms=%s",
                key,
                scope,
                self._in_flight.get(key, 0),
                waited_ms,
            )
            raise ServerBusyError(
                f"并发已满（{scope}={key}，在飞 {self._in_flight.get(key, 0)}）；"
                f"请按 Retry-After 退避",
                retry_after=self._retry_after_s,
            ) from exc

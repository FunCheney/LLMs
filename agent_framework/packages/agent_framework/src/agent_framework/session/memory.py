"""内存会话存储（测试 / 离线 / 单进程 demo 用，零依赖）。

刻意不继承什么抽象基类：`SessionStore` 是 `Protocol`，结构化匹配就够了。
继承会逼着实现方 import 契约文件里的私有辅助，把"可替换"变成"必须同族"。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from ..llm.types import Message
from .base import Turn, resolve_request_id, validate_session_id


class InMemorySessionStore:
    """进程内实现。

    只保证**单进程**语义：多副本部署时各自有各自的历史，所以生产必须换 Redis。
    这条写在类型注释里而不是文档里，是因为它决定了"能不能直接上线"。
    """

    def __init__(
        self,
        *,
        max_turns: int = 50,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")

        self._max_turns = max_turns
        self._clock = clock
        self._turns: dict[str, list[Turn]] = {}
        self._seen: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    # -- 读 -------------------------------------------------------------- #

    async def get_history(self, session_id: str) -> list[Message]:
        validate_session_id(session_id)

        async with self._lock:
            turns = self._turns.get(session_id, [])

            # 返回新列表：调用方 append 到结果里不能改到存储里的对象。
            return [
                message for turn in turns for message in turn.messages
            ]

    def turn_count(self, session_id: str) -> int:
        return len(self._turns.get(session_id, []))

    # -- 写 -------------------------------------------------------------- #

    async def append(
        self,
        session_id: str,
        messages: list[Message],
        *,
        request_id: str | None = None,
    ) -> bool:
        validate_session_id(session_id)

        rid = resolve_request_id(request_id)

        async with self._lock:
            seen = self._seen.setdefault(session_id, set())

            if rid in seen:
                # 幂等命中：调用方重试/前端双击不能把历史翻倍。
                return False

            seen.add(rid)
            turns = self._turns.setdefault(session_id, [])
            turns.append(
                Turn(request_id=rid, messages=list(messages), created_at=self._clock())
            )

            # 只留最近 N 轮：不裁剪的话，长会话会把 prompt 预算吃干净（W7 才做摘要，
            # 这里先保证"不会无限增长"这件更基础的事）。
            if len(turns) > self._max_turns:
                del turns[: len(turns) - self._max_turns]

            return True

    async def clear(self, session_id: str) -> None:
        validate_session_id(session_id)

        async with self._lock:
            self._turns.pop(session_id, None)
            self._seen.pop(session_id, None)

    async def aclose(self) -> None:
        async with self._lock:
            self._turns.clear()
            self._seen.clear()
            self._closed = True

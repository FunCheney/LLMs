"""Redis 会话存储（多副本部署时唯一正确的选择）。

Key 设计
--------
    session:{session_id}:turns        LIST，每轮一条 JSON
    session:{session_id}:req:{rid}    STRING，幂等标记（SET NX EX）

为什么幂等标记单独一把 key 而不是"扫描 list 找 request_id"
--------------------------------------------------------
LIST 没有成员索引，`LRANGE 0 -1` 再逐个比是 O(n)，20 轮以后每轮 append 都在做
无谓的全量读取；更有害的是**竞态**：两个并发请求同时扫描都发现"没有"，
于是各写一次。`SET NX` 是原子的，天然只有一个人能拿到 "新" 这个结论。

为什么 TTL 不用永久 key
----------------------
会话是"热数据"，不是档案。永久 key 的结局是 Redis 内存被半年前的对话占满，
然后有人上来 `FLUSHALL`——那才是真的数据丢失。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import ConfigError
from ..llm.types import Message
from .base import (
    Turn,
    decode_turn,
    encode_turn,
    resolve_request_id,
    validate_session_id,
)

__all__ = ["RedisSessionStore"]

_LOG = logging.getLogger("agent_framework.session")


class RedisSessionStore:
    """`redis.asyncio` 实现。

    `redis` 是可选依赖（`uv sync --extra redis`）：把它设成硬依赖会让"只想跑单测"
    的人也必须装 redis-py，而 CI 里还会因为连不上服务器挂掉。
    """

    def __init__(
        self,
        url: str,
        *,
        ttl_s: int = 86400,
        max_turns: int = 50,
        prefix: str = "session",
        client: Any = None,
    ) -> None:
        if ttl_s < 60:
            raise ValueError("ttl_s 至少 60 秒：更短的 TTL 会让多轮对话随机失忆")

        self._url = url
        self._ttl_s = ttl_s
        self._max_turns = max_turns
        self._prefix = prefix
        self._client: Any = client

    # -- 生命周期 -------------------------------------------------------- #

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - 只在未装额外依赖时触发
            raise ConfigError(
                "RedisSessionStore 需要 redis 依赖：uv sync --extra redis"
            ) from exc

        # 建客户端不建立连接（redis-py 是懒连接），所以这里不会拖慢启动。
        self._client = Redis.from_url(self._url, decode_responses=True)

        return self._client

    async def ping(self) -> bool:
        return bool(await self._ensure_client().ping())

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- 读 -------------------------------------------------------------- #

    async def get_history(self, session_id: str) -> list[Message]:
        validate_session_id(session_id)

        client = self._ensure_client()
        raw_items = await client.lrange(self._turns_key(session_id), 0, -1)

        messages: list[Message] = []

        for raw in raw_items:
            messages.extend(decode_turn(raw).messages)

        return messages

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
        client = self._ensure_client()
        marker = self._request_key(session_id, rid)

        # NX + EX：原子地"占坑并过期"。返回 None 说明这个 request_id 来过。
        claimed = await client.set(marker, "1", nx=True, ex=self._ttl_s)

        if not claimed:
            return False

        turn = Turn(request_id=rid, messages=list(messages), created_at=time.time())
        turns_key = self._turns_key(session_id)

        await client.rpush(turns_key, encode_turn(turn))
        # 裁剪 + 续期放在一次 pipeline 里：拆开写会出现"写进去了但没续期"，
        # 表现为会话在一小时后突然断开（而日志里一切正常）。
        async with client.pipeline(transaction=False) as pipe:
            pipe.ltrim(turns_key, -self._max_turns, -1)
            pipe.expire(turns_key, self._ttl_s)
            await pipe.execute()

        return True

    async def clear(self, session_id: str) -> None:
        validate_session_id(session_id)

        client = self._ensure_client()
        pattern = f"{self._prefix}:{session_id}:*"
        keys = [key async for key in client.scan_iter(match=pattern)]

        if keys:
            await client.delete(*keys)

    # -- 内部 ------------------------------------------------------------ #

    def _turns_key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}:turns"

    def _request_key(self, session_id: str, request_id: str) -> str:
        return f"{self._prefix}:{session_id}:req:{request_id}"

"""会话契约与序列化（W2 D1）。

为什么把 `encode_turn` / `decode_turn` 放在契约层而不是 `redis.py`：
它们是"内部存储格式"的唯一定义。放在某个实现里，第二个实现（InMemory / 将来的
Postgres）就会复制一遍，然后有一处忘了带 `provider_meta`——这种 bug 只会在
真实多轮对话的第二轮才暴露。

序列化用 `model_dump(mode="json")` 是**允许**的：这是内部存储，不是对外响应。
Day4 的红线是"不许把 `LLMResponse.model_dump()` 当 HTTP 响应体"，两者不是一回事。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from ..llm.types import Message
from ..observability import current_request_id

__all__ = [
    "InvalidSessionIdError",
    "SESSION_ID_PATTERN",
    "SessionError",
    "SessionStore",
    "Turn",
    "decode_turn",
    "encode_turn",
    "resolve_request_id",
    "validate_session_id",
]

# 会话 id 会拼进 Redis key。不校验等于允许调用方写 `session:../../x` 这类 key，
# 与 request_id 的头校验是同一类问题（日志/key 注入）。
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionError(ValueError):
    """会话层错误基类。"""


class InvalidSessionIdError(SessionError):
    """会话 id 非法 → 调用方输入问题（对外 400，不是 500）。"""


@dataclass(frozen=True, slots=True)
class Turn:
    """一轮对话：调用方一次 append 进来的消息 + 幂等键 + 时间戳。"""

    request_id: str
    messages: list[Message]
    created_at: float


class SessionStore(Protocol):
    """会话存储协议。

    两个实现（InMemory / Redis）必须同签名，否则单测与生产走的是两条路，
    而"单测通过"就变成了对生产的无效承诺。
    """

    async def get_history(self, session_id: str) -> list[Message]:
        """取该会话归一化后的历史（已是内部 `Message`，不含厂商字段）。"""
        ...

    async def append(
        self,
        session_id: str,
        messages: list[Message],
        *,
        request_id: str | None = None,
    ) -> bool:
        """追加一轮。

        返回 True = 真的写进去了；False = 这个 `request_id` 已经写过（幂等命中）。
        必须支持幂等：同一 request_id 重复 append 不能让历史翻倍。
        """
        ...

    async def aclose(self) -> None: ...


def validate_session_id(session_id: str) -> str:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise InvalidSessionIdError(
            f"非法 session_id：{session_id!r}；只允许 [A-Za-z0-9_-]，长度 1–64"
        )

    return session_id


def resolve_request_id(request_id: str | None) -> str:
    """幂等键的优先级：显式入参 → contextvar → 新 uuid。

    第三档必须有：脚本/评测/队列消费者直接调 `append` 时可能没有请求上下文，
    没有兜底就会产出 `request_id=""`，第二次 append 立刻被误判成重复。
    放在契约层是为了让两个实现共用一份——各写一份，迟早一处忘了兜底。
    """

    if request_id:
        return request_id

    return current_request_id() or f"local_{uuid4().hex[:12]}"


def encode_turn(turn: Turn) -> str:
    """一轮 → 一行 JSON。

    `sort_keys` 让夹具/日志可 diff；`ensure_ascii=False` 让中文在 redis-cli 里可读
    （排查时能直接 `LRANGE` 看一眼，比接调试器快得多）。
    """

    payload: dict[str, Any] = {
        "request_id": turn.request_id,
        "created_at": turn.created_at,
        "messages": [message.model_dump(mode="json") for message in turn.messages],
    }

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def decode_turn(raw: str) -> Turn:
    """一行 JSON → 一轮。

    解析失败抛 `SessionError` 并带上原文片段：历史脏数据是运维会遇到的真实情况，
    报错里没有可定位的片段等于让人去翻整个 list。
    """

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise SessionError(f"会话历史不是合法 JSON：{raw[:120]!r}") from exc

    if not isinstance(payload, dict):
        raise SessionError(f"会话历史必须是对象：{raw[:120]!r}")

    try:
        messages = [Message.model_validate(item) for item in payload["messages"]]
        request_id = str(payload["request_id"])
        created_at = float(payload.get("created_at", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionError(f"会话历史字段缺失或非法：{raw[:120]!r}") from exc

    return Turn(request_id=request_id, messages=messages, created_at=created_at)

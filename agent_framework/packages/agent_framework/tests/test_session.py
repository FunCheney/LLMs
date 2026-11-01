"""会话层测试（W2 D1）。

守四件事：多轮顺序、`request_id` 幂等、会话隔离、`provider_meta` 不丢。
最后一条是 ADR-007 的直接证据：DeepSeek 第二轮要回传 `reasoning_content`，
存在 Redis 里被丢掉，线上表现是"聊到第二轮就 400"。
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework.config import Settings
from agent_framework.llm.types import Message, MessageRole
from agent_framework.session import (
    InMemorySessionStore,
    InvalidSessionIdError,
    SessionError,
    Turn,
    build_session_store,
    decode_turn,
    encode_turn,
    resolve_request_id,
)
from agent_framework.session.base import validate_session_id


def user(text: str) -> Message:
    return Message(role=MessageRole.USER, content=text)


def assistant(text: str, *, provider_meta: dict[str, Any] | None = None) -> Message:
    return Message(role=MessageRole.ASSISTANT, content=text, provider_meta=provider_meta)


# --------------------------------------------------------------------------- #
# 读写
# --------------------------------------------------------------------------- #


async def test_two_turns_return_four_messages_in_order() -> None:
    store = InMemorySessionStore()

    assert await store.append("s1", [user("你好"), assistant("你好，我是助手")], request_id="r1")
    assert await store.append("s1", [user("继续")], request_id="r2")

    history = await store.get_history("s1")

    assert [message.content for message in history] == [
        "你好",
        "你好，我是助手",
        "继续",
    ]


async def test_append_is_idempotent_by_request_id() -> None:
    store = InMemorySessionStore()
    turn = [user("只应该出现一次")]

    assert await store.append("s1", turn, request_id="req-dup") is True
    assert await store.append("s1", turn, request_id="req-dup") is False
    assert store.turn_count("s1") == 1

    history = await store.get_history("s1")
    assert len(history) == 1


async def test_request_id_falls_back_to_generated_when_missing() -> None:
    """没有上下文时不能让 rid 变成空串，否则第二次 append 会被误判成重复。"""

    store = InMemorySessionStore()

    assert await store.append("s1", [user("a")]) is True
    assert await store.append("s1", [user("b")]) is True
    assert store.turn_count("s1") == 2


async def test_provider_meta_survives_memory_round_trip() -> None:
    store = InMemorySessionStore()
    meta = {"reasoning_content": "用户要的是最新进展"}

    await store.append("s1", [assistant("我先检索一下", provider_meta=meta)], request_id="r1")

    history = await store.get_history("s1")

    assert history[0].provider_meta == meta


async def test_sessions_are_isolated() -> None:
    store = InMemorySessionStore()

    await store.append("alice", [user("Alice 的问题")], request_id="r1")
    await store.append("bob", [user("Bob 的问题")], request_id="r2")

    assert [m.content for m in await store.get_history("alice")] == ["Alice 的问题"]
    assert [m.content for m in await store.get_history("bob")] == ["Bob 的问题"]


async def test_trim_keeps_the_most_recent_turns() -> None:
    store = InMemorySessionStore(max_turns=2)

    for index in range(4):
        await store.append("s1", [user(f"t{index}")], request_id=f"r{index}")

    history = await store.get_history("s1")

    assert [message.content for message in history] == ["t2", "t3"]


async def test_aclose_clears_state() -> None:
    store = InMemorySessionStore()
    await store.append("s1", [user("hi")], request_id="r1")

    await store.aclose()

    assert await store.get_history("s1") == []


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["", "with space", "a" * 65, "colon:key", "中文"])
async def test_invalid_session_id_is_rejected(bad: str) -> None:
    """会话 id 会拼进 Redis key，不校验等于允许调用方写任意 key。"""

    store = InMemorySessionStore()

    with pytest.raises(InvalidSessionIdError):
        await store.get_history(bad)


def test_validate_session_id_accepts_uuid_like_values() -> None:
    assert validate_session_id("sess_0123456789abcdef-ABC") == "sess_0123456789abcdef-ABC"


# --------------------------------------------------------------------------- #
# 序列化（内部存储格式的唯一定义）
# --------------------------------------------------------------------------- #


def test_encode_decode_round_trip_keeps_provider_meta() -> None:
    turn = Turn(
        request_id="r1",
        messages=[assistant("答案", provider_meta={"thinking": {"signature": "sig"}})],
        created_at=1700000000.0,
    )

    restored = decode_turn(encode_turn(turn))

    assert restored.request_id == "r1"
    assert restored.created_at == 1700000000.0
    assert restored.messages[0].provider_meta == {"thinking": {"signature": "sig"}}


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        '{"messages": []}',
        '{"request_id": "r", "messages": [{"role": "wat"}]}',
    ],
)
def test_decode_rejects_corrupt_payload(raw: str) -> None:
    with pytest.raises(SessionError):
        decode_turn(raw)


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #


def test_factory_defaults_to_memory() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert isinstance(build_session_store(settings), InMemorySessionStore)


def test_resolve_request_id_prefers_explicit_value() -> None:
    assert resolve_request_id("explicit") == "explicit"
    assert resolve_request_id(None).startswith("local_")

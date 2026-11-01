"""并发闸门测试（W2 D2 / Gate「过载不自重试」）。

最关键的一条不是"限流生效"，而是**限流生效时 LLMGateway 根本没被调用**：
如果过载错误是可重试的，100 并发会被放大成 300 并发，压测第一分钟就把自己打死。
`test_busy_does_not_reach_gateway` 就是这条红线的守卫。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agent_framework.bootstrap import build_limiter
from agent_framework.config import load_catalog
from agent_framework.llm.errors import ServerBusyError, error_code_for, http_status_for
from agent_framework.resilience import ConcurrencyLimiter


async def test_limit_one_blocks_second_acquire() -> None:
    limiter = ConcurrencyLimiter({"alias": 1}, default_timeout_s=0.01)

    await limiter.acquire("alias")

    with pytest.raises(ServerBusyError):
        await limiter.acquire("alias")

    await limiter.release("alias")
    await limiter.acquire("alias")  # 释放后必须能再拿到
    await limiter.release("alias")


async def test_busy_error_shape() -> None:
    limiter = ConcurrencyLimiter({"alias": 1}, default_timeout_s=0.01, retry_after_s=3.0)
    await limiter.acquire("alias")

    with pytest.raises(ServerBusyError) as info:
        await limiter.acquire("alias")

    error = info.value

    # retryable 必须 False：否则 LLMGateway 会把入口过载当成上游抖动再打一遍。
    assert error.retryable is False
    assert error.retry_after == 3.0
    assert error_code_for(error) == "SERVER_BUSY"
    assert http_status_for(error) == 503

    await limiter.release("alias")


async def test_global_cap_applies_across_keys() -> None:
    limiter = ConcurrencyLimiter({"a": 5, "b": 5}, max_inflight=1, default_timeout_s=0.01)

    await limiter.acquire("a")

    with pytest.raises(ServerBusyError):
        await limiter.acquire("b")

    await limiter.release("a")

    # 全局槽位被正确归还：不然这里会 503，而且重启前不会恢复。
    await limiter.acquire("b")
    await limiter.release("b")


async def test_key_timeout_returns_global_slot() -> None:
    """只拿到总闸、没拿到 key 时必须归还总闸，否则全局额度永久泄漏。"""

    limiter = ConcurrencyLimiter({"a": 1}, max_inflight=2, default_timeout_s=0.01)

    await limiter.acquire("a")  # 全局 2 → 1，key a 用满

    with pytest.raises(ServerBusyError):
        # 这次会先把全局拿成 0，再在 key 上超时；必须把全局还回去。
        await limiter.acquire("a")

    # 如果没有归还，全局只剩 0，下面这一把必然 503。
    await limiter.acquire("b")

    await limiter.release("b")
    await limiter.release("a")


async def test_slot_releases_on_exception() -> None:
    limiter = ConcurrencyLimiter({"alias": 1}, default_timeout_s=0.01)

    with pytest.raises(RuntimeError):
        async with limiter.slot("alias"):
            raise RuntimeError("boom")

    assert limiter.in_flight("alias") == 0

    await limiter.acquire("alias")
    await limiter.release("alias")


async def test_in_flight_snapshot() -> None:
    limiter = ConcurrencyLimiter({"a": 2, "b": 2}, default_timeout_s=0.01)

    async with limiter.slot("a"):
        assert limiter.in_flight("a") == 1
        assert limiter.in_flight() == {"a": 1}

    assert limiter.in_flight("a") == 0


async def test_busy_does_not_reach_gateway() -> None:
    """过载时 gateway 的调用次数必须保持 0（说明没有重试放大）。"""

    from gateway_helpers import StubClient, build_test_gateway, make_catalog

    client = StubClient("primary")
    gateway = build_test_gateway(
        make_catalog(["primary"]), {"primary": client}
    )
    limiter = ConcurrencyLimiter({"primary": 1}, default_timeout_s=0.01)
    await limiter.acquire("primary")

    with pytest.raises(ServerBusyError):
        async with limiter.slot("primary"):
            await gateway.chat(  # pragma: no cover - 不该被执行
                _call()
            )

    assert client.chat_calls == 0

    await limiter.release("primary")


def test_limiter_reads_limits_from_yaml() -> None:
    """数字只在 models.yaml 里：代码里出现 `Semaphore(8)` 就无法按档位调。"""

    catalog = load_catalog()
    limiter = build_limiter(catalog)

    assert (
        limiter.limit_for("claude-sonnet-4-6")
        == catalog.limits.per_alias["claude-sonnet-4-6"]
    )
    assert limiter.limit_for("qwen3.7-plus") == catalog.limits.per_provider["qwen"]
    assert limiter.limit_for("unknown-alias") == catalog.limits.max_inflight


async def test_waiting_acquire_succeeds_when_slot_frees() -> None:
    limiter = ConcurrencyLimiter({"alias": 1}, default_timeout_s=1.0)
    await limiter.acquire("alias")

    async def release_soon() -> None:
        await asyncio.sleep(0.02)
        await limiter.release("alias")

    task = asyncio.create_task(release_soon())
    await limiter.acquire("alias")  # 应该等到释放后拿到
    await task

    assert limiter.in_flight("alias") == 1
    await limiter.release("alias")


def _call() -> Any:
    from agent_framework.llm.client import ChatCall
    from agent_framework.llm.types import Message, MessageRole

    return ChatCall(messages=[Message(role=MessageRole.USER, content="hi")])

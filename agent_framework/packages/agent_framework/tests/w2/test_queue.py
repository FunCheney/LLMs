"""任务队列测试（W2 D2）：状态机、失败可见、队列满、trace 贯穿。

守两件事：
1. **失败必须可见**：`failed` + `{code, message}`，不能让任务永远停在 running；
2. **消费者要带上下文**：队列不在 HTTP 里跑，不显式进 scope，任务内部的 LLM 调用
   就会产出空 trace_id——trace 从这里断掉，W4 的根因归因直接失效。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agent_framework.llm.client import ChatCall
from agent_framework.llm.errors import (
    RateLimitError,
    ServerBusyError,
    TaskNotFoundError,
)
from agent_framework.llm.types import Message, MessageRole
from agent_framework.observability import current_request_id, current_trace_id
from agent_framework.queue import InMemoryTaskQueue, TaskStatus


def make_call(text: str = "总结一下这篇文章") -> ChatCall:
    return ChatCall(messages=[Message(role=MessageRole.USER, content=text)])


async def wait_for(
    queue: InMemoryTaskQueue, task_id: str, status: TaskStatus, *, timeout: float = 2.0
) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout

    while asyncio.get_running_loop().time() < deadline:
        # 必须让出事件循环：`queue.get()` 没有 await 点，纯轮询会把消费者饿死，
        # 表现是"任务永远停在 queued"——那是测试自己的 bug，不是队列的。
        await asyncio.sleep(0.001)
        record = await queue.get(task_id)

        if record.status is status:
            return record

    record = await queue.get(task_id)
    raise AssertionError(f"任务未在 {timeout}s 内到达 {status}，实际 {record.status}")


async def test_enqueue_then_succeeded() -> None:
    async def handler(call: ChatCall) -> dict[str, Any]:
        return {"echo": call.messages[-1].content}

    queue = InMemoryTaskQueue(handler, workers=1)
    record = await queue.enqueue(make_call("你好"), request_id="req_1")

    assert record.status is TaskStatus.QUEUED

    done = await wait_for(queue, record.task_id, TaskStatus.SUCCEEDED)

    assert done.result == {"echo": "你好"}
    assert done.started_at is not None
    assert done.finished_at is not None
    assert done.request_id == "req_1"

    await queue.aclose()


async def test_llm_error_marks_failed_with_code() -> None:
    async def handler(call: ChatCall) -> dict[str, Any]:
        raise RateLimitError("上游限流", provider="qwen", retry_after=1.0)

    queue = InMemoryTaskQueue(handler, workers=1)
    record = await queue.enqueue(make_call())

    failed = await wait_for(queue, record.task_id, TaskStatus.FAILED)

    assert failed.error is not None
    assert failed.error["code"] == "RATE_LIMITED"
    assert failed.result is None

    await queue.aclose()


async def test_crash_marks_failed_internal() -> None:
    async def handler(call: ChatCall) -> dict[str, Any]:
        raise RuntimeError("代码 bug")

    queue = InMemoryTaskQueue(handler, workers=1)
    record = await queue.enqueue(make_call())

    failed = await wait_for(queue, record.task_id, TaskStatus.FAILED)

    assert failed.error is not None
    assert failed.error["code"] == "INTERNAL"

    await queue.aclose()


async def test_unknown_task_raises_not_found() -> None:
    queue = InMemoryTaskQueue(_noop, workers=1)

    with pytest.raises(TaskNotFoundError):
        await queue.get("task_missing")


async def test_worker_runs_with_request_and_trace_scope() -> None:
    seen: dict[str, str] = {}

    async def handler(call: ChatCall) -> dict[str, Any]:
        # 消费者不在 HTTP 上下文里：这两个值必须由 queue 显式传下去。
        seen["request_id"] = current_request_id()
        seen["trace_id"] = current_trace_id()

        return {"ok": True}

    queue = InMemoryTaskQueue(handler, workers=1)
    record = await queue.enqueue(make_call(), request_id="req-ctx", trace_id="trace-ctx")

    await wait_for(queue, record.task_id, TaskStatus.SUCCEEDED)

    assert seen == {"request_id": "req-ctx", "trace_id": "trace-ctx"}

    await queue.aclose()


async def test_queue_full_raises_busy_with_retry_after() -> None:
    gate = asyncio.Event()

    async def handler(call: ChatCall) -> dict[str, Any]:
        await gate.wait()

        return {"ok": True}

    queue = InMemoryTaskQueue(handler, max_size=1, workers=1)

    first = await queue.enqueue(make_call("1"))
    await wait_for(queue, first.task_id, TaskStatus.RUNNING)  # 消费者已占住

    await queue.enqueue(make_call("2"))  # 占满队列的唯一位置

    with pytest.raises(ServerBusyError) as info:
        await queue.enqueue(make_call("3"))

    assert info.value.retry_after is not None
    assert info.value.retryable is False

    gate.set()
    await queue.aclose()


async def test_enqueue_after_close_raises() -> None:
    queue = InMemoryTaskQueue(_noop, workers=1)
    await queue.aclose()

    with pytest.raises(ServerBusyError):
        await queue.enqueue(make_call())


async def test_snapshot_reports_pending_and_workers() -> None:
    gate = asyncio.Event()

    async def handler(call: ChatCall) -> dict[str, Any]:
        await gate.wait()

        return {"ok": True}

    queue = InMemoryTaskQueue(handler, max_size=4, workers=2)
    await queue.enqueue(make_call())

    snapshot = queue.snapshot()

    assert snapshot["workers"] == 2
    assert snapshot["closed"] is False

    gate.set()
    await queue.aclose()


async def test_status_progression_is_monotonic() -> None:
    """queued → running → succeeded，不能跳步。"""

    steps: list[str] = []

    async def handler(call: ChatCall) -> dict[str, Any]:
        await asyncio.sleep(0.005)

        return {"ok": True}

    queue = InMemoryTaskQueue(handler, workers=1)
    record = await queue.enqueue(make_call())
    steps.append(record.status.value)

    running = await wait_for(queue, record.task_id, TaskStatus.RUNNING)
    steps.append(running.status.value)

    done = await wait_for(queue, record.task_id, TaskStatus.SUCCEEDED)
    steps.append(done.status.value)

    assert steps == ["queued", "running", "succeeded"]

    await queue.aclose()


async def _noop(call: ChatCall) -> dict[str, Any]:
    return {"ok": True}

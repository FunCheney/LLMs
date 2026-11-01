"""`POST /v1/tasks` / `GET /v1/tasks/{id}` 的 HTTP 层测试（W2 D2）。

重点不是"接口能返回 202"，而是三条边界：
1. 入队前先 `precheck()`：注定失败的请求不该占队列槽位；
2. 队列满 → **503 + Retry-After**，且 `LLMGateway` 一次都没被调用（不放大）；
3. 未知 task_id → **404**，不是 500。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from agent_framework.llm.client import LLMGateway
from agent_framework.queue.inmemory import InMemoryTaskQueue
from agent_framework.resilience import ConcurrencyLimiter
from agent_framework.server.app import create_app
from agent_framework.server.routes_task import make_task_handler
from gateway_helpers import (
    StubClient,
    build_test_gateway,
    make_catalog,
    make_settings,
    text_response,
)

USER_BODY = {"messages": [{"role": "user", "content": "写一段商品文案"}]}


def gateway_with(client: StubClient) -> LLMGateway:
    return build_test_gateway(
        make_catalog(["primary"]), {"primary": client}, settings=make_settings(max_attempts=1)
    )


@pytest.fixture
async def client_factory() -> AsyncIterator[Any]:
    opened: list[httpx.AsyncClient] = []

    async def build(app: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        )
        opened.append(client)

        return client

    try:
        yield build
    finally:
        for client in opened:
            await client.aclose()


async def _wait_succeeded(http: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
    for _ in range(200):
        body = (await http.get(f"/v1/tasks/{task_id}")).json()

        if body["status"] in ("succeeded", "failed"):
            return body

        await asyncio.sleep(0.005)

    raise AssertionError(f"任务未完成：{body}")


async def test_submit_returns_202_and_result_is_queryable(client_factory: Any) -> None:
    client = StubClient("primary", chat_results=[text_response("这是一版文案")])
    gateway = gateway_with(client)
    app = create_app(gateway=gateway)
    http = await client_factory(app)

    response = await http.post("/v1/tasks", json=USER_BODY)

    assert response.status_code == 202

    submitted = response.json()
    assert submitted["status"] == "queued"
    assert submitted["task_id"].startswith("task_")
    assert response.headers["X-Task-ID"] == submitted["task_id"]

    done = await _wait_succeeded(http, submitted["task_id"])

    assert done["status"] == "succeeded"
    assert done["result"]["text"] == "这是一版文案"
    assert done["trace_id"]

    await app.state.task_queue.aclose()


async def test_unknown_task_returns_404(client_factory: Any) -> None:
    client = StubClient("primary")
    app = create_app(gateway=gateway_with(client))
    http = await client_factory(app)

    response = await http.get("/v1/tasks/task_ghost")

    assert response.status_code == 404
    assert response.json()["code"] == "TASK_NOT_FOUND"

    await app.state.task_queue.aclose()


async def test_stream_true_is_rejected_before_enqueue(client_factory: Any) -> None:
    client = StubClient("primary")
    app = create_app(gateway=gateway_with(client))
    http = await client_factory(app)

    response = await http.post("/v1/tasks", json={**USER_BODY, "stream": True})

    assert response.status_code == 400
    assert response.json()["code"] == "BAD_REQUEST"
    assert app.state.task_queue.snapshot()["records"] == 0

    await app.state.task_queue.aclose()


async def test_unknown_model_is_rejected_before_enqueue(client_factory: Any) -> None:
    """入队前 precheck：输入错误不该占队列槽位，更不该排 30 秒后才 failed。"""

    client = StubClient("primary")
    app = create_app(gateway=gateway_with(client))
    http = await client_factory(app)

    response = await http.post("/v1/tasks", json={**USER_BODY, "model": "ghost"})

    assert response.status_code == 400
    assert app.state.task_queue.snapshot()["records"] == 0

    await app.state.task_queue.aclose()


async def test_queue_full_returns_503_with_retry_after(client_factory: Any) -> None:
    gate = asyncio.Event()

    async def slow_handler(call: Any) -> dict[str, Any]:
        await gate.wait()

        return {"ok": True}

    gateway = gateway_with(StubClient("primary"))
    queue = InMemoryTaskQueue(slow_handler, max_size=1, workers=1)
    app = create_app(gateway=gateway, task_queue=queue)
    http = await client_factory(app)

    first = await http.post("/v1/tasks", json=USER_BODY)
    assert first.status_code == 202

    # 等消费者占住第一个任务，队列的 1 个位置才真正空出来
    for _ in range(200):
        running = (await queue.get(first.json()["task_id"])).status.value == "running"

        if running and queue.snapshot()["pending"] == 0:
            break

        await asyncio.sleep(0.005)

    second = await http.post("/v1/tasks", json=USER_BODY)
    assert second.status_code == 202

    third = await http.post("/v1/tasks", json=USER_BODY)

    assert third.status_code == 503
    assert third.json()["code"] == "SERVER_BUSY"
    assert third.headers["Retry-After"]

    gate.set()
    await queue.aclose()


async def test_overload_returns_503_without_touching_gateway(client_factory: Any) -> None:
    """入口过载必须是 503，而且 gateway 的调用次数保持 0（没有重试放大）。"""

    client = StubClient("primary", chat_results=[text_response("never")])
    limiter = ConcurrencyLimiter({"primary": 1}, default_timeout_s=0.01, retry_after_s=2.0)
    await limiter.acquire("primary")

    app = create_app(gateway=gateway_with(client), limiter=limiter)
    http = await client_factory(app)

    response = await http.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 503
    assert response.json()["code"] == "SERVER_BUSY"
    assert response.headers["Retry-After"] == "2"
    assert client.chat_calls == 0

    await limiter.release("primary")
    await app.state.task_queue.aclose()


async def test_health_reports_queue_and_in_flight(client_factory: Any) -> None:
    client = StubClient("primary")
    app = create_app(gateway=gateway_with(client))
    http = await client_factory(app)

    body = (await http.get("/health")).json()

    assert body["status"] == "ok"
    assert body["session_backend"] == "memory"
    assert body["queue"]["workers"] == 0  # 懒启动：没有任务就不建后台 task
    assert body["in_flight"] == {}

    await app.state.task_queue.aclose()


async def test_task_handler_projects_response_without_provider_meta() -> None:
    """任务结果同样要投影：不能让 provider_meta 从队列里漏出去。"""

    from agent_framework.llm.types import LLMResponse, Message, MessageRole, Usage

    leaky = LLMResponse(
        id="resp",
        model="m",
        provider="primary",
        message=Message(
            role=MessageRole.ASSISTANT,
            content="答案",
            provider_meta={"reasoning_content": "内部思考"},
        ),
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    client = StubClient("primary", chat_results=[leaky])
    handler = make_task_handler(gateway_with(client))

    from agent_framework.llm.client import ChatCall

    result = await handler(ChatCall(messages=[Message(role=MessageRole.USER, content="hi")]))

    assert result["text"] == "答案"
    assert "provider_meta" not in result
    assert "reasoning_content" not in str(result)

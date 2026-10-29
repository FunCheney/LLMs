"""HTTP 层测试：`POST /v1/chat` 的状态码、投影与帧格式。

用 `httpx.ASGITransport` 在进程内打真 app（含中间件与异常处理器），不起服务、不联网。
这一层最容易犯的两个错都在这里守着：
- 能力错误返回 200 + error 帧（因为 StreamingResponse 先发 header）；
- 直接把 `model_dump()` 当响应体（会把 provider_meta 漏给调用方）。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from agent_framework.llm.base import Capability
from agent_framework.llm.client import LLMGateway
from agent_framework.llm.errors import RateLimitError, UpstreamAuthError
from agent_framework.llm.types import StreamEvent, StreamEventType, ToolCall
from agent_framework.server.app import create_app
from agent_framework.server.sse import decode_frame
from gateway_helpers import (
    StubClient,
    build_test_gateway,
    make_catalog,
    make_settings,
    message_end,
    text_delta,
    text_response,
)

USER_BODY = {"messages": [{"role": "user", "content": "你好"}]}


@pytest.fixture
async def client_factory() -> AsyncIterator[Any]:
    opened: list[httpx.AsyncClient] = []

    async def build(
        gateway: LLMGateway,
    ) -> httpx.AsyncClient:
        app = create_app(gateway=gateway)
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


def gateway_with(
    client: StubClient,
    *,
    aliases: tuple[str, ...] = ("primary",),
    max_attempts: int = 1,
) -> LLMGateway:
    return build_test_gateway(
        make_catalog(list(aliases)),
        {alias: client for alias in aliases},
        settings=make_settings(max_attempts=max_attempts),
    )


# --------------------------------------------------------------------------- #
# 非流式
# --------------------------------------------------------------------------- #


async def test_non_stream_returns_projected_body(client_factory: Any) -> None:
    client = StubClient("primary", chat_results=[text_response("你好，我是助手")])
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json=USER_BODY)

    assert response.status_code == 200

    body = response.json()
    assert body["text"] == "你好，我是助手"
    assert body["finish_reason"] == "stop"
    assert body["usage"]["input_tokens"] == 10
    assert body["request_id"].startswith("req_")
    assert response.headers["X-Model-Alias"] == "primary"
    assert response.headers["X-Request-ID"] == body["request_id"]


async def test_provider_meta_is_not_exposed(client_factory: Any) -> None:
    """投影是红线：内部 provider_meta 绝不能出现在响应体里。"""

    from agent_framework.llm.types import LLMResponse, Message, MessageRole, Usage

    leaky = LLMResponse(
        id="resp",
        model="m",
        provider="primary",
        message=Message(
            role=MessageRole.ASSISTANT,
            content="答案",
            provider_meta={"reasoning_content": "我的内部思考"},
        ),
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    client = StubClient("primary", chat_results=[leaky])
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json=USER_BODY)

    assert response.status_code == 200
    assert "reasoning_content" not in response.text
    assert "provider_meta" not in response.text


async def test_tool_call_is_projected(client_factory: Any) -> None:
    from agent_framework.llm.types import FinishReason, LLMResponse, Message, MessageRole

    with_call = LLMResponse(
        id="resp",
        model="m",
        provider="primary",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "MCP"})],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        raw_finish_reason="tool_calls",
    )
    client = StubClient("primary", chat_results=[with_call])
    http = await client_factory(gateway_with(client))

    body = (await http.post("/v1/chat", json=USER_BODY)).json()

    assert body["tool_calls"] == [
        {"id": "call_1", "name": "search", "arguments": {"q": "MCP"}, "arguments_error": None}
    ]
    assert body["finish_reason"] == "tool_calls"


# --------------------------------------------------------------------------- #
# request_id
# --------------------------------------------------------------------------- #


async def test_incoming_request_id_is_reused(client_factory: Any) -> None:
    client = StubClient("primary", chat_results=[text_response("ok")])
    http = await client_factory(gateway_with(client))

    response = await http.post(
        "/v1/chat", json=USER_BODY, headers={"X-Request-ID": "trace-abc_123"}
    )

    assert response.headers["X-Request-ID"] == "trace-abc_123"
    assert response.json()["request_id"] == "trace-abc_123"


async def test_invalid_request_id_is_replaced(client_factory: Any) -> None:
    """不校验就接受调用方的 id = 给日志注入开口子。"""

    client = StubClient("primary", chat_results=[text_response("ok")])
    http = await client_factory(gateway_with(client))

    response = await http.post(
        "/v1/chat", json=USER_BODY, headers={"X-Request-ID": "bad id!"}
    )

    assert response.headers["X-Request-ID"] != "bad id!"
    assert response.headers["X-Request-ID"].startswith("req_")


# --------------------------------------------------------------------------- #
# 错误映射（§3.6）
# --------------------------------------------------------------------------- #


async def test_validation_error_returns_400_with_details(client_factory: Any) -> None:
    client = StubClient("primary", chat_results=[text_response("never")])
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json={"messages": []})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "BAD_REQUEST"
    assert body["details"]


async def test_tool_choice_name_requires_required_mode(client_factory: Any) -> None:
    client = StubClient("primary", chat_results=[text_response("never")])
    http = await client_factory(gateway_with(client))

    response = await http.post(
        "/v1/chat",
        json={**USER_BODY, "tool_choice": "auto", "tool_choice_name": "search"},
    )

    assert response.status_code == 400


async def test_capability_error_is_400_not_200(client_factory: Any) -> None:
    """StreamingResponse 先发 header：能力校验必须挡在 200 之前。"""

    client = StubClient("primary", capabilities=frozenset({Capability.STREAM}))
    http = await client_factory(gateway_with(client))

    response = await http.post(
        "/v1/chat",
        json={
            **USER_BODY,
            "tools": [{"name": "search", "description": "", "parameters": {}}],
        },
    )

    assert response.status_code == 400
    assert response.json()["code"] == "CAPABILITY_UNSUPPORTED"


async def test_upstream_auth_error_is_502(client_factory: Any) -> None:
    """ADR-005：我方 401 只表示"调用方没通过我方鉴权"。"""

    client = StubClient(
        "primary", chat_results=[UpstreamAuthError("bad key", provider="primary")]
    )
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json=USER_BODY)

    assert response.status_code == 502
    assert response.json()["code"] == "UPSTREAM_AUTH"


async def test_rate_limit_passes_retry_after(client_factory: Any) -> None:
    client = StubClient(
        "primary",
        chat_results=[RateLimitError("slow", provider="primary", retry_after=2.0)],
    )
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json=USER_BODY)

    assert response.status_code == 429
    assert response.json()["code"] == "RATE_LIMITED"
    assert response.headers["Retry-After"] == "2"


async def test_unknown_model_alias_is_bad_request(client_factory: Any) -> None:
    """调用方拼错 model → 400；只有 models.yaml 里配错才是 500 CONFIG_ERROR。"""

    client = StubClient("primary", chat_results=[text_response("never")])
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json={**USER_BODY, "model": "ghost"})

    assert response.status_code == 400
    assert response.json()["code"] == "BAD_REQUEST"


# --------------------------------------------------------------------------- #
# 流式
# --------------------------------------------------------------------------- #


async def test_stream_emits_parseable_frames(client_factory: Any) -> None:
    event = StreamEvent(
        type=StreamEventType.TOOL_CALL,
        tool_call=ToolCall(id="c1", name="search", arguments={"q": "MCP"}),
    )
    client = StubClient(
        "primary",
        stream_scripts=[
            [text_delta("你好"), event, message_end(input_tokens=3, output_tokens=2)]
        ],
    )
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json={**USER_BODY, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Model-Alias"] == "primary"
    assert response.headers["X-Request-ID"]

    frames = [frame for frame in response.text.split("\n\n") if frame.strip()]
    decoded = [decode_frame(frame) for frame in frames]

    assert [name for name, _payload in decoded] == [
        "text_delta",
        "tool_call",
        "message_end",
    ]
    assert decoded[0][1]["text_delta"] == "你好"
    assert decoded[1][1]["tool_call"]["arguments"] == {"q": "MCP"}
    assert decoded[2][1]["usage"]["total_tokens"] == 5
    # 每帧都带同一个 request_id
    assert len({payload["request_id"] for _name, payload in decoded}) == 1


async def test_stream_error_becomes_error_frame(client_factory: Any) -> None:
    from agent_framework.llm.errors import UpstreamServerError

    client = StubClient(
        "primary",
        stream_scripts=[
            [text_delta("半截"), UpstreamServerError("boom", provider="primary")]
        ],
    )
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json={**USER_BODY, "stream": True})

    # 已发 header，所以这里必须是 200 + error 帧——这正是 why precheck 要前置。
    assert response.status_code == 200

    frames = [frame for frame in response.text.split("\n\n") if frame.strip()]
    names = [decode_frame(frame)[0] for frame in frames]
    assert names == ["text_delta", "error"]

    payload = json.loads(frames[-1].split("data:", 1)[1])
    assert payload["data"]["code"] == "UPSTREAM_ERROR"


async def test_stream_precheck_failure_is_400_not_200(client_factory: Any) -> None:
    client = StubClient("primary", capabilities=frozenset({Capability.TOOLS}))
    http = await client_factory(gateway_with(client))

    response = await http.post("/v1/chat", json={**USER_BODY, "stream": True})

    assert response.status_code == 400
    assert response.json()["code"] == "CAPABILITY_UNSUPPORTED"


async def test_health_still_works(client_factory: Any) -> None:
    http = await client_factory(gateway_with(StubClient("primary")))

    response = await http.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

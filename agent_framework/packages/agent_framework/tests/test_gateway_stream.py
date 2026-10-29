"""流式链路测试：committed 状态机（A10）、空流、资源释放、计量一致（A11）。

`committed` 是这一层唯一真正难写对的东西：
已经吐给用户的内容**不能重放**，所以"是否已提交"必须是运行时状态，
而不是"上次调用是否成功"这种事后判断。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
import pytest
from agent_framework.llm.base import Capability, LLMClient
from agent_framework.llm.errors import UpstreamServerError, UpstreamTimeoutError
from agent_framework.llm.providers import OpenAICompatClient
from agent_framework.llm.types import (
    Message,
    StreamEvent,
    StreamEventType,
    ToolChoice,
    ToolSpec,
)
from conftest import load_case, recorded_client
from gateway_helpers import (
    FailingByteStream,
    RecordingSleep,
    SequenceTransport,
    StubClient,
    build_test_gateway,
    json_response,
    make_catalog,
    make_settings,
    message_end,
    sse_response,
    text_delta,
    user_call,
)


def event_types(events: list[StreamEvent]) -> list[StreamEventType]:
    return [event.type for event in events]


def request_logs(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "agent_framework.request"
    ]


# --------------------------------------------------------------------------- #
# 提交边界
# --------------------------------------------------------------------------- #


async def test_retries_when_failure_happens_before_first_event() -> None:
    """首个事件之前失败 = 用户什么都没看到 → 允许重试。"""

    client = StubClient(
        "primary",
        stream_scripts=[
            [UpstreamServerError("boom", provider="primary")],
            [text_delta("你好"), message_end()],
        ],
    )
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    events = [event async for event in gateway.stream(user_call())]

    assert client.stream_calls == 2
    assert event_types(events) == [
        StreamEventType.TEXT_DELTA,
        StreamEventType.MESSAGE_END,
    ]


async def test_no_retry_after_text_committed() -> None:
    """A10 核心：已发出 text_delta 后再失败，只能以 error 帧终止，绝不重放。"""

    client = StubClient(
        "primary",
        stream_scripts=[
            [text_delta("半截回答"), UpstreamServerError("boom", provider="primary")],
            [text_delta("重放的内容")],
        ],
    )
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    events = [event async for event in gateway.stream(user_call())]

    assert client.stream_calls == 1
    assert event_types(events) == [StreamEventType.TEXT_DELTA, StreamEventType.ERROR]
    assert events[0].text_delta == "半截回答"
    assert events[-1].data["code"] == "UPSTREAM_ERROR"
    assert "重放" not in json.dumps(
        [event.model_dump(mode="json") for event in events], ensure_ascii=False
    )


async def test_error_frame_carries_request_id() -> None:
    from agent_framework.observability import request_scope

    client = StubClient(
        "primary",
        stream_scripts=[
            [text_delta("x"), UpstreamServerError("boom", provider="primary")]
        ],
    )
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    with request_scope("req_stream_err"):
        events = [event async for event in gateway.stream(user_call())]

    assert events[-1].data["request_id"] == "req_stream_err"


async def test_committed_stream_does_not_fall_back_to_next_candidate() -> None:
    """已提交就**连 fallback 也不能走**：换一家继续吐等于拼接两段不同回答。"""

    primary = StubClient(
        "primary",
        stream_scripts=[
            [text_delta("半截"), UpstreamServerError("boom", provider="primary")]
        ],
    )
    backup = StubClient("backup", stream_scripts=[[text_delta("另一家")]])
    gateway = build_test_gateway(
        make_catalog(["primary", "backup"]),
        {"primary": primary, "backup": backup},
    )

    events = [event async for event in gateway.stream(user_call())]

    assert backup.stream_calls == 0
    assert event_types(events)[-1] is StreamEventType.ERROR


# --------------------------------------------------------------------------- #
# 空流与超时
# --------------------------------------------------------------------------- #


async def test_empty_stream_is_an_error_not_a_success() -> None:
    """空流当成正常结束会返回空字符串 + finish_reason=stop（"模型偶尔不回答"）。"""

    client = StubClient("primary", stream_scripts=[[]])
    gateway = build_test_gateway(
        make_catalog(["primary"]),
        {"primary": client},
        settings=make_settings(max_attempts=1),
    )

    with pytest.raises(UpstreamServerError, match="空流"):
        [event async for event in gateway.stream(user_call())]


async def test_first_token_timeout_is_retryable() -> None:
    """首 token 超时走的是 `first_token_timeout_s`，不是整体请求超时。"""

    class SlowClient(StubClient):
        async def stream(  # type: ignore[override]
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
            *,
            tool_choice: ToolChoice | None = None,
            **opts: object,
        ) -> AsyncIterator[StreamEvent]:
            import asyncio

            self.stream_calls += 1
            await asyncio.sleep(5)

            yield text_delta("never")

    client = SlowClient("primary")
    sleep = RecordingSleep()
    gateway = build_test_gateway(
        make_catalog(["primary"]),
        {"primary": client},
        settings=make_settings(max_attempts=1, first_token_timeout_s=0.01),
        sleep=sleep,
    )

    with pytest.raises(UpstreamTimeoutError, match="首 token 超时"):
        [event async for event in gateway.stream(user_call())]


# --------------------------------------------------------------------------- #
# 资源释放
# --------------------------------------------------------------------------- #


class RecordingStream:
    """记录 `aclose()` 的假上游流。"""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = list(events)
        self.closed = False

    def __aiter__(self) -> RecordingStream:
        return self

    async def __anext__(self) -> StreamEvent:
        if not self._events:
            raise StopAsyncIteration

        return self._events.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class RecordingStreamClient(LLMClient):
    def __init__(self, stream: RecordingStream) -> None:
        self._stream = stream

    @property
    def name(self) -> str:
        return "primary"

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.STREAM, Capability.TOOLS})

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: object,
    ) -> object:
        raise AssertionError("本用例不走非流式")

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: object,
    ) -> AsyncIterator[StreamEvent]:
        return self._stream  # type: ignore[return-value]


async def test_consumer_abandon_closes_upstream_stream() -> None:
    """客户端断开时 generator 被 aclose → 必须把上游流也关掉。

    漏掉这一步功能测试一天都发现不了，W2 压测立刻 `Too many open files`。
    """

    stream = RecordingStream([text_delta("a"), text_delta("b")])
    gateway = build_test_gateway(
        make_catalog(["primary"]), {"primary": RecordingStreamClient(stream)}
    )

    agen = gateway.stream(user_call())
    first = await agen.__anext__()
    assert first.text_delta == "a"

    await agen.aclose()

    assert stream.closed is True


# --------------------------------------------------------------------------- #
# A10 / A11：真 adapter 全链路
# --------------------------------------------------------------------------- #


async def test_a10_real_adapter_mid_stream_failure_does_not_replay() -> None:
    """真 adapter + 断连：只发 1 次 HTTP，文本只出现一次，末尾是 error 帧。"""

    first_chunk = (
        b'data: {"choices": [{"index": 0, "delta": {"content": "MCP"}, '
        b'"finish_reason": null}]}\n\n'
    )
    transport = SequenceTransport(
        [
            lambda: httpx.Response(
                200,
                stream=FailingByteStream([first_chunk]),
                headers={"content-type": "text/event-stream"},
            )
        ]
    )
    client = OpenAICompatClient(
        name="primary",
        model="gpt-test",
        base_url="https://api.openai.test/v1",
        api_key="sk-test",
        transport=transport.transport(),
    )
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    events = [event async for event in gateway.stream(user_call())]

    assert transport.count == 1
    texts = [event.text_delta for event in events if event.type is StreamEventType.TEXT_DELTA]
    assert texts == ["MCP"]
    assert events[-1].type is StreamEventType.ERROR


async def test_a11_cost_is_equal_for_stream_and_non_stream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同一夹具的流式与非流式必须算出相同成本，否则成本曲线无法汇总。"""

    case = load_case("openai", "text_basic")
    settings = make_settings(max_attempts=1)
    costs: dict[str, float] = {}
    estimated: dict[str, bool] = {}

    for mode in ("chat", "stream"):
        client, _recorder = recorded_client(case)
        gateway = build_test_gateway(
            make_catalog(["primary"]), {"primary": client}, settings=settings
        )
        caplog.clear()

        with caplog.at_level(logging.INFO, logger="agent_framework.request"):
            if mode == "chat":
                await gateway.chat(user_call())
            else:
                [event async for event in gateway.stream(user_call())]

        logs = request_logs(caplog)
        assert len(logs) == 1, f"{mode} 应只产出一条 RequestLog"

        costs[mode] = float(logs[0]["cost"])  # type: ignore[arg-type]
        estimated[mode] = bool(logs[0]["usage_estimated"])

    assert costs["chat"] == costs["stream"]
    assert estimated["chat"] == estimated["stream"] is False


async def test_a11_missing_usage_marks_estimated_in_both_modes() -> None:
    """厂商不返回 usage 时，两种模式都要标 estimated=True（否则成本看起来是 0）。

    `usage_missing` 夹具没有 stream.txt（线上这种响应只会出现在非流式），
    所以流式那一路用一段"只有内容、没有 usage 片"的内联 SSE——
    它恰好是"DeepSeek 之外的三家忘了开 include_usage"的真实形状。
    """

    case = load_case("openai", "usage_missing")
    settings = make_settings(max_attempts=1)
    stream_text = (
        'data: {"choices": [{"index": 0, "delta": {"content": "你好"},'
        ' "finish_reason": null}]}\n\n'
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    flags: dict[str, bool] = {}

    chat_client, _recorder = recorded_client(case)
    chat_gateway = build_test_gateway(
        make_catalog(["primary"]), {"primary": chat_client}, settings=settings
    )
    chat_outcome = await chat_gateway.chat(user_call())
    flags["chat"] = chat_outcome.response.usage.estimated

    transport = SequenceTransport([lambda: sse_response(stream_text)])
    stream_client = OpenAICompatClient(
        name="primary",
        model="gpt-test",
        base_url="https://api.openai.test/v1",
        api_key="sk-test",
        transport=transport.transport(),
    )
    stream_gateway = build_test_gateway(
        make_catalog(["primary"]), {"primary": stream_client}, settings=settings
    )
    events = [event async for event in stream_gateway.stream(user_call())]
    usage = next(event.usage for event in events if event.usage is not None)
    flags["stream"] = usage.estimated

    assert flags["chat"] is True
    assert flags["stream"] is True


async def test_stream_log_records_first_token_ms(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = StubClient("primary", stream_scripts=[[text_delta("hi"), message_end()]])
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    with caplog.at_level(logging.INFO, logger="agent_framework.request"):
        [event async for event in gateway.stream(user_call())]

    logs = request_logs(caplog)
    assert len(logs) == 1
    assert logs[0]["first_token_ms"] is not None
    assert logs[0]["stream"] is True


async def test_reasoning_only_stream_has_no_first_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """思考型模型的 reasoning 不算首 token，否则 p95 会虚高到 5s+。"""

    reasoning = StreamEvent(
        type=StreamEventType.REASONING_DELTA, reasoning_delta="想一下"
    )
    client = StubClient("primary", stream_scripts=[[reasoning, message_end()]])
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    with caplog.at_level(logging.INFO, logger="agent_framework.request"):
        [event async for event in gateway.stream(user_call())]

    logs = request_logs(caplog)
    assert logs[0]["first_token_ms"] is None


async def test_gateway_aclose_closes_all_clients() -> None:
    client = StubClient("primary")
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    await gateway.aclose()

    assert client.closed == 1


def test_sse_frame_headers_disable_buffering() -> None:
    from agent_framework.server.sse import SSE_HEADERS

    assert SSE_HEADERS["X-Accel-Buffering"] == "no"
    assert SSE_HEADERS["Cache-Control"] == "no-cache"


def test_sequence_transport_counts_requests() -> None:
    """A9/A10 的"发了几次请求"必须由 transport 计数，不能靠 adapter 自报。"""

    transport = SequenceTransport([lambda: json_response({})])

    assert transport.count == 0

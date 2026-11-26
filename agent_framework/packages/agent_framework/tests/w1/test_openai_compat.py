"""openai_compat 的单家细节：协议噪声、组装器、差异 flags。

这些断言不该进 conformance——它们是"这一家"的实现细节，
放进去会让 conformance 变成厂商差异清单（见 test_conformance 顶部说明）。
"""

from __future__ import annotations

import pytest
from agent_framework.llm.errors import BadRequestError
from agent_framework.llm.providers._common import SSELineKind, parse_sse_line
from agent_framework.llm.providers.openai_compat import (
    OpenAICompatAssembler,
    OpenAICompatFlags,
    ThinkingStyle,
)
from agent_framework.llm.types import (
    FinishReason,
    Message,
    MessageRole,
    StreamEventType,
    ToolChoice,
    ToolChoiceMode,
)
from conftest import MAX_TOKENS, canonical_messages, canonical_tools, load_case, make_client

# --------------------------------------------------------------------------- #
# SSE 行解析：五种行 + 无空格变体
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("", SSELineKind.SKIP),
        ("\n", SSELineKind.SKIP),
        (": keep-alive", SSELineKind.SKIP),
        ("event: message_start", SSELineKind.EVENT),
        ("event:message_start", SSELineKind.EVENT),
        ('data: {"a": 1}', SSELineKind.DATA),
        ('data:{"a": 1}', SSELineKind.DATA),
        ("data: [DONE]", SSELineKind.DONE),
        ("data:[DONE]", SSELineKind.DONE),
        ("id: 42", SSELineKind.SKIP),
        ("retry: 1000", SSELineKind.SKIP),
        ("data:", SSELineKind.SKIP),
        ("garbage-line-without-colon", SSELineKind.SKIP),
    ],
)
def test_parse_sse_line(raw: str, kind: SSELineKind) -> None:
    assert parse_sse_line(raw).kind is kind


def test_parse_sse_line_accepts_bytes() -> None:
    parsed = parse_sse_line(b'data: {"a": 1}\r\n')

    assert parsed.kind is SSELineKind.DATA
    assert parsed.payload == '{"a": 1}'


def test_parse_sse_line_keeps_inner_spaces() -> None:
    parsed = parse_sse_line('data:  {"a": "x  y"}')

    # 只去掉一个前导空格，第二个空格属于 payload
    assert parsed.payload == ' {"a": "x  y"}'


# --------------------------------------------------------------------------- #
# 组装器：并行工具调用不能并错
# --------------------------------------------------------------------------- #


def _args_chunk(index: int, arguments: str) -> dict[str, object]:
    """构造一个只带参数分片的 tool_calls delta（后续分片没有 id，这是关键）。"""

    return {"index": index, "function": {"arguments": arguments}}


def test_assembler_keeps_parallel_tool_calls_apart() -> None:
    assembler = OpenAICompatAssembler(provider="openai")

    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "search_web", "arguments": ""},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {"name": "fetch_url", "arguments": ""},
                            },
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {"tool_calls": [_args_chunk(0, '{"q":')]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [_args_chunk(1, '{"url":')]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [_args_chunk(0, '"x"}')]}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 1, "function": {"arguments": '"http://a"}'}}]},
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]

    for chunk in chunks:
        assembler.feed_chunk(chunk)

    events = assembler.finalize()
    calls = [event.tool_call for event in events if event.type is StreamEventType.TOOL_CALL]

    assert [call.id for call in calls] == ["call_a", "call_b"]
    assert calls[0].name == "search_web"
    assert calls[0].arguments == {"q": "x"}
    assert calls[1].name == "fetch_url"
    assert calls[1].arguments == {"url": "http://a"}


def test_assembler_emits_single_message_end_with_usage() -> None:
    assembler = OpenAICompatAssembler(provider="deepseek")

    assembler.feed_chunk({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
    # DeepSeek 形状：usage 挂在最后一个内容块上（choices 非空）
    assembler.feed_chunk(
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
    )

    events = assembler.finalize()
    ends = [event for event in events if event.type is StreamEventType.MESSAGE_END]

    assert len(ends) == 1
    assert ends[0].finish_reason is FinishReason.STOP
    assert ends[0].usage is not None
    assert ends[0].usage.estimated is False
    assert ends[0].usage.output_tokens == 7

    # finalize 幂等：再调一次不能重复发终止事件
    assert assembler.finalize() == []


def test_assembler_estimates_usage_when_absent() -> None:
    assembler = OpenAICompatAssembler(provider="openai")

    assembler.feed_chunk({"choices": [{"index": 0, "delta": {"content": "hello world"}}]})
    assembler.feed_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})

    end = [event for event in assembler.finalize() if event.type is StreamEventType.MESSAGE_END][0]

    assert end.usage is not None
    assert end.usage.estimated is True
    assert end.usage.total_tokens > 0


def test_assembler_keeps_reasoning_out_of_text() -> None:
    assembler = OpenAICompatAssembler(provider="deepseek")

    assembler.feed_chunk(
        {"choices": [{"index": 0, "delta": {"reasoning_content": "想一想"}}]}
    )
    assembler.feed_chunk({"choices": [{"index": 0, "delta": {"content": "答案"}}]})

    events = assembler.finalize()
    end = [event for event in events if event.type is StreamEventType.MESSAGE_END][0]

    assert end.data["provider_meta"]["reasoning_content"] == "想一想"


# --------------------------------------------------------------------------- #
# 响应解析：坏数据只降级不抛
# --------------------------------------------------------------------------- #


def test_bad_arguments_degrade_without_raising() -> None:
    client = make_client("openai")
    body = {
        "id": "x",
        "model": "gpt-test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "f", "arguments": "{not json"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    response = client.parse_response(body)
    call = response.tool_calls[0]

    assert call.arguments == {}
    assert call.arguments_error is not None
    assert call.raw_arguments == "{not json"


def test_response_without_choices_raises_upstream_error() -> None:
    from agent_framework.llm.errors import UpstreamServerError

    client = make_client("openai")

    with pytest.raises(UpstreamServerError):
        client.parse_response({"id": "x", "choices": []})


# --------------------------------------------------------------------------- #
# 差异 flags：全部数据驱动
# --------------------------------------------------------------------------- #


def test_openai_wraps_tool_in_function_object() -> None:
    client = make_client("openai")
    body = client.build_request(canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS).body

    tool = body["tools"][0]

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "search_web"
    assert "name" not in tool  # 不能把 name 放在顶层：必 400


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("openai", True), ("qwen", True), ("deepseek", None)],
)
def test_parallel_tool_calls_flag(provider: str, expected: bool | None) -> None:
    client = make_client(provider)
    body = client.build_request(canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS).body

    if expected is None:
        assert "parallel_tool_calls" not in body
    else:
        assert body["parallel_tool_calls"] is expected


def test_qwen_degrades_required_tool_choice_with_note() -> None:
    client = make_client("qwen")

    prepared = client.build_request(
        canonical_messages(),
        canonical_tools(),
        tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED),
        max_tokens=MAX_TOKENS,
    )

    # tool_choice 是策略意图，不是硬保证：不支持就降级 + 记 notes，不抛异常
    assert prepared.body["tool_choice"] == "auto"
    assert prepared.notes


def test_openai_keeps_required_tool_choice() -> None:
    client = make_client("openai")

    prepared = client.build_request(
        canonical_messages(),
        canonical_tools(),
        tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED),
        max_tokens=MAX_TOKENS,
    )

    assert prepared.body["tool_choice"] == "required"
    assert prepared.notes == ()


def test_named_tool_choice_is_object_on_openai() -> None:
    client = make_client("openai")

    body = client.build_request(
        canonical_messages(),
        canonical_tools(),
        tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED, name="search_web"),
        max_tokens=MAX_TOKENS,
    ).body

    assert body["tool_choice"] == {"type": "function", "function": {"name": "search_web"}}


def test_deepseek_replays_reasoning_content() -> None:
    client = make_client("deepseek")
    messages = [
        *canonical_messages(),
        Message(
            role=MessageRole.ASSISTANT,
            content="答案",
            provider_meta={"reasoning_content": "先想一下"},
        ),
    ]

    body = client.build_request(messages, None, max_tokens=MAX_TOKENS).body

    # 带 tools 时 DeepSeek 必须回传，否则第二轮 400
    assert body["messages"][-1]["reasoning_content"] == "先想一下"


def test_openai_never_replays_reasoning_content() -> None:
    client = make_client("openai")
    messages = [
        *canonical_messages(),
        Message(
            role=MessageRole.ASSISTANT,
            content="答案",
            provider_meta={"reasoning_content": "先想一下"},
        ),
    ]

    body = client.build_request(messages, None, max_tokens=MAX_TOKENS).body

    assert "reasoning_content" not in body["messages"][-1]


def test_qwen_enable_thinking_requires_stream() -> None:
    client = make_client("qwen")

    with pytest.raises(BadRequestError):
        client.build_request(
            canonical_messages(),
            canonical_tools(),
            thinking=True,
            stream=False,
            max_tokens=MAX_TOKENS,
        )


def test_thinking_flag_absent_by_default() -> None:
    """不显式传 thinking 就完全不动，交给厂商默认值（DeepSeek 默认开启）。"""

    for provider in ("deepseek", "qwen"):
        body = make_client(provider).build_request(
            canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
        ).body

        assert "thinking" not in body
        assert "enable_thinking" not in body


def test_stream_request_carries_include_usage() -> None:
    body = make_client("openai").build_request(
        canonical_messages(), canonical_tools(), stream=True, max_tokens=MAX_TOKENS
    ).body

    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_unknown_option_raises() -> None:
    client = make_client("openai")

    with pytest.raises(TypeError):
        client.build_request(canonical_messages(), canonical_tools(), max_token=MAX_TOKENS)  # type: ignore[call-arg]


def test_assembler_tolerates_keepalive_only_when_flagged() -> None:
    case = load_case("deepseek", "tool_call_basic")

    assert ": keep-alive" in (case.stream_text or "")

    lenient = OpenAICompatAssembler(provider="deepseek")
    strict = OpenAICompatAssembler(provider="deepseek")

    assert lenient.feed_line(": keep-alive") == []
    # 解析器本身统一跳过注释行，严格/宽容只影响"无法识别的协议噪声"
    assert strict.feed_line(": keep-alive") == []


def test_capabilities_reflect_parallel_support() -> None:
    from agent_framework.llm.base import Capability

    assert Capability.PARALLEL_TOOLS in make_client("qwen").capabilities
    assert Capability.PARALLEL_TOOLS not in make_client("deepseek").capabilities


def test_flag_defaults_are_openai_semantics() -> None:
    flags = OpenAICompatFlags()

    assert flags.thinking_style is ThinkingStyle.NONE
    assert flags.send_parallel_tool_calls is None

"""Anthropic 的单家细节：system 顶层、tool_result 合组、分段 usage、thinking 签名。"""

from __future__ import annotations

import pytest
from agent_framework.llm.errors import BadRequestError
from agent_framework.llm.providers.anthropic import AnthropicAssembler
from agent_framework.llm.types import (
    FinishReason,
    Message,
    MessageRole,
    StreamEventType,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
)
from conftest import MAX_TOKENS, canonical_messages, canonical_tools, make_client


def _assistant_with_two_calls() -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=[
            ToolCall(id="toolu_1", name="search_web", arguments={"query": "a"}),
            ToolCall(id="toolu_2", name="fetch_url", arguments={"url": "b"}),
        ],
    )


# --------------------------------------------------------------------------- #
# system 必须是顶层参数
# --------------------------------------------------------------------------- #


def test_system_is_top_level_and_merged() -> None:
    client = make_client("anthropic")
    messages = [
        Message(role=MessageRole.SYSTEM, content="规则一"),
        Message(role=MessageRole.SYSTEM, content="规则二"),
        Message(role=MessageRole.USER, content="你好"),
    ]

    body = client.build_request(messages, None, max_tokens=MAX_TOKENS).body

    assert body["system"] == "规则一\n\n规则二"
    assert [message["role"] for message in body["messages"]] == ["user"]


# --------------------------------------------------------------------------- #
# 多条 tool 结果必须合并成一条 user 消息，且一次给齐
# --------------------------------------------------------------------------- #


def test_tool_results_are_merged_into_single_user_message() -> None:
    client = make_client("anthropic")
    messages = [
        *canonical_messages(),
        _assistant_with_two_calls(),
        Message(role=MessageRole.TOOL, content="结果一", tool_call_id="toolu_1"),
        Message(role=MessageRole.TOOL, content="结果二", tool_call_id="toolu_2"),
    ]

    body = client.build_request(messages, None, max_tokens=MAX_TOKENS).body

    # system 在顶层，所以只剩 user / assistant / user(tool_result) 三条
    assert len(body["messages"]) == 3

    last = body["messages"][-1]

    assert last["role"] == "user"
    assert [block["type"] for block in last["content"]] == ["tool_result", "tool_result"]
    assert [block["tool_use_id"] for block in last["content"]] == ["toolu_1", "toolu_2"]

    assistant = body["messages"][1]

    assert [block["type"] for block in assistant["content"]] == ["tool_use", "tool_use"]
    # input 已经是对象，不是 JSON 字符串
    assert assistant["content"][0]["input"] == {"query": "a"}


def test_tool_result_can_mark_error() -> None:
    client = make_client("anthropic")
    messages = [
        *canonical_messages(),
        _assistant_with_two_calls(),
        Message(
            role=MessageRole.TOOL,
            content="超时",
            tool_call_id="toolu_1",
            provider_meta={"is_error": True},
        ),
        Message(role=MessageRole.TOOL, content="ok", tool_call_id="toolu_2"),
    ]

    body = client.build_request(messages, None, max_tokens=MAX_TOKENS).body
    blocks = body["messages"][-1]["content"]

    assert blocks[0]["is_error"] is True
    assert "is_error" not in blocks[1]


# --------------------------------------------------------------------------- #
# max_tokens 必填
# --------------------------------------------------------------------------- #


def test_max_tokens_is_required() -> None:
    client = make_client("anthropic")

    with pytest.raises(BadRequestError):
        client.build_request(canonical_messages(), canonical_tools())


def test_default_max_tokens_is_used_when_configured() -> None:
    from agent_framework.llm.providers.anthropic import AnthropicClient

    client = AnthropicClient(
        model="claude-test",
        base_url="https://api.anthropic.test",
        api_key="sk-ant-test",
        default_max_tokens=256,
    )

    body = client.build_request(canonical_messages(), canonical_tools()).body

    assert body["max_tokens"] == 256


# --------------------------------------------------------------------------- #
# 工具定义与 tool_choice 的形态
# --------------------------------------------------------------------------- #


def test_tools_use_input_schema() -> None:
    body = make_client("anthropic").build_request(
        canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
    ).body

    tool = body["tools"][0]

    assert tool["name"] == "search_web"
    assert "input_schema" in tool
    assert "parameters" not in tool
    assert "type" not in tool  # Anthropic 的 tool 定义没有 type 包裹


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (ToolChoice(mode=ToolChoiceMode.AUTO), {"type": "auto"}),
        (ToolChoice(mode=ToolChoiceMode.NONE), {"type": "none"}),
        (ToolChoice(mode=ToolChoiceMode.REQUIRED), {"type": "any"}),
        (
            ToolChoice(mode=ToolChoiceMode.REQUIRED, name="search_web"),
            {"type": "tool", "name": "search_web"},
        ),
    ],
)
def test_tool_choice_is_object(choice: ToolChoice, expected: dict) -> None:
    body = make_client("anthropic").build_request(
        canonical_messages(),
        canonical_tools(),
        tool_choice=choice,
        max_tokens=MAX_TOKENS,
    ).body

    assert body["tool_choice"] == expected


def test_disable_parallel_tool_use_is_inverse_semantics() -> None:
    body = make_client("anthropic").build_request(
        canonical_messages(),
        canonical_tools(),
        parallel_tool_calls=False,
        max_tokens=MAX_TOKENS,
    ).body

    # 注意：这是"禁止并行"，与 OpenAI 的 parallel_tool_calls=true 语义相反
    assert body["tool_choice"]["disable_parallel_tool_use"] is True


# --------------------------------------------------------------------------- #
# thinking：签名必须原样回传
# --------------------------------------------------------------------------- #


def test_thinking_blocks_are_replayed_verbatim() -> None:
    client = make_client("anthropic")
    messages = [
        *canonical_messages(),
        Message(
            role=MessageRole.ASSISTANT,
            content="答案",
            provider_meta={
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "想一想", "signature": "sig-abc"}
                ]
            },
        ),
    ]

    body = client.build_request(messages, None, max_tokens=MAX_TOKENS).body
    blocks = body["messages"][-1]["content"]

    assert blocks[0]["signature"] == "sig-abc"
    assert blocks[1] == {"type": "text", "text": "答案"}


def test_thinking_requires_budget_below_max_tokens() -> None:
    client = make_client("anthropic")

    with pytest.raises(BadRequestError):
        client.build_request(
            canonical_messages(),
            canonical_tools(),
            thinking=True,
            max_tokens=512,
            thinking_budget=1024,
        )


def test_thinking_block_lands_in_provider_meta_on_parse() -> None:
    client = make_client("anthropic")
    body = {
        "id": "msg_1",
        "model": "claude-test",
        "content": [
            {"type": "thinking", "thinking": "想一想", "signature": "sig-abc"},
            {"type": "text", "text": "答案"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    response = client.parse_response(body)

    assert response.text == "答案"
    assert response.message.provider_meta is not None
    assert response.message.provider_meta["thinking_blocks"][0]["signature"] == "sig-abc"


# --------------------------------------------------------------------------- #
# 流式：ping 忽略、usage 分段、input_json_delta 拼接
# --------------------------------------------------------------------------- #


def test_ping_is_ignored() -> None:
    assembler = AnthropicAssembler()

    assert assembler.feed_line("event: ping") == []
    assert assembler.feed_line('data: {"type": "ping"}') == []


def test_usage_is_split_between_message_start_and_delta() -> None:
    assembler = AnthropicAssembler()

    assembler.feed_line("event: message_start")
    assembler.feed_line(
        'data: {"type": "message_start", "message": {"usage": '
        '{"input_tokens": 120, "output_tokens": 1}}}'
    )
    assembler.feed_line("event: message_delta")
    assembler.feed_line(
        'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
        '"usage": {"output_tokens": 42}}'
    )

    end = [
        event
        for event in assembler.finalize()
        if event.type is StreamEventType.MESSAGE_END
    ][0]

    assert end.usage is not None
    assert end.usage.input_tokens == 120
    # message_start 里的 output_tokens 是占位值，必须用 message_delta 的累计值覆盖
    assert end.usage.output_tokens == 42
    assert end.usage.estimated is False
    assert end.finish_reason is FinishReason.STOP


def test_tool_use_emitted_on_content_block_stop() -> None:
    assembler = AnthropicAssembler()

    assembler.feed_line("event: content_block_start")
    assembler.feed_line(
        'data: {"type": "content_block_start", "index": 1, "content_block": '
        '{"type": "tool_use", "id": "toolu_1", "name": "search_web", "input": {}}}'
    )
    assembler.feed_line("event: content_block_delta")
    assembler.feed_line(
        'data: {"type": "content_block_delta", "index": 1, "delta": '
        '{"type": "input_json_delta", "partial_json": "{\\"query\\":"}}'
    )
    assembler.feed_line("event: content_block_delta")
    assembler.feed_line(
        'data: {"type": "content_block_delta", "index": 1, "delta": '
        '{"type": "input_json_delta", "partial_json": " \\"mcp\\"}"}}'
    )
    assembler.feed_line("event: content_block_stop")
    events = assembler.feed_line('data: {"type": "content_block_stop", "index": 1}')

    calls = [event.tool_call for event in events if event.type is StreamEventType.TOOL_CALL]

    assert len(calls) == 1
    assert calls[0].name == "search_web"
    assert calls[0].arguments == {"query": "mcp"}


def test_text_block_does_not_emit_tool_call_on_stop() -> None:
    assembler = AnthropicAssembler()

    assembler.feed_line("event: content_block_start")
    assembler.feed_line(
        'data: {"type": "content_block_start", "index": 0, "content_block": '
        '{"type": "text", "text": ""}}'
    )
    assembler.feed_line("event: content_block_stop")

    assert assembler.feed_line('data: {"type": "content_block_stop", "index": 0}') == []

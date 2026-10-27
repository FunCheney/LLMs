import pytest
from agent_framework.llm.types import (
    Conversation,
    FinishReason,
    LLMResponse,
    Message,
    MessageRole,
    ToolCall,
    normalize_finish_reason,
)
from pydantic import ValidationError


def test_tool_message_requires_tool_call_id():
    with pytest.raises(ValidationError):
        Message(
            role=MessageRole.TOOL,
            content="天气晴朗",
        )


def test_assistant_can_have_tool_calls_without_content():
    message = Message(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=[
            ToolCall(
                id="call_123",
                name="get_weather",
                arguments={"city": "Singapore"},
            )
        ],
    )

    assert message.content is None
    assert len(message.tool_calls) == 1


def test_system_message_must_be_first():
    with pytest.raises(ValidationError):
        Conversation(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content="你好",
                ),
                Message(
                    role=MessageRole.SYSTEM,
                    content="系统提示词",
                ),
            ]
        )


def test_tool_arguments_json_is_parsed():
    tool_call = ToolCall(
        id="call_123",
        name="get_weather",
        arguments='{"city": "Singapore"}',
    )

    assert tool_call.arguments == {
        "city": "Singapore",
    }
    assert tool_call.arguments_error is None


def test_invalid_tool_arguments_are_recorded():
    tool_call = ToolCall(
        id="call_123",
        name="get_weather",
        arguments='{"city": ',
    )

    assert tool_call.arguments == {}
    assert tool_call.arguments_error is not None


def test_unknown_finish_reason_keeps_raw_value() -> None:
    raw = "some_vendor_specific_reason"

    response = LLMResponse(
        message=Message(
            role=MessageRole.ASSISTANT,
            content=None,
        ),
        finish_reason=normalize_finish_reason(raw),
        raw_finish_reason=raw,
    )

    assert response.finish_reason == FinishReason.UNKNOWN
    assert response.raw_finish_reason == raw
    assert response.finish_reason != FinishReason.STOP


def test_deepseek_resource_exhaustion_is_not_success() -> None:
    """半截回答绝不能被当成 stop：A5 要求落到 upstream_error。"""

    reason = normalize_finish_reason("insufficient_system_resource")

    assert reason == FinishReason.UPSTREAM_ERROR
    assert reason != FinishReason.STOP
import pytest
from pydantic import ValidationError


from agent_framework.llm.types import (
    Conversation,
    FinishReason,
    LLMResponse,
    Message,
    MessageRole,
    ToolCall,
    normalize_finish_reason,
)


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


def test_unknown_finish_reason_is_preserved():
    raw = "insufficient_system_resource"

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
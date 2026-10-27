import json
from dataclasses import dataclass
from typing import Any

from ..types import Message, ToolCall, ToolSpec, LLMResponse, StreamEvent, Usage, FinishReason, StreamEventType


@dataclass
class capabilities:
    send_parallel_tool_calls: bool | None = None
    supports_tool_choice_required: bool | None = None
    thinking_style: str | None = None
    stream_keepalive_comment: bool = False
    preserve_reasoning_content: bool = False



def message_to_write(message: Message) -> dict:
    """将内部 Message 转换成 API 消息。"""
    result: dict[str, Any] = {"role": message.role}

    if message.content is not None:
        result["content"] = message.content
    if message.name:
        result["name"] = message.name
    if message.tool_call_id:
        result["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return result



def tool_spec_to_dict(tool: ToolSpec) -> dict:
    """将内部 ToolSpec 转换成 API 工具定义。"""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters
    }



def response_from_wire(data: dict) -> LLMResponse:
    """将 API 响应转换成内部响应。"""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in ("text", "output_text")
            ]
            content = "".join(text_parts) or None
        else:
            content = None

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(message.get("tool_calls") or []):
        function = raw_call.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON arguments in tool call at index {index}") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool call arguments at index {index} must decode to an object")
        calls.append(ToolCall(
            id=raw_call.get("id") or "",
            name=function.get("name") or "",
            arguments=arguments,
            index=index,
        ))

    return LLMResponse(
        id=data.get("id") or "",
        model=data.get("model") or "",
        message=Message(
            role=message.get("role") or "",
            content=content,
            name=message.get("name") or "",
            tool_call_id=message.get("tool_call_id") or "",
            tool_calls=calls,
            provider_meta=None
        ),
        finish_reason=choice.get("finish_reason") or "",
        raw_finish_reason=choice.get("finish_reason"),
        usage=choice.get("usage") or "",
    )


def chunk_to_events(chunk: dict) -> list[StreamEvent]:
    """将 OpenAI 兼容 API 的流式 chunk 转换为内部事件。"""
    result: list[StreamEvent] = []

    # 1. 处理 usage
    cur_usage = chunk.get("usage")

    if cur_usage:
        usage = Usage(
            input_tokens=cur_usage.get("prompt_tokens") or 0,
            output_tokens=cur_usage.get("completion_tokens") or 0,
            total_tokens=cur_usage.get("total_tokens") or 0,
        )

        result.append(
            StreamEvent(
                type=StreamEventType.MESSAGE_END,
                usage=usage,
            )
        )

    # 2. 处理 choices
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}

        # 3. 处理文本增量
        content = delta.get("content")

        if isinstance(content, str) and content:
            result.append(
                StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=content,
                )
            )

        # 4. 处理推理内容增量
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
        )

        if isinstance(reasoning, str) and reasoning:
            result.append(
                StreamEvent(
                    type=StreamEventType.REASONING_DELTA,
                    reasoning_delta=reasoning,
                )
            )

        # 5. 处理结束原因
        raw_finish_reason = choice.get("finish_reason")

        if raw_finish_reason is not None:
            finish_reason = map_finish_reason(raw_finish_reason)

            result.append(
                StreamEvent(
                    type=StreamEventType.MESSAGE_END,
                    finish_reason=finish_reason,
                    raw_finish_reason=raw_finish_reason,
                )
            )

    return result


def map_finish_reason(raw: str) -> FinishReason:
    """将供应商的结束原因转换为内部统一枚举。"""
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }

    return mapping.get(raw, FinishReason.UNKNOWN)



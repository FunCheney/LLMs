"""SSE 编解码（A12）：帧必须可逆，且不许把内部字段漏出去。

为什么值得单测：W3 批量跑评测时要在进程内消费事件流，靠 curl 眼验不了
"编码后解析回来还是同一个事件"。这一层一旦有损，评测数据会静默失真。
"""

from __future__ import annotations

import json

import pytest
from agent_framework.llm.types import (
    FinishReason,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)
from agent_framework.server.sse import decode_frame, encode_event


def test_text_delta_roundtrip() -> None:
    event = StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta="你好\n世界")

    name, payload = decode_frame(encode_event(event, request_id="req_1"))

    assert name == "text_delta"
    assert payload["type"] == "text_delta"
    assert payload["text_delta"] == "你好\n世界"
    assert payload["request_id"] == "req_1"


def test_tool_call_roundtrip() -> None:
    event = StreamEvent(
        type=StreamEventType.TOOL_CALL,
        tool_call=ToolCall(id="call_1", name="search", arguments={"query": "MCP"}),
    )

    name, payload = decode_frame(encode_event(event, request_id="req_2"))

    assert name == "tool_call"
    assert payload["tool_call"]["id"] == "call_1"
    assert payload["tool_call"]["arguments"] == {"query": "MCP"}


def test_message_end_keeps_usage_and_finish_reason() -> None:
    event = StreamEvent(
        type=StreamEventType.MESSAGE_END,
        finish_reason=FinishReason.STOP,
        raw_finish_reason="end_turn",
        usage=Usage(input_tokens=12, output_tokens=7),
    )

    name, payload = decode_frame(encode_event(event, request_id="req_3"))

    assert name == "message_end"
    assert payload["finish_reason"] == "stop"
    assert payload["raw_finish_reason"] == "end_turn"
    assert payload["usage"]["input_tokens"] == 12
    assert payload["usage"]["total_tokens"] == 19


def test_provider_meta_is_never_sent() -> None:
    """`reasoning_content` / `thinking.signature` 是要还给上游的，不是给调用方看的。"""

    event = StreamEvent(
        type=StreamEventType.MESSAGE_END,
        finish_reason=FinishReason.STOP,
        data={"provider_meta": {"reasoning_content": "内部思考", "signature": "sig"}},
    )

    frame = encode_event(event, request_id="req_4")

    assert "reasoning_content" not in frame
    assert "signature" not in frame
    assert "provider_meta" not in frame

    _name, payload = decode_frame(frame)
    assert "data" not in payload


def test_error_event_carries_code() -> None:
    event = StreamEvent(
        type=StreamEventType.ERROR,
        data={"code": "UPSTREAM_ERROR", "message": "boom", "request_id": "req_5"},
    )

    name, payload = decode_frame(encode_event(event, request_id="req_5"))

    assert name == "error"
    assert payload["data"]["code"] == "UPSTREAM_ERROR"


def test_frame_is_single_data_line_with_trailing_blank_line() -> None:
    event = StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta="a\nb")

    frame = encode_event(event, request_id="req_6")
    lines = frame.splitlines()

    # 第一行 event:，第二行 data:（且只有一行），末尾是空行 → 帧分隔 \n\n
    assert lines[0] == "event: text_delta"
    assert len([line for line in lines if line.startswith("data:")]) == 1
    assert frame.endswith("\n\n")

    body = lines[1][len("data:") :].strip()
    assert json.loads(body)["text_delta"] == "a\nb"


def test_missing_request_id_falls_back_to_dash() -> None:
    event = StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta="x")

    _name, payload = decode_frame(encode_event(event, request_id=""))

    assert payload["request_id"] == "-"


def test_decode_rejects_non_frame_input() -> None:
    with pytest.raises(ValueError, match="不是合法的 SSE 帧"):
        decode_frame("hello world")


def test_decode_rejects_non_object_data() -> None:
    with pytest.raises(ValueError, match="JSON 对象"):
        decode_frame('event: text_delta\ndata: [1, 2, 3]\n\n')

"""StreamEvent → SSE 帧编码。

帧格式逐字节确定，四条规矩：

1. 事件名 = `StreamEvent.type`（text_delta / tool_call / message_end / error ...）
2. `data` 是**单行** JSON——`json.dumps` 默认不换行；绝不用 f-string 拼原始文本，
   文本里的 `\\n` 会直接断帧。
3. 每帧带 `request_id`。
4. 帧间空行必须是 `\\n\\n`，少一个 `\\n` 浏览器会把两帧合并。

**不发送 `data.provider_meta`**：里面是 `reasoning_content` / `thinking.signature`
这类"要原样还给上游"的数据，属于内部回传信息，不是给调用方看的。
"""

from __future__ import annotations

import json
from typing import Any

from ..llm.types import StreamEvent

SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 以后上 nginx 不加这个，流式会被整段缓冲，前端表现为"等 30 秒后一次性出现"。
    "X-Accel-Buffering": "no",
}


def encode_event(event: StreamEvent, *, request_id: str) -> str:
    payload = event.model_dump(mode="json", exclude_none=True, exclude={"type"})

    data = payload.get("data")

    if isinstance(data, dict):
        data.pop("provider_meta", None)

        if data:
            payload["data"] = data
        else:
            payload.pop("data", None)

    body = {"type": event.type.value, "request_id": request_id or "-", **payload}

    return (
        f"event: {event.type.value}\n"
        f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
    )


def decode_frame(frame: str) -> tuple[str, dict[str, Any]]:
    """`encode_event` 的逆操作。

    给测试与 W3 的批量评测用：能把自己发出去的帧读回来，
    才算证明"编解码无损"（A12）。注意它只处理我们自己的单帧格式。
    """

    name = ""
    chunks: list[str] = []

    for line in frame.splitlines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            chunks.append(line[len("data:") :].strip())

    if not name or not chunks:
        raise ValueError(f"不是合法的 SSE 帧：{frame[:80]!r}")

    payload = json.loads("\n".join(chunks))

    if not isinstance(payload, dict):
        raise ValueError("SSE data 必须是 JSON 对象")

    return name, payload

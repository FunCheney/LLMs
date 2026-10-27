"""provider 无关的纯函数与轻量数据结构。

准入标准（docs/w1/w1-day3.md §2）：**三家以上 provider 都要用，且不含任何厂商专属字段名**。
只要出现 reasoning_content / input_schema / enable_thinking 这类词，就说明它属于某个 adapter。

这里唯一"有状态"的东西是 StreamAccumulator，它只做聚合不做解析：
两家的分片标识和累加字段完全不同（OpenAI 按 delta index，Anthropic 按 content_block），
所以拼装状态留在各自 adapter 的 Assembler 里。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..errors import UpstreamServerError
from ..types import ContentBlock, StreamEvent, ToolCall

# --------------------------------------------------------------------------- #
# 请求
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """adapter 翻译请求的结果。

    notes 记录"按能力声明做了哪些降级"，例如 tool_choice=required 被降为 auto。
    adapter 不打日志（ADR-003），由 client 层消费 notes 后统一记录。
    """

    body: dict[str, Any]
    notes: tuple[str, ...] = ()


def content_to_text(content: str | list[ContentBlock] | None) -> str:
    """把内部 content 压平成文本。"""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    return "".join(block.data for block in content if isinstance(block.data, str))


def prompt_text(messages: list[Any]) -> str:
    """把整轮消息压平成一段文本，只用于 usage 缺失时估算 input_tokens。"""

    return "\n".join(content_to_text(getattr(message, "content", None)) for message in messages)


def filter_opts(opts: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
    """只保留 adapter 认识的参数，其余直接报错。

    静默忽略未知参数会让"参数名拼错"这类 bug 一直藏着（比如 max_token）。
    """

    unknown = set(opts) - set(allowed)

    if unknown:
        raise TypeError(f"unsupported options: {sorted(unknown)}")

    return {key: opts[key] for key in allowed if key in opts}


# --------------------------------------------------------------------------- #
# 工具调用
# --------------------------------------------------------------------------- #


def make_tool_call(
    *,
    call_id: str | None,
    name: str,
    raw_arguments: Any,
    index: int | None = None,
) -> ToolCall:
    """构造统一的 ToolCall。

    - id 缺失时本地生成（architecture §3.3），保证能稳定贯穿到 tool 消息
    - raw_arguments 原样交给 ToolCall 的校验器解析：解析失败降级为 {} + arguments_error，
      **绝不在这里抛异常打断整轮**（ADR-001 的落地位置，别在 adapter 里再写一遍）
    """

    return ToolCall(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        name=name,
        arguments=raw_arguments,
        raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
        index=index,
    )


# --------------------------------------------------------------------------- #
# Token 估算（厂商没返回 usage 时兜底，标记 estimated=True）
# --------------------------------------------------------------------------- #

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """按字符数粗估 token，至少为 1（A4 要求缺失 usage 时 token 数 > 0）。"""

    return max(1, len(text) // _CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- #
# SSE 解析（唯一允许三家共用的解析代码）
# --------------------------------------------------------------------------- #


class SSELineKind(StrEnum):
    """一行 SSE 的分类。"""

    SKIP = "skip"
    EVENT = "event"
    DATA = "data"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class SSELine:
    """解析后的一行 SSE。"""

    kind: SSELineKind
    event: str | None = None
    payload: str | None = None


_SKIP = SSELine(kind=SSELineKind.SKIP)


def parse_sse_line(raw: str | bytes) -> SSELine:
    """解析单行 SSE。

    必须处理的五种行：

    | 行 | 处理 |
    |---|---|
    | 空行 | SKIP（事件分隔符） |
    | `: keep-alive` | SKIP（注释，不是 JSON；DeepSeek 长思考时会发） |
    | `event: xxx` | EVENT（仅 Anthropic） |
    | `data: [DONE]` | DONE（**先判等再 parse**，否则 json.loads 会崩） |
    | `data: {...}` | DATA |

    不支持多行 data 拼接：四家实际都是一行一个 JSON 事件。
    没遇到真实证据前不预先实现，避免"测自己写的解析器"。
    """

    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
    line = text.rstrip("\r\n")

    stripped = line.strip()

    if not stripped:
        return _SKIP

    # 以 ":" 开头是注释行（SSE 心跳）。
    if stripped.startswith(":"):
        return _SKIP

    name, separator, value = line.partition(":")

    if not separator:
        return _SKIP

    field_name = name.strip().lower()

    # SSE 规范：只去掉一个前导空格。
    if value.startswith(" "):
        value = value[1:]

    if field_name == "event":
        event = value.strip()

        return SSELine(kind=SSELineKind.EVENT, event=event or None)

    if field_name != "data":
        # id: / retry: 等字段与业务无关。
        return _SKIP

    if not value.strip():
        return _SKIP

    # 只去掉一个前导空格，其余空白属于 payload（json.loads 自己会忽略前导空白）。
    if value.strip() == "[DONE]":
        return SSELine(kind=SSELineKind.DONE)

    return SSELine(kind=SSELineKind.DATA, payload=value)


def decode_payload(payload: str | None, *, provider: str) -> dict[str, Any]:
    """把 SSE data 解析成 dict。

    坏 JSON 属于"上游协议异常"，不是调用方的错，因此映射为可重试的 UpstreamServerError。
    """

    if not payload:
        raise UpstreamServerError("empty SSE data payload", provider=provider)

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise UpstreamServerError(
            f"invalid SSE JSON payload: {exc.msg}",
            provider=provider,
            raw_error=payload[:500],
        ) from exc

    if not isinstance(decoded, dict):
        raise UpstreamServerError(
            "SSE payload must be a JSON object",
            provider=provider,
            raw_error=payload[:500],
        )

    return decoded


# --------------------------------------------------------------------------- #
# 错误响应体规范化
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """从厂商错误响应里提取的公共字段。"""

    code: str | None = None
    message: str = "LLM request failed"
    request_id: str | None = None


def _request_id(payload: dict[str, Any]) -> str | None:
    for key in ("request_id", "requestId", "id"):
        value = payload.get(key)

        if isinstance(value, str) and value:
            return value

    return None


def extract_error_info(payload: Any) -> ErrorInfo:
    """从各家五花八门的错误体里取出 (code, message, request_id)。

    覆盖三种形状：

    - OpenAI / DeepSeek：{"error": {"code": ..., "type": ..., "message": ...}}
    - Anthropic：{"type": "error", "error": {"type": ..., "message": ...}, "request_id": ...}
    - Qwen / DashScope：{"code": ..., "message": ..., "request_id": ...}

    code 优先级 code > type：Qwen 只有 code 有语义，OpenAI 的 code 比 type 更具体。
    """

    if isinstance(payload, str):
        return ErrorInfo(message=payload.strip() or "LLM request failed")

    if not isinstance(payload, dict):
        return ErrorInfo()

    error = payload.get("error")

    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        message = error.get("message") or error.get("msg")

        return ErrorInfo(
            code=str(code) if code else None,
            message=str(message) if message else "LLM request failed",
            request_id=_request_id(payload) or _request_id(error),
        )

    if isinstance(error, str):
        return ErrorInfo(message=error, request_id=_request_id(payload))

    code = payload.get("code") or payload.get("error_code")
    message = payload.get("message") or payload.get("msg")

    return ErrorInfo(
        code=str(code) if code else None,
        message=str(message) if message else "LLM request failed",
        request_id=_request_id(payload),
    )


def parse_retry_after(headers: Any) -> float | None:
    """读取 Retry-After（秒）。

    只支持数字形式：HTTP-date 形式没法直接参与退避计算，宁可退回指数退避。
    DeepSeek 干脆不返回这个 header，所以退避策略必须有指数兜底。
    """

    getter = getattr(headers, "get", None)

    if getter is None:
        return None

    raw = getter("retry-after")

    if raw is None:
        return None

    try:
        value = float(str(raw).strip())
    except ValueError:
        return None

    return value if value >= 0 else None


# --------------------------------------------------------------------------- #
# 流式聚合（测试比对 + W9 planner 在线消费）
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class StreamAccumulator:
    """把 StreamEvent 序列聚合成"等价于非流式响应"的形状。"""

    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Any = None
    raw_finish_reason: str | None = None
    usage: Any = None
    provider_meta: dict[str, Any] = field(default_factory=dict)

    def feed(self, events: list[StreamEvent]) -> None:
        for event in events:
            if event.text_delta:
                self.text_parts.append(event.text_delta)
            elif event.reasoning_delta:
                self.reasoning_parts.append(event.reasoning_delta)
            elif event.tool_call is not None:
                self.tool_calls.append(event.tool_call)

            if event.finish_reason is not None:
                self.finish_reason = event.finish_reason

            if event.raw_finish_reason is not None:
                self.raw_finish_reason = event.raw_finish_reason

            if event.usage is not None:
                self.usage = event.usage

            meta = event.data.get("provider_meta")

            if isinstance(meta, dict):
                self.provider_meta = meta

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

"""OpenAI 兼容协议适配器（覆盖 OpenAI / DeepSeek / Qwen / vLLM / Ollama）。

设计要点（详见 docs/w1/w1-day3.md §4）：

1. **三段分离**：build_request / parse_response / Assembler，全部无 IO、可离线单测。
   唯一碰网络的是 chat()/stream() 那层薄壳，薄壳里不许有业务逻辑。
2. **差异全部走 flags**，不写 `if "qwen" in base_url`。加第 5 家只改 presets 表。
3. **不重试、不打日志、不做计量**（ADR-003）。降级说明通过 PreparedRequest.notes 交给 client。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel

from ..base import Capability, LLMClient
from ..errors import BadRequestError, UpstreamServerError, map_provider_error
from ..types import (
    LLMResponse,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolSpec,
    Usage,
    normalize_finish_reason,
)
from ._common import (
    PreparedRequest,
    SSELineKind,
    content_to_text,
    decode_payload,
    estimate_tokens,
    extract_error_info,
    filter_opts,
    make_tool_call,
    parse_retry_after,
    parse_sse_line,
    prompt_text,
)

__all__ = [
    "PROVIDER_FLAGS",
    "OpenAICompatAssembler",
    "OpenAICompatClient",
    "OpenAICompatFlags",
    "PreserveReasoning",
    "ThinkingStyle",
]


class ThinkingStyle(StrEnum):
    """思考模式在 wire 上的表达方式。"""

    NONE = "none"
    THINKING_OBJECT = "thinking"
    ENABLE_THINKING = "enable_thinking"


class PreserveReasoning(StrEnum):
    """多轮回传 reasoning_content 的策略。"""

    NEVER = "never"
    ALWAYS = "always"
    PER_MODEL = "per_model"


class OpenAICompatFlags(BaseModel):
    """同一份协议在不同厂商上的差异开关。

    来源是 models.yaml（Day4 落盘）；W1 允许测试直接构造。
    默认值取 OpenAI 的语义，presets 里再覆盖成各家真实值。
    """

    # None = 不发该字段（DeepSeek 没有这个参数）
    send_parallel_tool_calls: bool | None = None

    # tool_choice: "required" / 指定函数 是否被支持（Qwen、DeepSeek 思考模式不支持）
    supports_tool_choice_required: bool = True
    supports_named_tool_choice: bool = True

    thinking_style: ThinkingStyle = ThinkingStyle.NONE

    # 流式会发 ": keep-alive" 注释行（DeepSeek），宽容模式跳过未知行
    stream_keepalive_comment: bool = False

    preserve_reasoning_content: PreserveReasoning = PreserveReasoning.NEVER

    # PER_MODEL 策略下由 registry 按模型注入（例如 Qwen 3.8 必须回传、3.7 不能回传）
    reasoning_replay_override: bool = False

    # 是否发送 stream_options.include_usage（不发就完全没有 usage，成本埋点归零）
    send_stream_options: bool = True


PROVIDER_FLAGS: dict[str, OpenAICompatFlags] = {
    "openai": OpenAICompatFlags(
        send_parallel_tool_calls=True,
    ),
    "deepseek": OpenAICompatFlags(
        # 没有 parallel_tool_calls 参数；思考模式默认开启且 reasoning 必须回传
        send_parallel_tool_calls=None,
        supports_tool_choice_required=False,
        supports_named_tool_choice=False,
        thinking_style=ThinkingStyle.THINKING_OBJECT,
        stream_keepalive_comment=True,
        preserve_reasoning_content=PreserveReasoning.ALWAYS,
    ),
    "qwen": OpenAICompatFlags(
        # 默认 false，不显式开启并行就静默变串行
        send_parallel_tool_calls=True,
        supports_tool_choice_required=False,
        supports_named_tool_choice=False,
        thinking_style=ThinkingStyle.ENABLE_THINKING,
        preserve_reasoning_content=PreserveReasoning.PER_MODEL,
    ),
    # 本地推理常用：协议同 OpenAI，但没有思考参数
    "vllm": OpenAICompatFlags(send_parallel_tool_calls=True),
    "ollama": OpenAICompatFlags(send_parallel_tool_calls=None),
}


# --------------------------------------------------------------------------- #
# wire -> internal 的公共解析
# --------------------------------------------------------------------------- #


def _content_text(content: Any) -> str | None:
    """OpenAI 的 content 可能是 None / str / 分片数组。"""

    if content is None:
        return None

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "output_text")
        ]
        joined = "".join(parts)

        return joined or None

    return None


def _usage_from_wire(
    raw: Any,
    *,
    text: str,
    prompt_text: str | None,
) -> Usage:
    """把 usage 映射为内部结构；缺失时估算并标记 estimated。

    注意 DeepSeek 把 usage 挂在**最后一个内容块**上（choices 非空），
    所以调用方不能靠 "choices 为空" 判断有没有 usage。
    """

    if isinstance(raw, dict):
        return Usage(
            input_tokens=int(raw.get("prompt_tokens") or 0),
            output_tokens=int(raw.get("completion_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
            estimated=False,
        )

    return Usage(
        input_tokens=estimate_tokens(prompt_text or ""),
        output_tokens=estimate_tokens(text),
        estimated=True,
    )


# --------------------------------------------------------------------------- #
# 流式组装器（按 delta index 累加参数，ADR-002）
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _ToolCallSlot:
    """一次工具调用在流式过程中的累积状态。"""

    index: int
    call_id: str | None = None
    name: str = ""
    argument_parts: list[str] = field(default_factory=list)


class OpenAICompatAssembler:
    """把 OpenAI 系 SSE chunk 组装成内部事件。

    状态只在这里：丢片、乱序、第二个工具调用被并进第一个，这三类 bug
    只有把状态抽出来才能离线断言（喂 5 个 chunk → 出 1 个完整 ToolCall）。
    """

    def __init__(self, *, provider: str, prompt_text: str | None = None) -> None:
        self._provider = provider
        self._prompt_text = prompt_text
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._slots: dict[int, _ToolCallSlot] = {}
        self._raw_finish_reason: str | None = None
        self._usage: Usage | None = None
        self._finalized = False

    # -- 对外 ------------------------------------------------------------ #

    def feed_line(self, raw: str | bytes) -> list[StreamEvent]:
        """吃一行原始 SSE，吐出内部事件。"""

        line = parse_sse_line(raw)

        if line.kind is SSELineKind.SKIP or line.kind is SSELineKind.DONE:
            # [DONE] 只表示"没有更多 chunk"，收尾由 finalize() 统一做，
            # 这样 usage 与 finish_reason 能合并进同一个 message_end。
            return []

        if line.kind is SSELineKind.EVENT:
            # OpenAI 系不发 event 行；容忍但不当数据。
            return []

        return self.feed_chunk(decode_payload(line.payload, provider=self._provider))

    def feed_chunk(self, chunk: dict[str, Any]) -> list[StreamEvent]:
        """吃一个已 json 化的 chunk。"""

        events: list[StreamEvent] = []

        # usage 覆盖式记录：只要出现就更新，不看 choices 是否为空（DeepSeek 形状差异）。
        if chunk.get("usage"):
            self._usage = _usage_from_wire(
                chunk["usage"],
                text="".join(self._text_parts),
                prompt_text=self._prompt_text,
            )

        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta") or {}

            content = delta.get("content")

            if isinstance(content, str) and content:
                self._text_parts.append(content)
                events.append(
                    StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=content)
                )

            # reasoning 不进 text（A7），只作为独立事件 + 结束时的 provider_meta
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")

            if isinstance(reasoning, str) and reasoning:
                self._reasoning_parts.append(reasoning)
                events.append(
                    StreamEvent(
                        type=StreamEventType.REASONING_DELTA,
                        reasoning_delta=reasoning,
                    )
                )

            for raw_call in delta.get("tool_calls") or []:
                self._absorb(raw_call)

            raw_finish = choice.get("finish_reason")

            if raw_finish:
                self._raw_finish_reason = str(raw_finish)

        return events

    def finalize(self) -> list[StreamEvent]:
        """流结束后调用一次：补齐 tool_call + 唯一的 message_end。"""

        if self._finalized:
            return []

        self._finalized = True

        events: list[StreamEvent] = []

        for index in sorted(self._slots):
            slot = self._slots[index]
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_CALL,
                    tool_call=make_tool_call(
                        call_id=slot.call_id,
                        name=slot.name,
                        raw_arguments="".join(slot.argument_parts),
                        index=index,
                    ),
                )
            )

        meta: dict[str, Any] = {}

        if self._reasoning_parts:
            meta["reasoning_content"] = "".join(self._reasoning_parts)

        usage = self._usage or _usage_from_wire(
            None,
            text="".join(self._text_parts),
            prompt_text=self._prompt_text,
        )

        events.append(
            StreamEvent(
                type=StreamEventType.MESSAGE_END,
                finish_reason=normalize_finish_reason(self._raw_finish_reason),
                raw_finish_reason=self._raw_finish_reason,
                usage=usage,
                data={"provider_meta": meta} if meta else {},
            )
        )

        return events

    # -- 内部 ------------------------------------------------------------ #

    def _absorb(self, raw_call: Any) -> None:
        """把一个 tool_calls 分片并入对应槽位。

        key 必须用 `index`：后续分片没有 id，用 id 做 key 会把第二个工具调用并进第一个。
        """

        if not isinstance(raw_call, dict):
            return

        raw_index = raw_call.get("index")

        if isinstance(raw_index, int):
            index = raw_index
        elif self._slots:
            index = max(self._slots)
        else:
            index = 0

        slot = self._slots.get(index)

        if slot is None:
            slot = _ToolCallSlot(index=index)
            self._slots[index] = slot

        call_id = raw_call.get("id")

        if isinstance(call_id, str) and call_id:
            slot.call_id = call_id

        function = raw_call.get("function") or {}

        name = function.get("name")

        # 名字高度集中在首片；后续片出现同名是重复信息，不同名才是分片拼接。
        if isinstance(name, str) and name:
            slot.name = name if not slot.name else slot.name + name

        arguments = function.get("arguments")

        if isinstance(arguments, str):
            slot.argument_parts.append(arguments)


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #


class OpenAICompatClient(LLMClient):
    """OpenAI 兼容协议客户端。"""

    _path = "/chat/completions"

    def __init__(
        self,
        *,
        name: str = "openai",
        model: str,
        base_url: str,
        api_key: str,
        flags: OpenAICompatFlags | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._flags = flags or PROVIDER_FLAGS.get(name) or OpenAICompatFlags()
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            transport=transport,
        )

    # -- 元信息 ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> frozenset[Capability]:
        caps = {Capability.STREAM, Capability.TOOLS, Capability.JSON_MODE}

        # None 表示该厂商根本没有这个开关（DeepSeek），按"未声明即不支持"处理：
        # 声称支持却拿到串行结果，比不声称更糟。
        if self._flags.send_parallel_tool_calls is True:
            caps.add(Capability.PARALLEL_TOOLS)

        return frozenset(caps)

    # -- 请求构造（纯函数，可离线断言） ---------------------------------- #

    def build_request(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking: bool | None = None,
    ) -> PreparedRequest:
        notes: list[str] = []

        body: dict[str, Any] = {
            "model": self._model,
            "messages": [self._encode_message(message) for message in messages],
        }

        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        if temperature is not None:
            body["temperature"] = temperature

        if tools:
            body["tools"] = [self._encode_tool(tool) for tool in tools]

            # 并行工具调用只在有工具时才有意义（部分厂商无工具时会报错）。
            if self._flags.send_parallel_tool_calls is not None:
                body["parallel_tool_calls"] = self._flags.send_parallel_tool_calls

        encoded_choice = self._encode_tool_choice(tool_choice, notes)

        if encoded_choice is not None:
            body["tool_choice"] = encoded_choice

        self._apply_thinking(body, thinking=thinking, stream=stream, notes=notes)

        if stream:
            body["stream"] = True

            if self._flags.send_stream_options:
                body["stream_options"] = {"include_usage": True}

        return PreparedRequest(body=body, notes=tuple(notes))

    def _encode_message(self, message: Message) -> dict[str, Any]:
        role = str(message.role)
        text = content_to_text(message.content)

        if role == MessageRole.TOOL.value:
            encoded: dict[str, Any] = {"role": role, "content": text}

            if message.tool_call_id:
                encoded["tool_call_id"] = message.tool_call_id

            if message.name:
                encoded["name"] = message.name

            return encoded

        encoded = {"role": role, "content": text or None}

        if message.tool_calls:
            encoded["content"] = text or None
            encoded["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # OpenAI 要求 arguments 是 JSON 字符串（不是对象）
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]

        if role == MessageRole.ASSISTANT.value and self._should_replay_reasoning():
            reasoning = (message.provider_meta or {}).get("reasoning_content")

            if isinstance(reasoning, str) and reasoning:
                encoded["reasoning_content"] = reasoning

        return encoded

    def _encode_tool(self, tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _encode_tool_choice(
        self,
        choice: ToolChoice | None,
        notes: list[str],
    ) -> Any:
        """tool_choice 是**策略意图**，不是硬保证：不支持就降级并记 notes。

        与并行开关的方向容易搞反：OpenAI 是 parallel_tool_calls=true 表示开启，
        Anthropic 是 disable_parallel_tool_use=true 表示关闭，语义相反。
        """

        if choice is None:
            return None

        if choice.mode is ToolChoiceMode.NONE:
            return "none"

        if choice.mode is ToolChoiceMode.AUTO:
            return "auto"

        if choice.name is not None:
            if not self._flags.supports_named_tool_choice:
                notes.append(
                    f"tool_choice named '{choice.name}' degraded to auto: provider unsupported"
                )
                return "auto"

            return {"type": "function", "function": {"name": choice.name}}

        if not self._flags.supports_tool_choice_required:
            notes.append("tool_choice required degraded to auto: provider unsupported")
            return "auto"

        return "required"

    def _apply_thinking(
        self,
        body: dict[str, Any],
        *,
        thinking: bool | None,
        stream: bool,
        notes: list[str],
    ) -> None:
        if thinking is None:
            # 不显式传就完全不动，交给厂商默认值（DeepSeek 默认开启）。
            return

        style = self._flags.thinking_style

        if style is ThinkingStyle.THINKING_OBJECT:
            body["thinking"] = {"type": "enabled" if thinking else "disabled"}

        elif style is ThinkingStyle.ENABLE_THINKING:
            if thinking and not stream:
                # Qwen：enable_thinking=true 必须配 stream=true，否则 400。
                raise BadRequestError(
                    "enable_thinking requires stream=true on this provider",
                    provider=self._name,
                    model=self._model,
                )

            body["enable_thinking"] = thinking

        elif thinking:
            notes.append("thinking ignored: provider has no thinking switch")

    def _should_replay_reasoning(self) -> bool:
        flag = self._flags.preserve_reasoning_content

        if flag is PreserveReasoning.ALWAYS:
            return True

        if flag is PreserveReasoning.NEVER:
            return False

        # PER_MODEL：由 registry 按具体模型注入；未注入时保守不回传
        # （Qwen 3.7 不回传是安全的，3.8 必须显式打开）。
        return self._flags.reasoning_replay_override

    # -- 响应解析（纯函数） ---------------------------------------------- #

    def make_assembler(self, *, prompt_text: str | None = None) -> OpenAICompatAssembler:
        return OpenAICompatAssembler(provider=self._name, prompt_text=prompt_text)

    def parse_response(
        self,
        body: dict[str, Any],
        *,
        prompt_text: str | None = None,
    ) -> LLMResponse:
        choices = body.get("choices")

        # 不做 body["choices"][0] 直取：一次异常响应就能打崩整个服务。
        if not isinstance(choices, list) or not choices:
            raise UpstreamServerError(
                "response contains no choices",
                provider=self._name,
                model=self._model,
                raw_error=body,
            )

        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}

        text = _content_text(message.get("content"))

        tool_calls: list[ToolCall] = []

        for index, raw_call in enumerate(message.get("tool_calls") or []):
            if not isinstance(raw_call, dict):
                continue

            function = raw_call.get("function")
            function = function if isinstance(function, dict) else {}

            tool_calls.append(
                make_tool_call(
                    call_id=raw_call.get("id"),
                    name=str(function.get("name") or ""),
                    # 字符串交给 ToolCall 校验器解析，坏 JSON 只降级不抛
                    raw_arguments=function.get("arguments"),
                    index=index,
                )
            )

        meta: dict[str, Any] = {}
        reasoning = message.get("reasoning_content") or message.get("reasoning")

        if isinstance(reasoning, str) and reasoning:
            meta["reasoning_content"] = reasoning

        raw_finish_reason = choice.get("finish_reason")
        raw_finish_reason = str(raw_finish_reason) if raw_finish_reason else None

        return LLMResponse(
            id=body.get("id"),
            model=body.get("model") or self._model,
            provider=self._name,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=text,
                tool_calls=tool_calls,
                provider_meta=meta or None,
            ),
            finish_reason=normalize_finish_reason(raw_finish_reason),
            raw_finish_reason=raw_finish_reason,
            usage=_usage_from_wire(
                body.get("usage"),
                text=text or "",
                prompt_text=prompt_text,
            ),
        )

    # -- 网络薄壳（唯一有 IO 的地方，不许有业务逻辑） -------------------- #

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: Any,
    ) -> LLMResponse:
        prepared = self.build_request(
            messages,
            tools,
            tool_choice=tool_choice,
            stream=False,
            **filter_opts(opts, _NETWORK_OPTS),
        )

        response = await self._http.post(self._path, json=prepared.body)
        self._ensure_success(response)

        return self.parse_response(
            response.json(),
            prompt_text=prompt_text(messages),
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamEvent]:
        prepared = self.build_request(
            messages,
            tools,
            tool_choice=tool_choice,
            stream=True,
            **filter_opts(opts, _NETWORK_OPTS),
        )
        assembler = self.make_assembler(prompt_text=prompt_text(messages))

        async with self._http.stream("POST", self._path, json=prepared.body) as response:
            if not response.is_success:
                await response.aread()
                self._ensure_success(response)

            async for line in response.aiter_lines():
                for event in _tolerate(self._flags, assembler, line, self._name):
                    yield event

        for event in assembler.finalize():
            yield event

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def _ensure_success(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text

        info = extract_error_info(payload)

        raise map_provider_error(
            provider=self._name,
            model=self._model,
            status_code=response.status_code,
            provider_code=info.code,
            message=info.message,
            retry_after=parse_retry_after(response.headers),
            request_id=info.request_id,
            raw_error=payload,
        )


# --------------------------------------------------------------------------- #
# 薄壳辅助
# --------------------------------------------------------------------------- #

_NETWORK_OPTS = ("max_tokens", "temperature", "thinking")


def _tolerate(
    flags: OpenAICompatFlags,
    assembler: OpenAICompatAssembler,
    line: str,
    provider: str,
) -> list[StreamEvent]:
    """解析一行，按 flag 决定对协议噪声的宽容度。

    stream_keepalive_comment=True 的厂商（DeepSeek）会在长思考时插入注释行，
    此时任何无法识别为 SSE 的行都只跳过；其他厂商则应视为协议异常。
    """

    try:
        return assembler.feed_line(line)
    except UpstreamServerError:
        if flags.stream_keepalive_comment:
            return []

        raise

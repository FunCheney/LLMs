"""Anthropic Messages API 适配器。

与 OpenAI 系的四处**结构性**差异（不是字段改名，是消息结构不同）：

| 差异 | Anthropic | OpenAI 系 |
|---|---|---|
| system | 顶层 `system` 参数，messages 里不许有 system 角色 | messages[0] |
| tool 定义 | 顶层 `tools`，schema 字段叫 `input_schema` | `tools[].function.parameters` |
| tool 结果 | **合并成一条 user 消息**，内含全部 `tool_result` | 每条一个 `role:"tool"` |
| tool 参数 | `input` 已经是对象 | `arguments` 是 JSON 字符串 |

合并逻辑**只放这里**：一旦"顺手"在 types.py 里加个合并辅助，另外三家就为它付账。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

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

__all__ = ["AnthropicAssembler", "AnthropicClient"]

_DEFAULT_THINKING_BUDGET = 1024


@dataclass(slots=True)
class _BlockState:
    """一个 content block 的流式状态。"""

    index: int
    kind: str
    call_id: str | None = None
    name: str = ""
    argument_parts: list[str] = field(default_factory=list)


class AnthropicAssembler:
    """把 Anthropic SSE 事件组装成内部事件。

    与 OpenAI 系的根本区别：分片标识是 `content_block` 的 index，
    结束信号是每个 block 各自的 `content_block_stop`（一个响应里可能有
    text + 2 个 tool_use，不能只用"当前块"）。
    """

    def __init__(self, *, provider: str = "anthropic", prompt_text: str | None = None) -> None:
        self._provider = provider
        self._prompt_text = prompt_text
        self._current_event: str | None = None
        self._blocks: dict[int, _BlockState] = {}
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._signature: str | None = None
        self._raw_stop_reason: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._finalized = False

    # -- 对外 ------------------------------------------------------------ #

    def feed_line(self, raw: str | bytes) -> list[StreamEvent]:
        """吃一行 SSE。`event:` 与 `data:` 是两行，所以要记住当前事件名。"""

        line = parse_sse_line(raw)

        if line.kind is SSELineKind.SKIP or line.kind is SSELineKind.DONE:
            return []

        if line.kind is SSELineKind.EVENT:
            self._current_event = line.event
            return []

        event_name = self._current_event or ""
        payload = decode_payload(line.payload, provider=self._provider)

        return self.feed_event(event_name, payload)

    def feed_event(self, event_name: str, payload: dict[str, Any]) -> list[StreamEvent]:
        """吃一个已解析的 (event_name, payload)。"""

        handler = getattr(self, f"_on_{event_name}", None)

        if handler is None:
            # ping / 未知事件：直接忽略，绝不能当业务事件。
            return []

        result: list[StreamEvent] = handler(payload)

        return result

    def finalize(self) -> list[StreamEvent]:
        if self._finalized:
            return []

        self._finalized = True

        meta: dict[str, Any] = {}
        thinking = "".join(self._thinking_parts)

        if thinking or self._signature:
            block: dict[str, Any] = {"type": "thinking", "thinking": thinking}

            if self._signature:
                block["signature"] = self._signature

            meta["thinking_blocks"] = [block]

        if self._input_tokens is None and self._output_tokens is None:
            usage = Usage(
                input_tokens=estimate_tokens(self._prompt_text or ""),
                output_tokens=estimate_tokens("".join(self._text_parts)),
                estimated=True,
            )
        else:
            usage = Usage(
                input_tokens=self._input_tokens or 0,
                output_tokens=self._output_tokens or 0,
                estimated=False,
            )

        return [
            StreamEvent(
                type=StreamEventType.MESSAGE_END,
                finish_reason=normalize_finish_reason(self._raw_stop_reason),
                raw_finish_reason=self._raw_stop_reason,
                usage=usage,
                data={"provider_meta": meta} if meta else {},
            )
        ]

    # -- 事件处理 -------------------------------------------------------- #

    def _on_message_start(self, payload: dict[str, Any]) -> list[StreamEvent]:
        message = payload.get("message")

        if isinstance(message, dict):
            usage = message.get("usage")

            if isinstance(usage, dict):
                # 流式下 input_tokens 只在这里；拿 output_tokens 计费会系统性偏低。
                self._input_tokens = int(usage.get("input_tokens") or 0)

        return []

    def _on_content_block_start(self, payload: dict[str, Any]) -> list[StreamEvent]:
        index = int(payload.get("index") or 0)
        block = payload.get("content_block")
        block = block if isinstance(block, dict) else {}
        kind = str(block.get("type") or "")

        self._blocks[index] = _BlockState(
            index=index,
            kind=kind,
            call_id=block.get("id"),
            name=str(block.get("name") or ""),
        )

        return []

    def _on_content_block_delta(self, payload: dict[str, Any]) -> list[StreamEvent]:
        index = int(payload.get("index") or 0)
        delta = payload.get("delta")
        delta = delta if isinstance(delta, dict) else {}
        delta_type = str(delta.get("type") or "")

        if delta_type == "text_delta":
            text = delta.get("text")

            if isinstance(text, str) and text:
                self._text_parts.append(text)

                return [StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=text)]

            return []

        if delta_type == "thinking_delta":
            thinking = delta.get("thinking")

            if isinstance(thinking, str) and thinking:
                self._thinking_parts.append(thinking)

                return [
                    StreamEvent(
                        type=StreamEventType.REASONING_DELTA,
                        reasoning_delta=thinking,
                    )
                ]

            return []

        if delta_type == "signature_delta":
            signature = delta.get("signature")

            if isinstance(signature, str) and signature:
                self._signature = signature

            return []

        if delta_type == "input_json_delta":
            partial = delta.get("partial_json")
            block = self._blocks.get(index)

            if block is not None and isinstance(partial, str):
                block.argument_parts.append(partial)

            return []

        return []

    def _on_content_block_stop(self, payload: dict[str, Any]) -> list[StreamEvent]:
        index = int(payload.get("index") or 0)
        block = self._blocks.get(index)

        if block is None or block.kind != "tool_use":
            return []

        # content_block_stop 就是"参数完整"的信号 → 此刻发一个完整 ToolCall（ADR-002）。
        return [
            StreamEvent(
                type=StreamEventType.TOOL_CALL,
                tool_call=make_tool_call(
                    call_id=block.call_id,
                    name=block.name,
                    raw_arguments="".join(block.argument_parts),
                    index=index,
                ),
            )
        ]

    def _on_message_delta(self, payload: dict[str, Any]) -> list[StreamEvent]:
        delta = payload.get("delta")
        delta = delta if isinstance(delta, dict) else {}

        stop_reason = delta.get("stop_reason")

        if stop_reason:
            self._raw_stop_reason = str(stop_reason)

        usage = payload.get("usage")

        if isinstance(usage, dict) and usage.get("output_tokens") is not None:
            # 累计值，取最后一次覆盖。
            self._output_tokens = int(usage.get("output_tokens") or 0)

        return []

    def _on_message_stop(self, payload: dict[str, Any]) -> list[StreamEvent]:
        return []

    def _on_error(self, payload: dict[str, Any]) -> list[StreamEvent]:
        info = extract_error_info(payload)

        raise map_provider_error(
            provider=self._provider,
            provider_code=info.code,
            message=info.message,
            request_id=info.request_id,
            raw_error=payload,
        )


class AnthropicClient(LLMClient):
    """Anthropic Messages API 客户端。"""

    _path = "/v1/messages"

    def __init__(
        self,
        *,
        name: str = "anthropic",
        model: str,
        base_url: str = "https://api.anthropic.com",
        api_key: str,
        anthropic_version: str = "2023-06-01",
        default_max_tokens: int | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": anthropic_version,
                "content-type": "application/json",
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
        return frozenset(
            {
                Capability.STREAM,
                Capability.TOOLS,
                Capability.PARALLEL_TOOLS,
                Capability.PROMPT_CACHING,
            }
        )

    # -- 请求构造 -------------------------------------------------------- #

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
        thinking_budget: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> PreparedRequest:
        notes: list[str] = []

        resolved_max_tokens = max_tokens or self._default_max_tokens

        if resolved_max_tokens is None:
            # 静默兜底会让"上游截断"变成"我们莫名截断"，报错必须指向根因。
            raise BadRequestError(
                "Anthropic requires max_tokens",
                provider=self._name,
                model=self._model,
            )

        system_text, encoded_messages = self._encode_messages(messages)
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": resolved_max_tokens,
            "messages": encoded_messages,
        }

        if system_text:
            body["system"] = system_text

        if temperature is not None:
            body["temperature"] = temperature

        if tools:
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]

        encoded_choice = self._encode_tool_choice(tool_choice, parallel_tool_calls, notes)

        if encoded_choice is not None:
            body["tool_choice"] = encoded_choice

        if thinking:
            budget = thinking_budget or _DEFAULT_THINKING_BUDGET

            if resolved_max_tokens <= budget:
                raise BadRequestError(
                    "max_tokens must be greater than thinking budget",
                    provider=self._name,
                    model=self._model,
                )

            body["thinking"] = {"type": "enabled", "budget_tokens": budget}

        if stream:
            body["stream"] = True

        return PreparedRequest(body=body, notes=tuple(notes))

    def _encode_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """把内部消息序列转成 Anthropic 形态。

        两个必须做对的地方：

        1. system 提升到顶层（messages 里出现 system 角色会 400）
        2. 连续多条 tool 消息合并成**一条** user 消息，内含全部 tool_result，
           少给一个就 400，且报错信息会误导成"id 不匹配"
        """

        system_parts: list[str] = []
        encoded: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            if pending_results:
                encoded.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for message in messages:
            if message.role is MessageRole.SYSTEM:
                system_parts.append(content_to_text(message.content))
                continue

            if message.role is MessageRole.TOOL:
                pending_results.append(self._encode_tool_result(message))
                continue

            flush_results()
            encoded.append(self._encode_message(message))

        flush_results()

        return "\n\n".join(part for part in system_parts if part), encoded

    def _encode_message(self, message: Message) -> dict[str, Any]:
        text = content_to_text(message.content)
        role = "assistant" if message.role is MessageRole.ASSISTANT else "user"

        blocks: list[dict[str, Any]] = []

        if message.role is MessageRole.ASSISTANT:
            # thinking block 必须原样回传（带 signature），否则多轮思考场景 400。
            for block in (message.provider_meta or {}).get("thinking_blocks") or []:
                if isinstance(block, dict):
                    blocks.append(block)

        if text:
            blocks.append({"type": "text", "text": text})

        for call in message.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )

        if not message.tool_calls and len(blocks) <= 1:
            return {"role": role, "content": text}

        return {"role": role, "content": blocks}

    def _encode_tool_result(self, message: Message) -> dict[str, Any]:
        text = content_to_text(message.content) or "(no output)"
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id,
            "content": text,
        }

        if (message.provider_meta or {}).get("is_error"):
            block["is_error"] = True

        return block

    def _encode_tool_choice(
        self,
        choice: ToolChoice | None,
        parallel_tool_calls: bool | None,
        notes: list[str],
    ) -> dict[str, Any] | None:
        """Anthropic 的 tool_choice 是**对象**，且并行开关是它的子字段。"""

        if choice is None and parallel_tool_calls is None:
            return None

        if choice is None:
            encoded: dict[str, Any] = {"type": "auto"}
        elif choice.mode is ToolChoiceMode.NONE:
            encoded = {"type": "none"}
        elif choice.mode is ToolChoiceMode.AUTO:
            encoded = {"type": "auto"}
        elif choice.name is not None:
            encoded = {"type": "tool", "name": choice.name}
        else:
            encoded = {"type": "any"}

        # 注意语义相反：这里是"禁止并行"，OpenAI 系是"允许并行"。
        if parallel_tool_calls is False:
            encoded["disable_parallel_tool_use"] = True

        if choice is not None and choice.mode is ToolChoiceMode.REQUIRED and parallel_tool_calls:
            notes.append("parallel_tool_calls ignored: anthropic only supports disabling")

        return encoded

    # -- 响应解析 -------------------------------------------------------- #

    def make_assembler(self, *, prompt_text: str | None = None) -> AnthropicAssembler:
        return AnthropicAssembler(provider=self._name, prompt_text=prompt_text)

    def parse_response(
        self,
        body: dict[str, Any],
        *,
        prompt_text: str | None = None,
    ) -> LLMResponse:
        blocks = body.get("content")

        if not isinstance(blocks, list):
            raise UpstreamServerError(
                "response contains no content blocks",
                provider=self._name,
                model=self._model,
                raw_error=body,
            )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_blocks: list[dict[str, Any]] = []

        for block in blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                value = block.get("text")

                if isinstance(value, str):
                    text_parts.append(value)

            elif block_type == "tool_use":
                tool_calls.append(
                    make_tool_call(
                        call_id=block.get("id"),
                        name=str(block.get("name") or ""),
                        # Anthropic 的 input 已经是对象，不需要 json.loads
                        raw_arguments=block.get("input"),
                    )
                )

            elif block_type == "thinking":
                thinking_blocks.append(block)

        meta: dict[str, Any] = {}

        if thinking_blocks:
            meta["thinking_blocks"] = thinking_blocks

        text = "".join(text_parts) or None
        usage = body.get("usage")
        usage = usage if isinstance(usage, dict) else None

        if usage is None:
            resolved_usage = Usage(
                input_tokens=estimate_tokens(prompt_text or ""),
                output_tokens=estimate_tokens(text or ""),
                estimated=True,
            )
        else:
            resolved_usage = Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                estimated=False,
            )

        raw_stop_reason = body.get("stop_reason")
        raw_stop_reason = str(raw_stop_reason) if raw_stop_reason else None

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
            finish_reason=normalize_finish_reason(raw_stop_reason),
            raw_finish_reason=raw_stop_reason,
            usage=resolved_usage,
        )

    # -- 网络薄壳 -------------------------------------------------------- #

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

        return self.parse_response(response.json(), prompt_text=prompt_text(messages))

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
                for event in assembler.feed_line(line):
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


_NETWORK_OPTS = (
    "max_tokens",
    "temperature",
    "thinking",
    "thinking_budget",
    "parallel_tool_calls",
)

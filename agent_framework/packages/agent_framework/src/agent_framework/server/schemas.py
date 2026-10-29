"""HTTP 请求/响应契约。

红线：**响应必须是手写投影**。`LLMResponse.message.provider_meta` 里装着
`reasoning_content` / `thinking.signature`，直接 `model_dump()` 会把模型的思考内容
吐给调用方，顺带把内部契约暴露出去。

一句话记住：adapter 往内翻译，schema 往外投影，中间任何一层都不许整体序列化。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..llm.client import ChatCall, ChatOutcome
from ..llm.types import (
    LLMResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolSpec,
    Usage,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "MessageIn",
    "ToolCallIn",
    "ToolCallOut",
    "ToolSpecIn",
    "UsageOut",
]


class ToolCallIn(BaseModel):
    """多轮对话里回传的工具调用（assistant 消息）。"""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MessageIn(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCallIn] = Field(default_factory=list)

    def to_message(self) -> Message:
        return Message(
            role=MessageRole(self.role),
            content=self.content,
            name=self.name,
            tool_call_id=self.tool_call_id,
            tool_calls=[
                ToolCall(id=call.id, name=call.name, arguments=call.arguments)
                for call in self.tool_calls
            ],
        )


class ToolSpecIn(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ChatRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    task: str = "chat"
    tools: list[ToolSpecIn] | None = None

    tool_choice: Literal["auto", "none", "required"] | None = None
    # 仅 tool_choice=required 时有意义：表示"必须调用这个工具"。
    tool_choice_name: str | None = None

    stream: bool = False

    # 覆盖路由，仅调试/评测用。
    model: str | None = None

    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)

    @model_validator(mode="after")
    def validate_tool_choice(self) -> ChatRequest:
        if self.tool_choice_name is not None and self.tool_choice != "required":
            raise ValueError("tool_choice_name 只能与 tool_choice='required' 一起使用")

        return self

    def to_call(self) -> ChatCall:
        opts: dict[str, Any] = {}

        if self.max_tokens is not None:
            opts["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            opts["temperature"] = self.temperature

        choice: ToolChoice | None = None

        if self.tool_choice is not None:
            choice = ToolChoice(
                mode=ToolChoiceMode(self.tool_choice),
                name=self.tool_choice_name,
            )

        return ChatCall(
            messages=[message.to_message() for message in self.messages],
            task=self.task,
            tools=[tool.to_spec() for tool in self.tools] if self.tools else None,
            tool_choice=choice,
            model=self.model,
            opts=opts,
        )


class ToolCallOut(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # 参数 JSON 解析失败时非空（architecture §3.3：不许打断整轮）。
    arguments_error: str | None = None


class UsageOut(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

    @classmethod
    def from_usage(cls, usage: Usage) -> UsageOut:
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated=usage.estimated,
        )


class ChatResponse(BaseModel):
    request_id: str
    id: str | None = None
    provider: str | None = None
    model: str | None = None
    text: str | None = None
    tool_calls: list[ToolCallOut] = Field(default_factory=list)
    finish_reason: str = "unknown"
    raw_finish_reason: str | None = None
    usage: UsageOut = Field(default_factory=UsageOut)
    latency_ms: float | None = None
    retry_count: int = 0
    fallback_from: str | None = None

    @classmethod
    def from_response(
        cls,
        response: LLMResponse,
        *,
        request_id: str,
        retry_count: int = 0,
        fallback_from: str | None = None,
    ) -> ChatResponse:
        return cls(
            request_id=request_id or "-",
            id=response.id,
            provider=response.provider,
            model=response.model,
            text=response.text,
            tool_calls=[
                ToolCallOut(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    arguments_error=call.arguments_error,
                )
                for call in response.tool_calls
            ],
            finish_reason=str(response.finish_reason),
            raw_finish_reason=response.raw_finish_reason,
            usage=UsageOut.from_usage(response.usage),
            latency_ms=response.latency_ms,
            retry_count=retry_count,
            fallback_from=fallback_from,
        )

    @classmethod
    def from_outcome(cls, outcome: ChatOutcome, *, request_id: str) -> ChatResponse:
        return cls.from_response(
            outcome.response,
            request_id=request_id,
            retry_count=outcome.retry_count,
            fallback_from=outcome.fallback_from,
        )

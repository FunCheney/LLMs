import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FinishReason(StrEnum):
    """
    统一的模型结束原因。

    UNKNOWN 用于表示无法识别或无法映射的供应商结束原因。
    原始值应保存在 LLMResponse.raw_finish_reason 中。
    """

    UNKNOWN = "unknown"
    STOP = "stop"  # pause_turn → stop
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"  # refusal → content_filter
    ABORTED = "aborted"
    UPSTREAM_ERROR = "upstream_error"  # insufficient_system_resource → upstream_error


class MessageRole(StrEnum):
    """
    统一消息的角色
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StreamEventType(StrEnum):
    """
    统一的流式事件类型。

    只有 TOOL_CALL，没有 tool_call_start/delta：ADR-002 规定增量由 adapter 内部组装，
    只在参数完整后发一个事件。留 delta 类型等于给了上层"可以自己拼"的错觉。
    """

    MESSAGE_START = "message_start"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL = "tool_call"
    MESSAGE_END = "message_end"
    ERROR = "error"


class ContentBlock(BaseModel):
    """
    消息中的一个内容块。

    当前只提供通用结构，不在这里解释供应商特有的内容格式。
    后续可以扩展文本、图片、音频等内容块。
    """

    model_config = ConfigDict(extra="allow")

    type: str
    data: Any = None


class ToolSpec(BaseModel):
    """
    工具定义。

    用于描述工具名称、用途及其参数 JSON Schema。
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolChoiceMode(StrEnum):
    """调用方对工具选择的意图。"""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class ToolChoice(BaseModel):
    """
    工具选择策略。

    mode 是策略意图，不是硬保证：当目标厂商不支持某个取值时，
    由调用方（client 层的能力协商）降级，adapter 只负责如实翻译。

    name 仅在 mode=required 时允许出现，表示"必须调用这个工具"。
    """

    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    name: str | None = None

    @model_validator(mode="after")
    def validate_named_choice(self) -> "ToolChoice":
        """只有 required 模式可以指定具体工具。"""

        if self.name is not None and self.mode != ToolChoiceMode.REQUIRED:
            raise ValueError("tool_choice.name is only allowed when mode=required")

        return self


class ToolCall(BaseModel):
    """
    模型发起的一次工具调用。

    arguments 始终为 dict。
    当原始参数无法解析时，arguments 置为空字典，
    并通过 arguments_error 记录解析失败原因。
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_error: str | None = None
    raw_arguments: str | None = None
    index: int | None = None


    @model_validator(mode="before")
    @classmethod
    def normalize_arguments(cls, data: Any) -> Any:
        """
        将工具参数统一转换为字典。
        """

        if not isinstance(data, dict):
            return data

        data = dict(data)
        raw_arguments = data.get("arguments")

        # 已经是字典，无需解析。
        if isinstance(raw_arguments, dict):
            return data

        # None 表示没有提供参数，统一为空字典。
        if raw_arguments is None:
            data["arguments"] = {}
            return data

        # 某些供应商会将参数作为 JSON 字符串返回。
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)

                if isinstance(parsed, dict):
                    data["arguments"] = parsed
                else:
                    data["arguments"] = {}
                    data["arguments_error"] = (
                        "Tool arguments JSON must decode to an object"
                    )

            except json.JSONDecodeError as exc:
                data["arguments"] = {}
                data["arguments_error"] = (
                    f"Invalid tool arguments JSON: {exc.msg}"
                )

            return data

        # 其他类型不符合统一的数据结构。
        data["arguments"] = {}
        data["arguments_error"] = (
            f"Tool arguments must be a dict or JSON string, "
            f"got {type(raw_arguments).__name__}"
        )

        return data


class ToolResult(BaseModel):
    """
    工具执行结果。
    """

    tool_call_id: str
    content: Any = None
    is_error: bool = False


class Message(BaseModel):
    """
    统一的对话消息。

    provider_meta 用于原样保存供应商特有的消息元数据。

    例如：
    - reasoning_content
    - Anthropic thinking.signature

    业务层只能将 provider_meta 作为整体传递，不得自行构造、
    修改、拆解或解读其中的供应商特有字段。
    这些字段应由对应的 provider adapter 负责生成和消费。
    """

    role: MessageRole
    content: str | list[ContentBlock] | None = None

    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    provider_meta: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_tool_message(self) -> "Message":
        """tool 角色必须关联对应的工具调用。"""

        if self.role == MessageRole.TOOL and not self.tool_call_id:
            raise ValueError(
                "tool message must contain tool_call_id"
            )

        return self

    @model_validator(mode="after")
    def validate_assistant_content(self) -> "Message":
        """assistant 存在工具调用时，content 可以为空。"""

        if (
                self.role == MessageRole.ASSISTANT
                and self.tool_calls
                and self.content is None
        ):
            # 合法情况，不需要补充空字符串。
            return self

        return self


class Conversation(BaseModel):
    """一组对话消息，并校验消息之间的约束。"""

    messages: list[Message]

    @model_validator(mode="after")
    def validate_system_position(self) -> "Conversation":
        """system 消息只能出现在第一条。"""

        for index, message in enumerate(self.messages):
            if (
                    message.role == MessageRole.SYSTEM
                    and index != 0
            ):
                raise ValueError(
                    "system message can only appear as the first message"
                )

        return self


class Usage(BaseModel):
    """统一的 Token 用量统计。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # 标注该用量是否为估算值（厂商未返回 usage 时由 adapter 粗估）。
    estimated: bool = False

    @model_validator(mode="after")
    def fill_total_tokens(self) -> "Usage":
        """当总量未提供时，根据输入和输出 Token 数计算。"""

        if self.total_tokens == 0:
            self.total_tokens = (
                    self.input_tokens + self.output_tokens
            )

        return self


class LLMResponse(BaseModel):
    """统一的大模型非流式响应。"""

    id: str | None = None
    model: str | None = None

    # 产出该响应的 provider 名（openai / deepseek / qwen / anthropic ...）。
    provider: str | None = None

    message: Message

    finish_reason: FinishReason = FinishReason.UNKNOWN

    # 保留供应商返回的原始结束原因。
    raw_finish_reason: str | None = None

    usage: Usage = Field(default_factory=Usage)

    # 由 client 层填充（adapter 不测量延迟）。
    latency_ms: float | None = None

    @property
    def text(self) -> str | None:
        """便捷访问：消息中的纯文本内容。"""

        content = self.message.content

        if content is None:
            return None

        if isinstance(content, str):
            return content

        parts = [
            block.data
            for block in content
            if block.type == "text" and isinstance(block.data, str)
        ]

        return "".join(parts) or None

    @property
    def tool_calls(self) -> list[ToolCall]:
        """便捷访问：消息中的工具调用。"""

        return self.message.tool_calls

    @property
    def response_id(self) -> str | None:
        """便捷访问：厂商返回的响应 id。"""

        return self.id


class StreamEvent(BaseModel):
    """
    统一的流式响应事件。

    text_delta 用于增量文本输出。
    reasoning_delta 用于增量推理内容输出。
    tool_call 用于完整的工具调用信息。
    data 用于扩展其他类型的事件数据。
    """

    type: StreamEventType

    text_delta: str | None = None
    reasoning_delta: str | None = None

    tool_call: ToolCall | None = None

    finish_reason: FinishReason | None = None
    raw_finish_reason: str | None = None

    usage: Usage | None = None
    data: dict[str, Any] = Field(default_factory=dict)


def normalize_finish_reason(raw: str | None,) -> FinishReason:
    """将供应商的结束原因转换为统一枚举。

    这是全项目**唯一**一张 finish_reason 映射表：各 adapter 只调用它，
    不再各自维护一份（两份表必然会漂移，A5 会直接暴露）。

    未知值统一返回 UNKNOWN，原始值必须另外保存在 raw_finish_reason 中。
    """

    if raw is None:
        return FinishReason.UNKNOWN

    known_reasons = {
        # 正常结束
        "stop": FinishReason.STOP,
        "end_turn": FinishReason.STOP,
        "stop_sequence": FinishReason.STOP,
        "pause_turn": FinishReason.STOP,
        "unknown": FinishReason.UNKNOWN,
        # 触顶截断
        "length": FinishReason.LENGTH,
        "max_tokens": FinishReason.LENGTH,
        # 要调工具
        "tool_calls": FinishReason.TOOL_CALLS,
        "tool_use": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        # 被拦截
        "content_filter": FinishReason.CONTENT_FILTER,
        "refusal": FinishReason.CONTENT_FILTER,
        # 中断 / 上游资源不足（DeepSeek），绝不能当成 stop 的成功响应
        "aborted": FinishReason.ABORTED,
        "insufficient_system_resource": FinishReason.UPSTREAM_ERROR,
        "upstream_error": FinishReason.UPSTREAM_ERROR,
    }

    return known_reasons.get(raw, FinishReason.UNKNOWN)

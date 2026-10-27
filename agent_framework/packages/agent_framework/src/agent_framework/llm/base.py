"""LLMClient 抽象与能力声明。

三个 provider + mock 必须同签名，conformance 套件才能参数化（A1–A8）。
签名一旦分裂，测试就只能写三套，归一化的"证据"也就不存在了。
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum

from .types import LLMResponse, Message, StreamEvent, ToolChoice, ToolSpec


class Capability(StrEnum):
    """客户端支持的能力。"""

    STREAM = "stream"
    TOOLS = "tools"
    PARALLEL_TOOLS = "parallel_tools"
    JSON_MODE = "json_mode"
    VISION = "vision"
    PROMPT_CACHING = "prompt_caching"


class LLMClient(ABC):
    """统一的 LLM 客户端抽象接口。

    adapter 只做 wire <-> internal 翻译：不重试、不打日志、不做计量（ADR-003）。
    重试/计量/能力协商/fallback 全部由 client.py 在 Day4 叠加。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """客户端名称，同时作为日志里的 provider 字段。"""

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[Capability]:
        """客户端支持的能力集合（初始化后不可变）。"""

    @property
    def model(self) -> str:
        """当前绑定的模型名（mock 等无模型概念的实现可返回空串）。"""

        return ""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: object,
    ) -> LLMResponse:
        """非流式聊天。"""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: object,
    ) -> AsyncIterator[StreamEvent]:
        """流式聊天。

        事件顺序约定（ADR-002）：所有 text_delta 先于 tool_call，
        且 tool_call 只在参数完整后发出；最后一个事件一定是 message_end。
        """

    async def aclose(self) -> None:
        """释放底层连接。默认无需处理。"""

        return None

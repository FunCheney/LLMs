from enum import StrEnum
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

class Capability(StrEnum):
    """LLM 客户端支持的能力。"""
    STREAM = "stream"
    TOOLS = "tools"
    PARALLEL_TOOLS = "parallel_tools"
    JSON_MODE = "json_mode"
    VISION = "vision"
    PROMPT_CACHING="prompt_caching"


class LLMClient(ABC):
    """统一的 LLM 客户端抽象接口。"""

    # 使用 @property，可以在保持属性访问语法的同时，让子类通过方法动态提供属性值
    @property
    @abstractmethod
    def name(self) -> str:
        """客户端名称。"""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[Capability]:
        """
        客户端支持的能力集合。
            frozenset：因为客户端的能力一般在初始化后就应该保持稳定，是不可变集合，可以减少运行时被意外修改的风险
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """非流式聊天。"""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式聊天，逐步返回文本片段。"""
        ...
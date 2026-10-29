"""conformance 的**唯一**输入真相源。

为什么单独一个文件（Day5 引入）：录制器 `evals/scripts/record_fixtures.py` 不在
pytest 里跑，但它必须用**完全同一份**输入去发真实请求——否则录下来的 response
对不上 A1 断言的 request，夹具一替换就全红。

纪律：这里只有常量与两个构造函数，不 import pytest / httpx（要让 pytest 之外的
脚本也能安全 import）。
"""

from __future__ import annotations

from typing import Any

from agent_framework.llm.types import Message, MessageRole, ToolSpec

SYSTEM_TEXT = "你是研究助手，只回答可验证的事实。"
USER_TEXT = "帮我查一下 MCP 协议的最新进展"

# 录制与回放必须用同一个 max_tokens：它是 request.json 的一部分，A1 精确比对。
MAX_TOKENS = 512

TOOL_NAME = "search_web"
TOOL_DESCRIPTION = "当需要最新外部信息时使用；不要用于纯计算或改写。"
TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "检索关键词"}},
    "required": ["query"],
    "additionalProperties": False,
}


def canonical_messages() -> list[Message]:
    """conformance 的统一内部输入。"""

    return [
        Message(role=MessageRole.SYSTEM, content=SYSTEM_TEXT),
        Message(role=MessageRole.USER, content=USER_TEXT),
    ]


def canonical_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            parameters=TOOL_PARAMETERS,
        )
    ]

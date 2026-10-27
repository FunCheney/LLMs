"""厂商适配器（W1 Day3）。

每个 adapter 只做 wire format <-> internal format 的双向翻译：

- `_common.py`      三家以上都要用、且不含厂商字段名的纯函数（SSE 行解析 / 错误体 / 估算）
- `openai_compat.py`  OpenAI / DeepSeek / Qwen / Ollama / vLLM（差异走 flags，不写 if provider）
- `anthropic.py`      system 顶层参数、tool_result 走 user block、独立流式事件序列
- `mock.py`           读 tests/fixtures 回放 wire，解析复用真 adapter（不是假 LLM）

禁止：重试、日志、缓存、任何业务判断（ADR-003）。
"""

from .anthropic import AnthropicAssembler, AnthropicClient
from .mock import MockLLMClient, ReplayAdapter
from .openai_compat import (
    PROVIDER_FLAGS,
    OpenAICompatAssembler,
    OpenAICompatClient,
    OpenAICompatFlags,
    PreserveReasoning,
    ThinkingStyle,
)

__all__ = [
    "PROVIDER_FLAGS",
    "AnthropicAssembler",
    "AnthropicClient",
    "MockLLMClient",
    "OpenAICompatAssembler",
    "OpenAICompatClient",
    "OpenAICompatFlags",
    "PreserveReasoning",
    "ReplayAdapter",
    "ThinkingStyle",
]

"""LLM 适配层（W1）。

- types.py     内部数据契约：Message / ToolSpec / ToolCall / LLMResponse / StreamEvent / Usage
- errors.py    异常层次 + retryable 语义 + 厂商错误映射（映射只放这里，不放 adapter）
- base.py      LLMClient 抽象 + Capability 枚举
- client.py    装饰器链：trace/日志 -> 计量 -> 重试 -> 能力校验 -> adapter
- registry.py  name -> adapter 工厂（按 models.yaml 的 type 懒加载）
- router.py    task -> tier -> 候选模型列表（含 fallback 链）
- providers/   openai_compat / anthropic / mock

纪律：adapter 内禁止重试、日志、业务判断（ADR-003）。

## 为什么 `ChatCall` / `ChatOutcome` / `LLMGateway` 走 `__getattr__` 而不是直接 import

这三个是全框架最常用的入口，理应从本包浅路径导出（W12 D1 补的缺口：
产品原来只能写 `from agent_framework.llm.client import ChatCall`，穿进实现模块）。

但直接 `from .client import ...` 会造出一个**真环**：

    config.py  ── from .llm.base import Capability
                  └─ 触发 llm/__init__.py
                     └─ from .client import ...
                        └─ client.py: from ..config import Catalog  ← config 正卡在半路

根因不是 import 顺序，而是「`import llm.base` 顺手把整个 client 拉起来了」——
`config` 只需要一个枚举，却背上了 client 的全部依赖。

PEP 562 的模块级 `__getattr__` 正好对症：属性第一次被访问时才加载 client。
于是 `from agent_framework.llm import ChatCall` 可用，而
`from agent_framework.llm.base import Capability` 不会拉起 client。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Capability, LLMClient
from .errors import (
    BadRequestError,
    CapabilityError,
    ContentFilteredError,
    LLMError,
    RateLimitError,
    UpstreamAuthError,
    UpstreamServerError,
    UpstreamTimeoutError,
    error_code_for,
    http_status_for,
    map_provider_error,
    map_status_code,
)
from .types import (
    FinishReason,
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

if TYPE_CHECKING:  # 只为 mypy：运行时由 __getattr__ 提供
    from .client import ChatCall, ChatOutcome, LLMGateway

_LAZY = frozenset({"ChatCall", "ChatOutcome", "LLMGateway"})


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BadRequestError",
    "Capability",
    "CapabilityError",
    "ChatCall",
    "ChatOutcome",
    "ContentFilteredError",
    "FinishReason",
    "LLMClient",
    "LLMError",
    "LLMGateway",
    "LLMResponse",
    "Message",
    "MessageRole",
    "RateLimitError",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
    "ToolChoice",
    "ToolChoiceMode",
    "ToolSpec",
    "UpstreamAuthError",
    "UpstreamServerError",
    "UpstreamTimeoutError",
    "Usage",
    "error_code_for",
    "http_status_for",
    "map_provider_error",
    "map_status_code",
    "normalize_finish_reason",
]

"""LLM 适配层（W1）。

- types.py     内部数据契约：Message / ToolSpec / ToolCall / LLMResponse / StreamEvent / Usage
- errors.py    异常层次 + retryable 语义 + 厂商错误映射（映射只放这里，不放 adapter）
- base.py      LLMClient 抽象 + Capability 枚举
- client.py    装饰器链：trace/日志 -> 计量 -> 重试 -> 能力校验 -> adapter（Day4）
- registry.py  name -> adapter 工厂（按 models.yaml 的 type 懒加载，Day4）
- router.py    task -> tier -> 候选模型列表（含 fallback 链，Day4）
- providers/   openai_compat / anthropic / mock

纪律：adapter 内禁止重试、日志、业务判断（ADR-003）。
"""

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

__all__ = [
    "BadRequestError",
    "Capability",
    "CapabilityError",
    "ContentFilteredError",
    "FinishReason",
    "LLMClient",
    "LLMError",
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
    "map_provider_error",
    "map_status_code",
    "normalize_finish_reason",
]

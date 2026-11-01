"""入口层过载与并发控制（W2 D2）。

本包只做"**在调 gateway 之前**把过载挡住"这一件事。它不重试、不 fallback、不改写请求：
那些是 `LLMGateway` 的唯一职责（ADR-003）。两者的边界就一句话——
**信号量拿不到就立刻 503，绝不让 LLMGateway 看见这次调用**。

`ServerBusyError` 的类定义在 `llm/errors.py`（映射表在那里，反向 import 会成环），
这里 re-export 是为了给"resilience 的错误归 resilience"一个稳定的导入路径。
"""

from .concurrency import ConcurrencyLimiter
from .errors import ServerBusyError

__all__ = ["ConcurrencyLimiter", "ServerBusyError"]

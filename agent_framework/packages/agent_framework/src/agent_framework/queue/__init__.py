"""长任务层（W2 D2）。

边界：**只回答"任务到哪一步了"**，不做编排。真正的分步重试与续跑归 W9 的
Temporal workflow（`docs/w2/adr-w2-session-boundary.md` 有决策记录）。

`/v1/chat` 永远不进这里：首 token 延迟是它的生命线，套一层队列等于把 p50 从
800ms 变成"排队时间 + 800ms"。
"""

from .base import TaskHandler, TaskQueue, TaskRecord, TaskStatus
from .inmemory import InMemoryTaskQueue

__all__ = [
    "InMemoryTaskQueue",
    "TaskHandler",
    "TaskQueue",
    "TaskRecord",
    "TaskStatus",
]

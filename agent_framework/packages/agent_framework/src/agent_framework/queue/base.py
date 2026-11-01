"""长任务队列：契约与状态机（W2 D2）。

为什么要队列
-----------
"深度调研"这类任务要跑几分钟，塞进 HTTP 请求里必然超时（而且客户端一断开，
协程就被取消，工作全白做）。所以入口只做三件事：**校验 → 入队 → 返回 task_id**。

状态机只有四条边，多一条都是过度设计：

    queued ──▶ running ──▶ succeeded
                    └────▶ failed

`attempts` 字段现在恒为 1：真正的分步重试归 W9 的 workflow（Temporal）。
本周先把"提交/查询/失败可见"这条最小链路跑通——**能看见失败**比"能自动重试"更基础。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ..llm.client import ChatCall

__all__ = [
    "TaskHandler",
    "TaskQueue",
    "TaskRecord",
    "TaskStatus",
]


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# 消费者只依赖"给我一个 ChatCall，回一个可 JSON 化的 dict"。
# 不直接依赖 ChatOutcome / ChatResponse：队列不该知道 HTTP 投影长什么样，
# 否则 W9 换成 workflow 时要把序列化逻辑再抄一遍。
TaskHandler = Callable[[ChatCall], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class TaskRecord:
    """一个任务的可见状态。字段都是"查询接口要吐出去的东西"。"""

    task_id: str
    status: TaskStatus
    request_id: str
    trace_id: str
    created_at: float

    started_at: float | None = None
    finished_at: float | None = None
    attempts: int = 1

    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempts": self.attempts,
            "result": self.result,
            "error": self.error,
        }


class TaskQueue(Protocol):
    async def enqueue(
        self,
        call: ChatCall,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> TaskRecord: ...

    async def get(self, task_id: str) -> TaskRecord: ...

    async def aclose(self) -> None: ...

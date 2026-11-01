"""进程内任务队列（离线可测；生产可换 arq/Celery，契约不变）。

三个刻意的选择
-------------
1. **消费者是 asyncio task，不是后台线程**：`LLMGateway` 是 async 的，线程池里调
   只会把事件循环的并发能力浪费掉。
2. **入队立即返回，不在 HTTP 请求里跑任务**：这正是队列存在的意义。
   在请求里 `await handler(call)` 等于没做队列，只是多了一层壳。
3. **队列满了抛 `ServerBusyError`（503 + Retry-After）**：不阻塞、不丢任务、
   不无限堆积。无限堆积的结局是内存涨到 OOM，而调用方还以为任务排上了。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from ..llm.client import ChatCall
from ..llm.errors import LLMError, ServerBusyError, TaskNotFoundError, error_code_for
from ..observability import request_scope, trace_scope
from .base import TaskHandler, TaskRecord, TaskStatus

__all__ = ["InMemoryTaskQueue"]

_LOG = logging.getLogger("agent_framework.queue")


class InMemoryTaskQueue:
    """单进程队列。多副本部署时每个副本各跑各的消费者，任务不跨实例共享。"""

    def __init__(
        self,
        handler: TaskHandler,
        *,
        max_size: int = 100,
        workers: int = 2,
        clock: Callable[[], float] = time.time,
        retention: int = 1000,
    ) -> None:
        if workers < 1:
            raise ValueError("workers 必须 >= 1")
        if max_size < 1:
            raise ValueError("max_size 必须 >= 1")

        self._handler = handler
        self._max_size = max_size
        self._workers = workers
        self._clock = clock
        self._retention = retention

        self._pending: asyncio.Queue[str] = asyncio.Queue(maxsize=max_size)
        self._records: dict[str, TaskRecord] = {}
        self._calls: dict[str, ChatCall] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._start_guard = asyncio.Lock()
        self._closed = False

    # -- 生命周期 -------------------------------------------------------- #

    async def start(self) -> None:
        """懒启动：`create_app()` 不该在 import 时就创建后台 task。

        这也让测试可以"只入队不消费"地断言 queued 状态。
        """

        async with self._start_guard:
            if self._tasks or self._closed:
                return

            self._tasks = [
                asyncio.create_task(self._worker(index), name=f"task-worker-{index}")
                for index in range(self._workers)
            ]

    async def aclose(self) -> None:
        self._closed = True

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()

    # -- 入队 / 查询 ----------------------------------------------------- #

    async def enqueue(
        self,
        call: ChatCall,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> TaskRecord:
        if self._closed:
            raise ServerBusyError("任务队列已关闭", retry_after=5.0)

        await self.start()

        record = TaskRecord(
            task_id="task_" + uuid4().hex[:12],
            status=TaskStatus.QUEUED,
            request_id=request_id or "-",
            trace_id=trace_id or "-",
            created_at=self._clock(),
        )

        # 先登记再入队：反过来的话，跑得快的消费者可能在 `_records` 里还查不到
        # 这个 id，于是任务被静默丢掉（记录永远停在 queued，最难查的一类 bug）。
        self._records[record.task_id] = record
        self._calls[record.task_id] = call

        try:
            self._pending.put_nowait(record.task_id)
        except asyncio.QueueFull as exc:
            self._records.pop(record.task_id, None)
            self._calls.pop(record.task_id, None)
            # 满就是满：给调用方一个可退避的信号，而不是让他等 30 秒再超时。
            raise ServerBusyError(
                f"任务队列已满（上限 {self._max_size}）；请按 Retry-After 重投",
                retry_after=3.0,
            ) from exc

        self._prune()

        _LOG.info(
            "task enqueued task_id=%s request_id=%s trace_id=%s pending=%d",
            record.task_id,
            record.request_id,
            record.trace_id,
            self._pending.qsize(),
        )

        return record

    async def get(self, task_id: str) -> TaskRecord:
        record = self._records.get(task_id)

        if record is None:
            raise TaskNotFoundError(
                f"未知 task_id '{task_id}'：可能已过期或不属于当前实例"
            )

        return record

    # -- 消费者 ---------------------------------------------------------- #

    async def _worker(self, index: int) -> None:
        while True:
            task_id = await self._pending.get()

            try:
                await self._run_one(task_id)
            finally:
                self._pending.task_done()

    async def _run_one(self, task_id: str) -> None:
        record = self._records.get(task_id)
        call = self._calls.get(task_id)

        if record is None or call is None:
            _LOG.warning("task vanished task_id=%s", task_id)
            return

        # 消费者不在任何 HTTP 上下文里：不显式进 scope，任务内部的 LLM 调用
        # 会产出空 trace_id / 空 request_id，trace 从这里断掉。
        with request_scope(record.request_id), trace_scope(record.trace_id):
            record.status = TaskStatus.RUNNING
            record.started_at = self._clock()
            _LOG.info("task running task_id=%s", task_id)

            try:
                record.result = await self._handler(call)
                record.status = TaskStatus.SUCCEEDED
                _LOG.info("task succeeded task_id=%s", task_id)
            except LLMError as exc:
                record.status = TaskStatus.FAILED
                record.error = {
                    "code": error_code_for(exc),
                    "message": str(exc),
                    "provider": exc.provider,
                }
                # 失败不是异常：它是这个任务的结果。日志用 error 级别，但不 raise。
                _LOG.warning(
                    "task failed task_id=%s code=%s message=%s",
                    task_id,
                    record.error["code"],
                    exc,
                )
            except Exception as exc:  # noqa: BLE001 - 兜底：任务崩了也要落状态
                record.status = TaskStatus.FAILED
                record.error = {"code": "INTERNAL", "message": str(exc)}
                _LOG.exception("task crashed task_id=%s", task_id)
            finally:
                record.finished_at = self._clock()
                self._calls.pop(task_id, None)

    def _prune(self) -> None:
        """只保留最近 N 条已完成记录。

        不裁剪的话，一个跑一周的实例会把每个任务的结果都留在内存里——
        压测时看不出来，线上表现为"内存缓慢上涨"。
        """

        if len(self._records) <= self._retention:
            return

        finished = [
            task_id
            for task_id, record in self._records.items()
            if record.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED)
        ]

        for task_id in finished[: len(self._records) - self._retention]:
            self._records.pop(task_id, None)

    # -- 便于测试/运维 --------------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending": self._pending.qsize(),
            "records": len(self._records),
            "workers": len(self._tasks),
            "closed": self._closed,
        }

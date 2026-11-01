"""trace 的最小骨架（W2 D3）：`Span` + `span()` + `TraceExporter` 协议。

为什么不上 OpenTelemetry
-----------------------
本周只需要回答一个问题：**一条请求能不能追到它内部每一次 LLM 调用**。
OTel 带来的是 sampler / batch processor / 跨进程 context propagation 那一整套，
W3 接 Langfuse 之前都属于"为还没出现的问题付复杂度"。

所以这里只做三件克制的事：

1. 字段先定死（`trace_id / span_id / parent_span_id / latency_ms / attributes`），
   W3 换 `LangfuseExporter` 时业务代码一行都不用动；
2. exporter 用协议 + 模块级单例，默认 `NoopExporter` —— **库不擅自连外部服务**；
3. `span()` 用 `try/finally` 保证异常路径也 export：失败调用恰恰是最需要 trace 的那批，
   静默丢掉等于把"为什么这次 502"永远埋在日志里。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from .observability import (
    current_span_id,
    current_trace_id,
    new_trace_id,
    span_id_var,
    trace_id_var,
)

__all__ = [
    "InMemoryExporter",
    "NoopExporter",
    "Span",
    "TraceExporter",
    "configure_exporter",
    "current_exporter",
    "new_span_id",
    "span",
]

_LOG = logging.getLogger("agent_framework.trace")


def new_span_id() -> str:
    return "span_" + uuid4().hex[:12]


@dataclass(slots=True)
class Span:
    """一个工作单元。字段刻意少：属性进 `attributes`，不在这里加列。"""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str

    # 墙钟时间（epoch 秒）用于和日志对齐；`latency_ms` 用 monotonic 算，不受校时影响。
    started_at: float
    latency_ms: float = 0.0

    # ok / error。失败 span 必须留下来，否则"失败率高但 trace 里全是成功"。
    status: str = "ok"
    error_code: str | None = None

    attributes: dict[str, Any] = field(default_factory=dict)


class TraceExporter(Protocol):
    """出口。W3 的 LangfuseExporter 实现同一个方法即可。"""

    async def export(self, spans: list[Span]) -> None: ...


class NoopExporter:
    """默认出口：什么都不做。

    这不是"没实现"，是刻意的默认值：`import agent_framework` 不该产生网络连接，
    也不该在 pytest 里往某个 collector 打数据包。
    """

    async def export(self, spans: list[Span]) -> None:
        return None


class InMemoryExporter:
    """测试与本地排查用：收进列表，进程退出即丢。"""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    async def export(self, spans: list[Span]) -> None:
        self.spans.extend(spans)

    def by_name(self, name: str) -> list[Span]:
        return [item for item in self.spans if item.name == name]


_exporter: TraceExporter = NoopExporter()


def configure_exporter(exporter: TraceExporter | None = None) -> TraceExporter:
    """换出口。返回上一个，方便测试里用完还原（避免用例之间互相污染）。"""

    global _exporter

    previous = _exporter
    _exporter = exporter or NoopExporter()

    return previous


def current_exporter() -> TraceExporter:
    return _exporter


async def _safe_export(spans: list[Span]) -> None:
    """export 失败绝不影响主流程。

    trace 是观测手段，不是业务依赖：收集端挂了就把线上请求打成 500，
    比没有 trace 糟糕得多。
    """

    try:
        await _exporter.export(spans)
    except Exception:  # noqa: BLE001 - 故意兜住所有收集端异常
        _LOG.warning("trace export failed spans=%d", len(spans), exc_info=True)


@asynccontextmanager
async def span(name: str, **attributes: Any) -> AsyncIterator[Span]:
    """开一个 span，退出时算延迟并交给 exporter。

    - `parent_span_id` 取自 contextvar，所以嵌套调用自动成树；
    - 没有 trace_id 时**兜底新建**：后台任务/脚本不经过中间件，
      不兜底就会产出 `trace_id=""` 的孤儿 span；
    - `None` 的属性直接丢掉，避免看板里出现一堆 `null` 列。
    """

    trace_token = None
    trace_id = current_trace_id()

    if not trace_id:
        trace_id = new_trace_id()
        trace_token = trace_id_var.set(trace_id)

    item = Span(
        trace_id=trace_id,
        span_id=new_span_id(),
        parent_span_id=current_span_id() or None,
        name=name,
        started_at=time.time(),
        attributes={
            key: value for key, value in attributes.items() if value is not None
        },
    )

    span_token = span_id_var.set(item.span_id)
    started = time.perf_counter()

    try:
        yield item
    except BaseException as exc:
        # 记下异常类型就够：完整栈在日志里，span 里塞栈会让 trace 体积失控。
        item.status = "error"
        item.attributes.setdefault("error", type(exc).__name__)
        raise
    finally:
        item.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        span_id_var.reset(span_token)

        if trace_token is not None:
            trace_id_var.reset(trace_token)

        await _safe_export([item])

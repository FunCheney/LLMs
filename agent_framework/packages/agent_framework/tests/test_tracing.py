"""trace 贯穿测试（W2 D3）。

四件事必须有证据：
1. 成功/失败都 export（失败静默丢 = trace 里只剩成功，根因归因失效）；
2. 嵌套 span 共享 trace、parent 关系正确；
3. contextvar 在并发请求之间不串；
4. `RequestLog` 自动带上 trace_id/span_id（读日志的人不用猜）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from agent_framework.observability import (
    RequestLog,
    current_span_id,
    current_trace_id,
    emit,
    get_request_logger,
    trace_scope,
)
from agent_framework.tracing import (
    InMemoryExporter,
    Span,
    configure_exporter,
    span,
)


@pytest.fixture
def exporter() -> Iterator[InMemoryExporter]:
    target = InMemoryExporter()
    previous = configure_exporter(target)

    try:
        yield target
    finally:
        configure_exporter(previous)


async def test_span_exports_on_success(exporter: InMemoryExporter) -> None:
    with trace_scope("trace_ok"):
        async with span("llm.call", alias="qwen3.7-plus", attempt=1) as item:
            item.attributes["first_token_ms"] = 123.4

    assert len(exporter.spans) == 1

    recorded = exporter.spans[0]
    assert recorded.trace_id == "trace_ok"
    assert recorded.name == "llm.call"
    assert recorded.status == "ok"
    assert recorded.attributes["alias"] == "qwen3.7-plus"
    assert recorded.attributes["first_token_ms"] == 123.4
    assert recorded.latency_ms >= 0


async def test_span_exports_on_failure(exporter: InMemoryExporter) -> None:
    with pytest.raises(RuntimeError), trace_scope("trace_err"):
        async with span("llm.call", alias="qwen3.7-plus"):
            raise RuntimeError("upstream boom")

    assert len(exporter.spans) == 1

    recorded = exporter.spans[0]
    assert recorded.status == "error"
    assert recorded.attributes["error"] == "RuntimeError"


async def test_nested_spans_share_trace_and_parent(exporter: InMemoryExporter) -> None:
    with trace_scope("trace_tree"):
        async with span("request") as outer, span("llm.call") as inner:
            assert inner.trace_id == outer.trace_id
            assert inner.parent_span_id == outer.span_id

    assert [item.name for item in exporter.spans] == ["llm.call", "request"]
    assert exporter.spans[1].parent_span_id is None


async def test_span_creates_trace_when_missing(exporter: InMemoryExporter) -> None:
    """后台任务/脚本不经过中间件：不兜底就会产出 trace_id="" 的孤儿 span。"""

    async with span("background.job"):
        pass

    assert exporter.spans[0].trace_id.startswith("trace_")


async def test_contextvars_are_isolated_between_tasks(
    exporter: InMemoryExporter,
) -> None:
    async def work(name: str) -> str:
        with trace_scope(f"trace_{name}"):
            await asyncio.sleep(0.005)
            async with span("llm.call"):
                await asyncio.sleep(0)
                seen = current_trace_id()

            return seen

    results = await asyncio.gather(*(work(name) for name in ("a", "b", "c")))

    assert results == ["trace_a", "trace_b", "trace_c"]
    assert {item.trace_id for item in exporter.spans} == {
        "trace_a",
        "trace_b",
        "trace_c",
    }


async def test_span_ids_are_restored_after_exit() -> None:
    assert current_span_id() == ""

    async with span("outer"):
        inner_id = current_span_id()
        assert inner_id.startswith("span_")

    assert current_span_id() == ""


def test_emit_fills_trace_and_span_ids(
    caplog: pytest.LogCaptureFixture, exporter: InMemoryExporter
) -> None:
    caplog.set_level(logging.INFO, logger="agent_framework.request")

    with trace_scope("trace_log"):
        emit(RequestLog(task="chat", model="qwen3.7-plus"))

    payload = _last_payload(caplog)

    assert payload["trace_id"] == "trace_log"
    # 没有 span 时用 "-" 占位，而不是空串：看板按字段聚合时空串会被算成另一类。
    assert payload["span_id"] == "-"


async def test_emit_uses_current_span(
    caplog: pytest.LogCaptureFixture, exporter: InMemoryExporter
) -> None:
    caplog.set_level(logging.INFO, logger="agent_framework.request")

    with trace_scope("trace_span"):
        async with span("llm.call") as item:
            emit(RequestLog(task="chat"))

            assert _last_payload(caplog)["span_id"] == item.span_id


async def test_exporter_failure_does_not_break_caller() -> None:
    class Broken:
        async def export(self, spans: list[Span]) -> None:
            raise RuntimeError("collector down")

    previous = configure_exporter(Broken())

    try:
        async with span("llm.call"):
            ok = True

        assert ok
    finally:
        configure_exporter(previous)


def _last_payload(caplog: pytest.LogCaptureFixture) -> dict[str, Any]:
    records = [record for record in caplog.records if record.name == "agent_framework.request"]
    assert records, "没有 RequestLog 输出"

    return json.loads(records[-1].getMessage())


def test_request_logger_is_the_same_channel_used_by_readers() -> None:
    """读取方（基线表 / smoke runner）必须挂同一个 logger，否则会另造一套口径。"""

    assert get_request_logger().name == "agent_framework.request"

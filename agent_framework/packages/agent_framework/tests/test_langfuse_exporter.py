"""Langfuse exporter 的测试（W3 D4 / Gate 6「trace 落地 + 缺 key 自动降级」）。

三组断言对应三条纪律：

1. **缺 key 降级**：没有账号的机器必须能跑完整套测试与评测；
2. **隐私边界**：prompt 全文绝不上报，只报长度与哈希；
3. **失败不冒泡**：观测后端挂了，业务请求照常（连续失败后自禁用）。

全程用 `httpx.MockTransport`，不打真实网络。
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from agent_framework.config import Settings
from agent_framework.exporters.langfuse import LangfuseExporter, build_exporter
from agent_framework.tracing import NoopExporter, Span, configure_exporter, span
from pydantic import SecretStr


def _span(
    *,
    trace_id: str = "trace_1",
    span_id: str = "span_1",
    parent: str | None = None,
    name: str = "llm.call",
    attributes: dict[str, Any] | None = None,
    status: str = "ok",
) -> Span:
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        started_at=1_700_000_000.0,
        latency_ms=12.5,
        status=status,
        attributes=attributes or {},
    )


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class _Recorder:
    """收集 exporter 发出去的请求体。"""

    def __init__(self, *, status: int = 207, fail: bool = False) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []
        self._status = status
        self._fail = fail

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)

            if self._fail:
                raise httpx.ConnectError("boom", request=request)

            self.bodies.append(json.loads(request.content))

            return httpx.Response(self._status, json={"successes": []})

        return httpx.MockTransport(handler)


def _exporter(recorder: _Recorder, **kwargs: Any) -> LangfuseExporter:
    return LangfuseExporter(
        host="https://langfuse.test",
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        transport=recorder.transport(),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 降级
# --------------------------------------------------------------------------- #


def test_build_exporter_without_keys_returns_noop() -> None:
    exporter = build_exporter(_settings())

    assert isinstance(exporter, NoopExporter)


def test_build_exporter_with_partial_keys_returns_noop() -> None:
    """只配了 public key（常见的手滑）也要降级，不能带着空密钥去请求。"""

    exporter = build_exporter(
        _settings(langfuse_public_key=SecretStr("pk-lf-test"))
    )

    assert isinstance(exporter, NoopExporter)


def test_build_exporter_with_keys_returns_langfuse() -> None:
    exporter = build_exporter(
        _settings(
            langfuse_public_key=SecretStr("pk-lf-test"),
            langfuse_secret_key=SecretStr("sk-lf-test"),
            langfuse_host="https://langfuse.test",
        )
    )

    assert isinstance(exporter, LangfuseExporter)


async def test_export_failure_does_not_raise() -> None:
    """观测后端挂了，业务请求必须照常。"""

    recorder = _Recorder(fail=True)
    exporter = _exporter(recorder)

    await exporter.export([_span()])  # 不抛即通过


async def test_exporter_self_disables_after_repeated_failures() -> None:
    """连续失败 3 次后自禁用：否则每个请求都等满超时，比丢 trace 更糟。"""

    recorder = _Recorder(fail=True)
    exporter = _exporter(recorder)

    for _ in range(3):
        await exporter.export([_span()])

    assert exporter.disabled is True
    assert len(recorder.requests) == 3

    await exporter.export([_span()])
    assert len(recorder.requests) == 3  # 已禁用，不再发请求


async def test_http_error_status_triggers_degradation() -> None:
    recorder = _Recorder(status=500)
    exporter = _exporter(recorder)

    await exporter.export([_span()])

    assert exporter.disabled is False
    assert len(recorder.requests) == 1


# --------------------------------------------------------------------------- #
# 载荷
# --------------------------------------------------------------------------- #


async def test_payload_groups_spans_by_trace() -> None:
    recorder = _Recorder()
    exporter = _exporter(recorder)

    await exporter.export(
        [
            _span(trace_id="trace_a", span_id="span_1", name="request"),
            _span(trace_id="trace_a", span_id="span_2", parent="span_1", name="llm.call"),
            _span(trace_id="trace_b", span_id="span_3", name="request"),
        ]
    )

    batch = recorder.bodies[0]["batch"]
    traces = [item for item in batch if item["type"] == "trace-create"]
    spans = [item for item in batch if item["type"] == "span-create"]

    assert {item["body"]["id"] for item in traces} == {"trace_a", "trace_b"}
    assert len(spans) == 3


async def test_span_tree_keeps_parent_link() -> None:
    """`parent_span_id` 必须原样映射到 `parentObservationId`，否则树上全是孤儿。"""

    recorder = _Recorder()
    exporter = _exporter(recorder)

    await exporter.export(
        [
            _span(trace_id="trace_a", span_id="span_1", name="request"),
            _span(trace_id="trace_a", span_id="span_2", parent="span_1", name="llm.call"),
        ]
    )

    spans = {
        item["body"]["id"]: item["body"]
        for item in recorder.bodies[0]["batch"]
        if item["type"] == "span-create"
    }

    assert spans["span_1"]["parentObservationId"] is None
    assert spans["span_2"]["parentObservationId"] == "span_1"


async def test_error_span_is_exported_as_error_level() -> None:
    """失败 span 必须留在 trace 里——W2 的 `finally export` 就是为这条。"""

    recorder = _Recorder()
    exporter = _exporter(recorder)

    await exporter.export([_span(status="error", name="llm.call")])

    span_body = next(
        item["body"] for item in recorder.bodies[0]["batch"] if item["type"] == "span-create"
    )
    assert span_body["level"] == "ERROR"


async def test_authorization_header_is_basic_auth() -> None:
    recorder = _Recorder()
    exporter = _exporter(recorder)

    await exporter.export([_span()])

    header = recorder.requests[0].headers["authorization"]
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
    assert decoded == "pk-lf-test:sk-lf-test"


# --------------------------------------------------------------------------- #
# 隐私边界
# --------------------------------------------------------------------------- #


async def test_prompt_content_is_hashed_not_uploaded() -> None:
    """默认只报长度与哈希。prompt 里可能有用户数据，观测平台是第三方。"""

    secret = "用户的身份证号 110101199001011234"
    recorder = _Recorder()
    exporter = _exporter(recorder)

    await exporter.export([_span(attributes={"prompt_content": secret})])

    metadata = next(
        item["body"]["metadata"]
        for item in recorder.bodies[0]["batch"]
        if item["type"] == "span-create"
    )
    payload = json.dumps(metadata, ensure_ascii=False)

    assert secret not in payload
    assert metadata["prompt_content"]["len"] == len(secret)
    assert "sha256" in metadata["prompt_content"]


async def test_non_content_attributes_pass_through() -> None:
    """model / tokens / cost 这些正是要看的东西，不能被一起哈希掉。"""

    recorder = _Recorder()
    exporter = _exporter(recorder)

    await exporter.export(
        [_span(attributes={"model": "qwen3.7-plus", "prompt_tokens": 58})]
    )

    metadata = next(
        item["body"]["metadata"]
        for item in recorder.bodies[0]["batch"]
        if item["type"] == "span-create"
    )

    assert metadata["model"] == "qwen3.7-plus"
    assert metadata["prompt_tokens"] == 58


async def test_long_strings_are_fingerprinted() -> None:
    """有人把整段 prompt 塞进自定义 attribute 时，长度兜底要拦住。"""

    long_text = "x" * 300
    recorder = _Recorder()
    exporter = _exporter(recorder)

    await exporter.export([_span(attributes={"note": long_text})])

    metadata = next(
        item["body"]["metadata"]
        for item in recorder.bodies[0]["batch"]
        if item["type"] == "span-create"
    )

    assert metadata["note"]["len"] == 300


async def test_capture_content_opt_in_keeps_raw_text() -> None:
    """显式打开才上报全文（W10 会核查这个开关的使用）。"""

    recorder = _Recorder()
    exporter = _exporter(recorder, capture_content=True)

    await exporter.export([_span(attributes={"prompt_content": "hello"})])

    metadata = next(
        item["body"]["metadata"]
        for item in recorder.bodies[0]["batch"]
        if item["type"] == "span-create"
    )

    assert metadata["prompt_content"] == "hello"


# --------------------------------------------------------------------------- #
# 与 W2 协议的对接
# --------------------------------------------------------------------------- #


async def test_span_context_exports_through_configured_exporter() -> None:
    """换实现不改协议：`span()` 里一行都没动，export 仍然收到 span。"""

    recorder = _Recorder()
    exporter = _exporter(recorder)
    previous = configure_exporter(exporter)

    try:
        async with span("llm.call", model="qwen3.7-plus") as item:
            trace_id = item.trace_id
    finally:
        configure_exporter(previous if isinstance(previous, LangfuseExporter) else None)

    assert recorder.bodies, "span() 没有把 span 交给 exporter"
    bodies = recorder.bodies[0]["batch"]
    trace_ids = [item["body"].get("id") for item in bodies if item["type"] == "trace-create"]

    assert trace_id in trace_ids


@pytest.mark.parametrize("bad_status", [400, 401, 429])
async def test_http_4xx_also_degrades_instead_of_raising(bad_status: int) -> None:
    recorder = _Recorder(status=bad_status)
    exporter = _exporter(recorder)

    await exporter.export([_span()])

    assert exporter.disabled is False

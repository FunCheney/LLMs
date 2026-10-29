"""可观测层测试：成本、打码、request_id 作用域。

这一层的 bug 特征是**不报错**：成本算成 0、request_id 串号、日志里出现 key 原文。
所以每条都要有断言守着，而不是"读一遍觉得对"。
"""

from __future__ import annotations

import json
import logging

import pytest
from agent_framework.config import Pricing
from agent_framework.llm.types import Usage
from agent_framework.observability import (
    RequestLog,
    compute_cost,
    configure_logging,
    current_request_id,
    emit,
    mask_secret,
    new_request_id,
    request_scope,
)

CNY_PRICING = Pricing(input=2.0, output=8.0, currency="CNY")


def test_cost_splits_input_and_output_price() -> None:
    usage = Usage(input_tokens=1000, output_tokens=500)

    # (1000 * 2 + 500 * 8) / 1e6
    assert compute_cost(CNY_PRICING, usage) == pytest.approx(0.006)


def test_estimated_usage_is_still_costed() -> None:
    """厂商没给 usage 时的估算值必须照算，否则成本恒为 0，熔断器失效。"""

    usage = Usage(input_tokens=1000, output_tokens=500, estimated=True)

    assert compute_cost(CNY_PRICING, usage) == pytest.approx(0.006)


def test_tiny_cost_is_not_rounded_to_zero() -> None:
    cheap = Pricing(input=0.15, output=0.6, currency="USD")
    cost = compute_cost(cheap, Usage(input_tokens=1, output_tokens=1))

    assert cost > 0
    assert cost == 1e-06


def test_zero_usage_costs_zero() -> None:
    assert compute_cost(CNY_PRICING, Usage()) == 0.0


def test_costing_is_linear_in_tokens() -> None:
    single = compute_cost(CNY_PRICING, Usage(input_tokens=100, output_tokens=100))
    double = compute_cost(CNY_PRICING, Usage(input_tokens=200, output_tokens=200))

    assert double == pytest.approx(single * 2)


def test_mask_secret_keeps_prefix_and_tail() -> None:
    masked = mask_secret("sk-abcdefghijklmnop")

    assert masked == "sk-***mnop"
    assert "abcdefghij" not in masked


def test_mask_secret_handles_short_and_empty() -> None:
    assert mask_secret(None) == "***"
    assert mask_secret("") == "***"
    assert mask_secret("sk-123") == "***"


def test_request_scope_sets_and_resets() -> None:
    assert current_request_id() == ""

    with request_scope("req_fixed") as resolved:
        assert resolved == "req_fixed"
        assert current_request_id() == "req_fixed"

    # 必须复位：ASGI 的 task 会复用，漏复位会串号。
    assert current_request_id() == ""


def test_request_scope_generates_id_when_absent() -> None:
    with request_scope() as resolved:
        assert resolved.startswith("req_")
        assert len(resolved) == len(new_request_id())


def test_request_log_defaults_missing_id_to_dash() -> None:
    assert RequestLog().as_dict()["request_id"] == "-"


def test_emit_writes_single_json_line(caplog: logging.LogRecord) -> None:
    record = RequestLog(request_id="req_x", model="m", cost=0.001, currency="CNY")

    with caplog.at_level(logging.INFO, logger="agent_framework.request"):
        emit(record)

    lines = [item.getMessage() for item in caplog.records]
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["request_id"] == "req_x"
    assert payload["cost"] == 0.001
    assert "\n" not in lines[0]


def test_configure_logging_is_idempotent() -> None:
    """重复调用不能重复挂 handler：否则一条请求日志会被打 N 遍。

    `create_app()` 每次导入都会调一次（含测试里多次建 app），所以幂等是硬要求。
    """

    logger = logging.getLogger("agent_framework")
    before = list(logger.handlers)

    try:
        configure_logging()
        configure_logging()

        owned = [h for h in logger.handlers if getattr(h, "_agent_framework_owned", False)]

        assert len(owned) <= 1
        assert logger.level == logging.INFO
    finally:
        for handler in logger.handlers:
            if handler not in before:
                logger.removeHandler(handler)

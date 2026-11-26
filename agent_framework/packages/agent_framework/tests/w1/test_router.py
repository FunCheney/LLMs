"""路由层测试：只回答"候选是谁"，不回答"要不要 fallback"。

未知 task 的静默回落必须留 warning：否则你以为评测跑的是 strong，
实际跑的是 cheap，而指标曲线看不出任何区别。
"""

from __future__ import annotations

import logging

from agent_framework.llm.router import Router, StaticRouter, TaskRouter
from gateway_helpers import make_catalog


def test_task_router_returns_primary_then_fallback() -> None:
    router = TaskRouter(make_catalog(["primary", "backup"]))

    assert router.route("chat") == ["primary", "backup"]


def test_duplicate_alias_is_deduped_and_order_kept() -> None:
    catalog = make_catalog(["primary", "backup"])
    catalog.routes["cheap"].fallback.append("primary")
    catalog.routes["default"].fallback.append("primary")

    assert TaskRouter(catalog).route("chat") == ["primary", "backup"]


def test_unknown_task_falls_back_to_default_and_warns(
    caplog: logging.LogRecord,
) -> None:
    router = TaskRouter(make_catalog(["primary"]))

    with caplog.at_level(logging.WARNING, logger="agent_framework.router"):
        aliases = router.route("no-such-task")

    assert aliases == ["primary"]
    assert any("falling back to 'default'" in record.getMessage() for record in caplog.records)


def test_known_task_does_not_warn(caplog: logging.LogRecord) -> None:
    router = TaskRouter(make_catalog(["primary"]))

    with caplog.at_level(logging.WARNING, logger="agent_framework.router"):
        router.route("chat")

    assert caplog.records == []


def test_static_router_ignores_task_and_dedupes() -> None:
    router = StaticRouter(["x", "x", "y"])

    assert router.route("anything") == ["x", "y"]
    assert router.route("chat") == ["x", "y"]


def test_static_router_name_is_configurable() -> None:
    assert StaticRouter(["x"]).name == "static"
    assert StaticRouter(["x"], name="debug").name == "debug"


def test_both_routers_satisfy_protocol() -> None:
    """抽象要有第二实现才算证明自己值得存在。"""

    assert isinstance(TaskRouter(make_catalog(["a"])), Router)
    assert isinstance(StaticRouter(["a"]), Router)

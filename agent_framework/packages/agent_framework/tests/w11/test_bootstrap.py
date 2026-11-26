"""bootstrap 接线（W11 D5）。

开关只在**一个地方**生效：`build_gateway`。这组测试确认"关着 = 旧行为"
不是一句注释——关着的时候 router 是 `TaskRouter`、cache 是 `None`。
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from agent_framework.bootstrap import build_gateway
from agent_framework.config import (
    Catalog,
    ModelConfig,
    Pricing,
    ProviderConfig,
    RouteConfig,
)
from agent_framework.llm.client import ChatCall
from agent_framework.llm.difficulty_router import DifficultyRouter
from agent_framework.llm.router import TaskRouter
from agent_framework.llm.types import Message, MessageRole, ToolSpec
from agent_framework.resilience.semantic_cache import SemanticCache
from gateway_helpers import make_catalog, make_settings


def two_tier_catalog() -> Catalog:
    provider = ProviderConfig(type="openai_compat", base_url="https://example.test/v1")
    pricing = Pricing(input=1.0, output=2.0, currency="CNY")

    def model(tier: str) -> ModelConfig:
        return ModelConfig(provider="p", model=f"{tier}-wire", tier=tier, pricing=pricing)

    return Catalog(
        providers={"p": provider},
        models={"cheap-a": model("cheap"), "strong-a": model("strong")},
        routes={
            "default": RouteConfig(primary="cheap-a", fallback=[]),
            "cheap": RouteConfig(primary="cheap-a", fallback=[]),
            "strong": RouteConfig(primary="strong-a", fallback=[]),
        },
        tasks={"chat": "cheap"},
    )


def build(**overrides: Any) -> Any:
    return build_gateway(make_settings(**overrides), catalog=two_tier_catalog())


# --------------------------------------------------------------------------- #
# 默认 = 旧行为
# --------------------------------------------------------------------------- #


def test_defaults_leave_every_behaviour_at_w10() -> None:
    gateway = build()

    assert isinstance(gateway._router, TaskRouter)
    assert gateway._cache is None
    assert gateway._breaker is not None  # 熔断不是 flag，由 BREAKER_ENABLED 管


def test_breaker_can_still_be_switched_off_on_its_own() -> None:
    gateway = build(breaker_enabled=False)

    assert gateway._breaker is None


# --------------------------------------------------------------------------- #
# 打开 = 换一条路径，其他都不动
# --------------------------------------------------------------------------- #


def test_semantic_cache_flag_wires_a_versioned_cache() -> None:
    gateway = build(features="semantic_cache=on")

    assert isinstance(gateway._cache, SemanticCache)
    # 版本必须来自启动那一刻的资产快照，否则重启后仍在回放上一套 prompt 的答案
    assert gateway._cache.asset_digest


def test_difficulty_router_flag_picks_aliases_from_models_yaml() -> None:
    """alias 从 `routes.cheap/strong` 取——模型名的唯一定义处仍是 models.yaml。"""

    gateway = build(features="difficulty_router=on")

    assert isinstance(gateway._router, DifficultyRouter)
    assert gateway._router.cheap == "cheap-a"
    assert gateway._router.strong == "strong-a"


def test_explicit_aliases_override_the_derived_ones() -> None:
    gateway = build(
        features="difficulty_router=on",
        difficulty_strong_alias="cheap-a",
    )

    assert gateway._router.strong == "cheap-a"


def test_tools_request_never_routes_to_a_model_without_tools() -> None:
    """`_ensure_capabilities` 在重试循环**之外**抛错：排错就整次调用失败、不 fallback。

    所以分流器必须带能力检查——否则"优化"会把一个本来能用的请求变成必失败。
    """

    catalog = two_tier_catalog()
    catalog.models["cheap-a"].capabilities = []  # 便宜档不支持 tools
    gateway = build_gateway(make_settings(features="difficulty_router=on"), catalog=catalog)
    call = ChatCall(
        messages=[Message(role=MessageRole.USER, content="查一下库存")],
        tools=[ToolSpec(name="lookup", description="查库存", parameters={})],
    )

    assert gateway._route(call)[0] == "strong-a"


def test_missing_strong_route_falls_back_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """配不出两端就退回旧行为并**喊一声**——不要造一个只在一档里打转的路由器。"""

    with caplog.at_level(logging.WARNING, logger="agent_framework.router"):
        gateway = build_gateway(
            make_settings(features="difficulty_router=on"),
            catalog=make_catalog(["only-one"]),
        )

    assert isinstance(gateway._router, TaskRouter)
    assert any("difficulty_router" in record.getMessage() for record in caplog.records)


def test_both_optimisations_can_be_on_at_once() -> None:
    gateway = build(features="semantic_cache=on,difficulty_router=on")

    assert isinstance(gateway._cache, SemanticCache)
    assert isinstance(gateway._router, DifficultyRouter)


# --------------------------------------------------------------------------- #
# Kill Switch 的入口必须是唯一的
# --------------------------------------------------------------------------- #


def test_typo_in_features_blocks_startup() -> None:
    """静默忽略拼错的开关 = 运维以为开了、指标没动、没有任何地方变红。"""

    with pytest.raises(ValueError, match="未登记"):
        build(features="semantic_cach=on")


def test_turning_the_flag_off_restores_the_old_path() -> None:
    """这就是 Kill Switch 的动作本身：改 env → 重启 → 回到 W10 行为。"""

    assert build(features="semantic_cache=off")._cache is None
    assert isinstance(build(features="")._router, TaskRouter)

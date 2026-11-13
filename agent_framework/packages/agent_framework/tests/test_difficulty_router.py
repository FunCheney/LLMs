"""难度分流（W11 D5 / architecture §8）。

这组测试要守住两件容易在实现里悄悄走样的事：

1. **"改顺序"不能变成"缩短链"**。便宜档排在前面失败时，链上必须还有强档；
   强档路由里配的 `fallback`（`gpt-4o-mini`）也不能因为"我把链写死成两档"而消失。
2. **能力不足是唯一允许删候选的理由**。这不是难度问题——把工具请求递给不支持
   tools 的便宜档，拿到的是 400 加一次无谓往返，而 `_plan` 一开始就知道它不行。
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework.config import (
    Catalog,
    ModelConfig,
    Pricing,
    ProviderConfig,
    RouteConfig,
)
from agent_framework.llm.client import ChatCall, LLMGateway
from agent_framework.llm.difficulty_router import DifficultyRouter, difficulty_signals
from agent_framework.llm.errors import BadRequestError, RateLimitError
from agent_framework.llm.router import StaticRouter
from agent_framework.llm.types import Message, MessageRole, ToolSpec
from gateway_helpers import (
    FakeRegistry,
    RecordingSleep,
    StubClient,
    make_settings,
    text_response,
    user_call,
)

CHEAP = "qwen"
STRONG = "claude"


def build_catalog() -> Catalog:
    """四档 alias：cheap 一主两备，strong 一主一备。

    刻意让 `strong` 的 fallback 是 `gpt40`（一个便宜档 alias）——真实
    `models.yaml` 就是这么配的，而"硬写 `[cheap, strong]`"会把这条兜底弄丢。
    """

    provider = ProviderConfig(type="openai_compat", base_url="https://example.test/v1")
    pricing = Pricing(input=1.0, output=2.0, currency="CNY")

    def model(tier: str) -> ModelConfig:
        return ModelConfig(provider="p", model=f"{tier}-wire", tier=tier, pricing=pricing)

    return Catalog(
        providers={"p": provider},
        models={
            "qwen": model("cheap"),
            "deepseek": model("cheap"),
            "gpt40": model("cheap"),
            "claude": model("strong"),
        },
        routes={
            "default": RouteConfig(primary="qwen", fallback=["deepseek", "gpt40"]),
            "cheap": RouteConfig(primary="qwen", fallback=["deepseek", "gpt40"]),
            "strong": RouteConfig(primary="claude", fallback=["gpt40"]),
        },
        tasks={"chat": "cheap", "research": "strong"},
    )


def build_router(**overrides: Any) -> DifficultyRouter:
    params: dict[str, Any] = {"catalog": build_catalog(), "cheap": CHEAP, "strong": STRONG}
    params.update(overrides)

    return DifficultyRouter(**params)


HARD_TEXT = "为什么这两套方案的成本差这么多？请对比一下并输出 json"


def call_with(text: str, *, task: str = "chat") -> ChatCall:
    return ChatCall(messages=[Message(role=MessageRole.USER, content=text)], task=task)


# --------------------------------------------------------------------------- #
# 信号：分数必须能逐条归因
# --------------------------------------------------------------------------- #


def test_signals_are_countable_not_a_black_box() -> None:
    signals = difficulty_signals(user_call(HARD_TEXT))

    assert signals["hard_intent"] == 2  # 权重 2：单独成立即可达阈值
    assert signals["format_strict"] == 1
    assert signals["long_context"] == 0


def test_a_short_hard_request_still_crosses_the_threshold() -> None:
    """短而难的"规划一下灰度方案"不能因为文本不够长就被当成简单请求。

    W11 D5 实测（`evals/scripts/cost_ab.py` 实验 2）：`hard_intent` 权重为 1 时，
    6 条硬样本有 4 条只拿 1 分、落回便宜档。两个方向的误判代价不对称——
    多花钱看得见，答错看不见，所以强信号必须能单独达标。
    """

    chain = build_router().route_for(user_call("规划一个三阶段的灰度方案"))

    assert chain[0] == STRONG


def test_short_simple_request_has_no_signals() -> None:
    signals = difficulty_signals(user_call("把这句话翻成英文"))

    assert sum(signals.values()) == 0


def test_tools_alone_make_it_hard() -> None:
    """带 tools 就是硬需求：便宜档的 tool-call 稳定性最差。"""

    call = ChatCall(
        messages=[Message(role=MessageRole.USER, content="查一下天气")],
        tools=[ToolSpec(name="get_weather", description="天气", parameters={})],
    )

    assert difficulty_signals(call)["needs_tools"] == 1


# --------------------------------------------------------------------------- #
# 顺序：改顺序，不删候选
# --------------------------------------------------------------------------- #


def test_easy_request_puts_cheap_first_and_keeps_every_candidate() -> None:
    chain = build_router().route_for(user_call("把这句话翻成英文"))

    assert chain[0] == CHEAP
    # 强档仍在链尾：便宜档失败时要落得下去。
    assert chain[-1] == STRONG
    assert set(chain) == {CHEAP, "deepseek", "gpt40", STRONG}


def test_hard_request_puts_strong_first_but_keeps_cheap_as_backstop() -> None:
    chain = build_router().route_for(user_call(HARD_TEXT))

    assert chain[0] == STRONG
    assert CHEAP in chain


def test_strong_tier_task_keeps_its_configured_fallback() -> None:
    """research 走 strong 路由，其 fallback `gpt40` 不能在重排后消失。"""

    chain = build_router().route_for(call_with("把这句话翻成英文", task="research"))

    assert chain[0] == CHEAP  # 任务本身简单，仍先试便宜档
    assert "gpt40" in chain  # strong 路由的兜底还在


def test_route_without_call_context_matches_task_router() -> None:
    """没有请求上下文时回落 `route(task)`，与 `TaskRouter` 完全一致。"""

    router = build_router()

    assert router.route("chat") == [CHEAP, "deepseek", "gpt40"]
    assert router.route("research") == [STRONG, "gpt40"]


# --------------------------------------------------------------------------- #
# 能力：唯一允许删候选的路径
# --------------------------------------------------------------------------- #


def test_capability_miss_removes_cheap_instead_of_trying_it() -> None:
    """便宜档不支持 tools → 直接挪走，不做"先试一次"。"""

    router = build_router(
        capability_check=lambda alias, call: alias != CHEAP,
    )
    call = ChatCall(
        messages=[Message(role=MessageRole.USER, content="查一下天气")],
        tools=[ToolSpec(name="get_weather", description="天气", parameters={})],
    )

    chain = router.route_for(call)

    assert CHEAP not in chain
    assert chain[0] == STRONG


def test_capability_check_does_not_apply_when_it_passes() -> None:
    router = build_router(capability_check=lambda alias, call: True)

    chain = router.route_for(user_call("查一下天气"))

    assert chain[0] == CHEAP
    assert STRONG in chain


def test_unknown_aliases_are_filtered_not_passed_through() -> None:
    """cheap 配错成不存在的 alias 时不能把它塞进链里——那会在 `_plan` 变成 500。"""

    chain = build_router(cheap="nope").route_for(user_call("把这句话翻成英文"))

    assert "nope" not in chain
    # 配置里的默认链仍然兜住了这个错误配置：退化到"按 task 的默认顺序"。
    assert chain[0] == CHEAP


# --------------------------------------------------------------------------- #
# 可解释性
# --------------------------------------------------------------------------- #


def test_explain_says_which_signal_fired_and_why() -> None:
    payload = build_router().explain(user_call(HARD_TEXT))

    assert payload["reason"] == "强档"
    assert payload["score"] >= payload["threshold"]
    assert payload["signals"]["hard_intent"] == 2
    assert payload["candidates"][0] == STRONG


def test_explain_reports_cheap_reason_for_simple_request() -> None:
    payload = build_router().explain(user_call("把这句话翻成英文"))

    assert payload["reason"] == "便宜档"
    assert payload["score"] == 0


# --------------------------------------------------------------------------- #
# 接线：gateway 走 `route_for`，没有它就走 `route`
# --------------------------------------------------------------------------- #


def build_gateway(
    router: Any, clients: dict[str, StubClient], **settings_overrides: Any
) -> LLMGateway:
    catalog = build_catalog()

    return LLMGateway(
        registry=FakeRegistry(catalog, clients),  # type: ignore[arg-type]
        router=router,
        catalog=catalog,
        settings=make_settings(max_attempts=1, **settings_overrides),
        sleep=RecordingSleep(),
    )


def clients_for(*aliases: str) -> dict[str, StubClient]:
    return {alias: StubClient(alias, chat_results=[text_response(alias)]) for alias in aliases}


@pytest.mark.asyncio
async def test_gateway_routes_hard_request_to_strong_first() -> None:
    clients = clients_for(CHEAP, "deepseek", "gpt40", STRONG)
    gateway = build_gateway(build_router(), clients)

    outcome = await gateway.chat(user_call(HARD_TEXT))

    assert outcome.alias == STRONG
    assert clients[STRONG].chat_calls == 1
    assert clients[CHEAP].chat_calls == 0


@pytest.mark.asyncio
async def test_gateway_routes_easy_request_to_cheap_first() -> None:
    clients = clients_for(CHEAP, "deepseek", "gpt40", STRONG)
    gateway = build_gateway(build_router(), clients)

    outcome = await gateway.chat(user_call("把这句话翻成英文"))

    assert outcome.alias == CHEAP
    assert clients[STRONG].chat_calls == 0


@pytest.mark.asyncio
async def test_cheap_failure_still_falls_through_to_strong() -> None:
    """这是"只改顺序"的真正意义：便宜档排在前面失败后，链没有断。

    三个便宜档全挂，最后落到强档。`fallback_from` 记的是**紧邻的上一个**
    候选（这里是 `gpt40`），不是链首——想看"从哪儿掉下来的整条路径"要看
    candidate 日志，`fallback_from` 只回答"我是不是 fallback 来的"。
    """

    def failing(alias: str) -> StubClient:
        return StubClient(alias, chat_results=[RateLimitError("429", provider="p")])

    clients = {
        CHEAP: failing(CHEAP),
        "deepseek": failing("deepseek"),
        "gpt40": failing("gpt40"),
        STRONG: StubClient(STRONG, chat_results=[text_response("strong answer")]),
    }
    gateway = build_gateway(build_router(), clients)

    outcome = await gateway.chat(user_call("把这句话翻成英文"))

    assert outcome.alias == STRONG
    assert outcome.fallback_from == "gpt40"


@pytest.mark.asyncio
async def test_static_router_is_untouched_by_the_new_hook() -> None:
    """没有 `route_for` 的 Router 行为一字不变——协议没被撑大。"""

    clients = clients_for(STRONG)
    gateway = build_gateway(StaticRouter([STRONG]), clients)

    outcome = await gateway.chat(user_call(HARD_TEXT))

    assert outcome.alias == STRONG


@pytest.mark.asyncio
async def test_empty_chain_names_the_router_that_emptied_it() -> None:
    """空候选时报错要能指向是哪个 router 干的——否则值班同学只能翻代码。

    `DifficultyRouter` 自己不会产出空链（`models.yaml` 的默认链总会兜住），
    所以这条契约用 `StaticRouter([])` 直接打到 `_plan`。
    """

    gateway = build_gateway(StaticRouter([]), clients_for(STRONG))

    with pytest.raises(BadRequestError) as caught:
        gateway._plan(user_call("把这句话翻成英文"))

    assert "static" in str(caught.value)

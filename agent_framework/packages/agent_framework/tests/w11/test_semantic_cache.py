"""语义缓存（W11 D5）。

这组测试的重点不是"能命中"，而是**三道闸都能各自单独把答案拦下来**：
阈值、TTL、资产版本。任意一道失效，系统都会安静地返回一个"读起来正常"的错答案，
而这比报错难发现得多——所以每一道都要有一条只让它失效的用例。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from agent_framework.llm.client import ChatCall, ChatOutcome, LLMGateway
from agent_framework.llm.router import StaticRouter
from agent_framework.llm.types import Message, MessageRole, ToolSpec
from agent_framework.observability import RequestLog
from agent_framework.resilience.semantic_cache import (
    CacheStats,
    HashingEmbedder,
    SemanticCache,
    cosine,
    has_time_sensitive_term,
    last_user_text,
)
from gateway_helpers import (
    FakeRegistry,
    RecordingSleep,
    StubClient,
    make_catalog,
    make_settings,
    text_response,
    user_call,
)

ALIAS = "primary"


def make_cache(**overrides: Any) -> SemanticCache:
    params: dict[str, Any] = {"threshold": 0.92, "ttl_s": 3600.0}
    params.update(overrides)

    return SemanticCache(**params)


def outcome(text: str = "ok", *, degraded: bool = False) -> ChatOutcome:
    return ChatOutcome(
        response=text_response(text),
        alias=ALIAS,
        degraded=degraded,
    )


# --------------------------------------------------------------------------- #
# 向量与时效词：两个最容易被"看起来对"糊弄过去的判断
# --------------------------------------------------------------------------- #


def test_embedding_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder()

    left = embedder.embed("预算怎么分配")
    right = embedder.embed("预算怎么分配")

    assert left == right
    assert abs(sum(item * item for item in left) - 1.0) < 1e-9


def test_cosine_rejects_dimension_mismatch_instead_of_padding() -> None:
    with pytest.raises(ValueError, match="维度"):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_same_question_is_closer_than_a_different_one() -> None:
    embedder = HashingEmbedder()
    base = embedder.embed("如何给 MCP server 配超时")

    near = cosine(base, embedder.embed("如何给 MCP server 配置超时"))
    far = cosine(base, embedder.embed("明天北京天气怎么样"))

    assert near > far
    assert near >= 0.92


@pytest.mark.parametrize("text", ["今天有什么新闻", "最新的政策", "what is the latest?"])
def test_time_sensitive_terms_are_blocked(text: str) -> None:
    assert has_time_sensitive_term(text)


@pytest.mark.parametrize("text", ["known issue 怎么排查", "snow 是什么梗"])
def test_english_word_boundary_does_not_catch_known_or_snow(text: str) -> None:
    """`"now" in "known"` 是真——用子串判断会把普通问题全判成时效问题。"""

    assert not has_time_sensitive_term(text)


# --------------------------------------------------------------------------- #
# 闸门一：阈值
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_repeat_question_hits_and_does_not_mutate_the_stored_outcome() -> None:
    cache = make_cache()
    stored = outcome("第一次的答案")
    await cache.put("怎么配置超时", stored)

    hit = await cache.get("怎么配置超时")

    assert hit is not None
    assert hit.cache_hit is True
    assert cache._entries[0].outcome.cache_hit is False  # 原件不能被标记


@pytest.mark.asyncio
async def test_paraphrase_hits_but_unrelated_question_misses() -> None:
    cache = make_cache()
    await cache.put("如何给 MCP server 配置超时", outcome())

    assert await cache.get("如何给 MCP server 配置超时参数") is not None
    assert await cache.get("明天北京天气怎么样") is None


@pytest.mark.asyncio
async def test_raising_the_threshold_turns_a_near_hit_into_a_miss() -> None:
    """阈值必须是唯一的松紧旋钮：调高就能把"差不多"挡掉。"""

    loose = make_cache(threshold=0.6)
    strict = make_cache(threshold=0.999)
    await loose.put("如何给 MCP server 配置超时", outcome())
    await strict.put("如何给 MCP server 配置超时", outcome())

    query = "如何给 MCP server 配置超时参数"

    assert await loose.get(query) is not None
    assert await strict.get(query) is None


def test_illegal_threshold_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="threshold"):
        make_cache(threshold=0.0)


# --------------------------------------------------------------------------- #
# 闸门二：TTL
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_expired_entry_is_counted_separately_from_a_miss() -> None:
    """`expired` 混进 `misses`，看板上就只剩"命中率低"，不知道该调 TTL 还是查版本。"""

    now = [1000.0]
    cache = make_cache(ttl_s=10.0, clock=lambda: now[0])
    await cache.put("怎么配置超时", outcome())

    now[0] += 11.0
    assert await cache.get("怎么配置超时") is None

    stats = cache.stats.as_dict()
    assert stats["expired"] == 1
    assert stats["misses"] == 0
    assert cache.size == 0  # 过期即删，不是"留着但不用"


# --------------------------------------------------------------------------- #
# 闸门三：资产版本
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_prompt_change_invalidates_old_answers() -> None:
    cache = make_cache(asset_digest="v1")
    await cache.put("怎么配置超时", outcome())

    cache.asset_digest = "v2"  # 换了 prompt / 换了模型

    assert await cache.get("怎么配置超时") is None
    assert cache.stats.as_dict()["invalidated"] == 1
    assert cache.stats.as_dict()["misses"] == 0


# --------------------------------------------------------------------------- #
# 不该缓存的东西
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_time_sensitive_query_is_blocked_before_embedding() -> None:
    cache = make_cache()

    assert await cache.get("今天汇率多少") is None
    assert cache.stats.as_dict()["blocked"] == 1


@pytest.mark.asyncio
async def test_degraded_answer_is_never_stored() -> None:
    """把"这不是真答案"缓存下来，会传播一整天。"""

    cache = make_cache()
    await cache.put("怎么配置超时", outcome(degraded=True))

    assert cache.size == 0
    assert await cache.get("怎么配置超时") is None


@pytest.mark.asyncio
async def test_cache_full_evicts_oldest_and_counts_it() -> None:
    cache = make_cache(max_entries=2)
    for index in range(3):
        await cache.put(f"问题编号 {index} 的细节", outcome(str(index)))

    assert cache.size == 2
    assert cache.stats.as_dict()["evicted"] == 1


# --------------------------------------------------------------------------- #
# 与 gateway 接线
# --------------------------------------------------------------------------- #


def build_gateway(cache: SemanticCache | None, clients: dict[str, StubClient]) -> LLMGateway:
    catalog = make_catalog([ALIAS])

    return LLMGateway(
        registry=FakeRegistry(catalog, clients),  # type: ignore[arg-type]
        router=StaticRouter([ALIAS]),
        catalog=catalog,
        settings=make_settings(max_attempts=1),
        sleep=RecordingSleep(),
        cache=cache,
    )


def request_logs(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "agent_framework.request"
    ]


@pytest.mark.asyncio
async def test_gateway_second_identical_call_does_not_reach_upstream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clients = {ALIAS: StubClient(ALIAS, chat_results=[text_response("answer")])}
    gateway = build_gateway(make_cache(), clients)

    with caplog.at_level(logging.INFO, logger="agent_framework.request"):
        first = await gateway.chat(user_call("怎么配置超时"))
        second = await gateway.chat(user_call("怎么配置超时"))

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert clients[ALIAS].chat_calls == 1  # 上游只被打了一次

    logs = request_logs(caplog)
    assert len(logs) == 2  # 命中也要有日志，否则请求量会凭空少一截
    assert logs[1]["cache_hit"] is True
    assert logs[1]["cost"] == 0.0
    assert logs[1]["prompt_tokens"] == 0


@pytest.mark.asyncio
async def test_tool_calls_are_never_cached() -> None:
    """工具答案取决于那一刻的真实世界，复用等于编数据。"""

    clients = {ALIAS: StubClient(ALIAS, chat_results=[text_response("a"), text_response("b")])}
    gateway = build_gateway(make_cache(), clients)
    tools = [ToolSpec(name="get_weather", description="天气", parameters={})]

    first = await gateway.chat(user_call("北京天气", tools=tools))
    second = await gateway.chat(user_call("北京天气", tools=tools))

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert clients[ALIAS].chat_calls == 2


@pytest.mark.asyncio
async def test_pinned_model_is_never_served_from_cache() -> None:
    """钉了 model 的调用方（评测/调试）要的是那个模型的结果，不是"某个答案"。"""

    clients = {ALIAS: StubClient(ALIAS, chat_results=[text_response("a"), text_response("b")])}
    gateway = build_gateway(make_cache(), clients)

    await gateway.chat(user_call("怎么配置超时"))
    pinned = await gateway.chat(user_call("怎么配置超时", model=ALIAS))

    assert pinned.cache_hit is False
    assert clients[ALIAS].chat_calls == 2


@pytest.mark.asyncio
async def test_same_temperature_still_hits() -> None:
    """产品端每个调用都带 `temperature`；把它当成"不许缓存"等于缓存从没生效过。"""

    clients = {ALIAS: StubClient(ALIAS, chat_results=[text_response("a")])}
    gateway = build_gateway(make_cache(), clients)

    def ask() -> ChatCall:
        return ChatCall(
            messages=[Message(role=MessageRole.USER, content="怎么配置超时")],
            opts={"temperature": 0.2, "max_tokens": 800},
        )

    first = await gateway.chat(ask())
    second = await gateway.chat(ask())

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert clients[ALIAS].chat_calls == 1


@pytest.mark.asyncio
async def test_different_temperature_misses() -> None:
    """同问题不同采样参数是**两个键**，不是"不许缓存"。"""

    clients = {ALIAS: StubClient(ALIAS, chat_results=[text_response("a"), text_response("b")])}
    gateway = build_gateway(make_cache(), clients)

    def ask(temperature: float) -> ChatCall:
        return ChatCall(
            messages=[Message(role=MessageRole.USER, content="怎么配置超时")],
            opts={"temperature": temperature},
        )

    await gateway.chat(ask(0.2))
    hot = await gateway.chat(ask(0.9))

    assert hot.cache_hit is False
    assert clients[ALIAS].chat_calls == 2


@pytest.mark.asyncio
async def test_no_cache_configured_means_behaviour_is_unchanged() -> None:
    clients = {ALIAS: StubClient(ALIAS, chat_results=[text_response("a"), text_response("b")])}
    gateway = build_gateway(None, clients)

    first = await gateway.chat(user_call("怎么配置超时"))
    second = await gateway.chat(user_call("怎么配置超时"))

    assert first.cache_hit is False
    assert second.cache_hit is False


# --------------------------------------------------------------------------- #
# 小而必要的口径
# --------------------------------------------------------------------------- #


def test_stats_lookups_include_blocked_so_rates_cannot_exceed_one() -> None:
    stats = CacheStats(hits=1, misses=1, blocked=2)

    assert stats.lookups == 4
    assert stats.hit_rate == 0.25


def test_last_user_text_uses_the_final_turn_only() -> None:
    call = ChatCall(messages=[
        Message(role=MessageRole.USER, content="第一问"),
        Message(role=MessageRole.ASSISTANT, content="答"),
        Message(role=MessageRole.USER, content="第二问"),
    ])

    assert last_user_text(call) == "第二问"


def test_request_log_has_cache_hit_field_default_off() -> None:
    assert RequestLog().as_dict()["cache_hit"] is False

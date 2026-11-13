"""W11 D2：降级策略——**不重试**、**不伪装成功**、**必须留结构化标记**。

这组用例守四条：

1. **默认不兜底**：`allow_canned=False` 时 `answered=False`，让原始 503 抛出去；
2. **兜底不能长得像成功**：`finish_reason != stop`，且 `degraded=True`——
   下游判"正常完成"靠 `finish_reason == stop`，这条不会混进正常率；
3. **缓存命中要复制**：直接改传进来的 `ChatOutcome`，会让同一份缓存被后续
   请求共享，`degraded` 字段串味；
4. **顺序是缓存 → 兜底 → 抛出**，每个手段只试一次（再试一次就是 ADR-003 禁止的第二次重试）。
"""

from __future__ import annotations

from agent_framework.llm.client import ChatCall, ChatOutcome
from agent_framework.llm.errors import UpstreamTimeoutError
from agent_framework.llm.types import (
    FinishReason,
    LLMResponse,
    Message,
    MessageRole,
    Usage,
)
from agent_framework.resilience.degradation import (
    CANNED_MESSAGE,
    DegradationPolicy,
    DegradedAnswer,
)


def make_call() -> ChatCall:
    return ChatCall(messages=[Message(role=MessageRole.USER, content="ping")])


def make_outcome(text: str = "缓存里的旧答案") -> ChatOutcome:
    return ChatOutcome(
        response=LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=text),
            finish_reason=FinishReason.STOP,
            model="gpt-4o-mini",
            usage=Usage(input_tokens=10, output_tokens=5),
        ),
        alias="gpt-4o-mini",
        retry_count=1,
        fallback_from="qwen3.7-plus",
    )


def timeout() -> UpstreamTimeoutError:
    return UpstreamTimeoutError("注入超时", provider="fake")


# --------------------------------------------------------------------------- #
# 默认：不兜底
# --------------------------------------------------------------------------- #


async def test_default_policy_refuses_to_invent_an_answer() -> None:
    """一个"请稍后再试"的假回答比 503 更糟：用户以为拿到了答案、SLO 少记一条失败。"""

    result = await DegradationPolicy().fallback(make_call(), error=timeout())

    assert result.answered is False
    assert result.answer is None
    assert result.reason, "拒绝了就得说清为什么拒绝"


async def test_cache_miss_alone_still_refuses() -> None:
    async def miss(_call: ChatCall) -> ChatOutcome | None:
        return None

    policy = DegradationPolicy(cache=miss)
    result = await policy.fallback(make_call(), error=timeout())

    assert result.answered is False
    assert policy.hits == {}, "没命中不该计数"


# --------------------------------------------------------------------------- #
# 兜底话术：能用，但必须一眼看出不是真答案
# --------------------------------------------------------------------------- #


async def test_canned_answer_is_marked_not_mistaken_for_success() -> None:
    policy = DegradationPolicy(allow_canned=True)
    result = await policy.fallback(make_call(), error=timeout())

    assert result.answered is True
    assert isinstance(result.answer, DegradedAnswer)
    assert result.answer.path == "canned"

    outcome = result.answer.outcome
    assert outcome.degraded is True
    assert outcome.cache_hit is False
    # 关键：不能长得像正常完成
    assert outcome.response.finish_reason is FinishReason.UNKNOWN
    assert outcome.response.finish_reason is not FinishReason.STOP
    assert outcome.response.raw_finish_reason == "degraded"
    assert outcome.response.text == CANNED_MESSAGE
    # 报告里要能回答"这次降级是因为超时还是 429"
    assert outcome.fallback_from == "UpstreamTimeoutError"


async def test_custom_canned_text_and_alias() -> None:
    policy = DegradationPolicy(allow_canned=True, canned_text="排队中", alias="shed")
    result = await policy.fallback(make_call())

    assert result.answer is not None
    assert result.answer.outcome.response.text == "排队中"
    assert result.answer.outcome.alias == "shed"
    # 没有 error 时不该编一个名字出来
    assert result.answer.outcome.fallback_from is None


# --------------------------------------------------------------------------- #
# 缓存：优先、复制、可统计
# --------------------------------------------------------------------------- #


async def test_cache_hit_wins_over_canned() -> None:
    async def hit(_call: ChatCall) -> ChatOutcome | None:
        return make_outcome()

    policy = DegradationPolicy(cache=hit, allow_canned=True)
    result = await policy.fallback(make_call(), error=timeout())

    assert result.answer is not None
    assert result.answer.path == "cache", "缓存优先：它给的是真答案，只是旧"
    assert result.answer.outcome.cache_hit is True
    assert result.answer.outcome.response.text == "缓存里的旧答案"
    assert policy.hits == {"cache": 1}


async def test_cache_hit_does_not_mutate_the_shared_outcome() -> None:
    """打标记必须复制。原地改会让同一份缓存对象在后续请求里串味。"""

    shared = make_outcome()

    async def hit(_call: ChatCall) -> ChatOutcome | None:
        return shared

    policy = DegradationPolicy(cache=hit)
    result = await policy.fallback(make_call(), error=timeout())

    assert result.answer is not None
    assert result.answer.outcome is not shared
    assert result.answer.outcome.degraded is True

    # 原件必须是干净的
    assert shared.degraded is False
    assert shared.cache_hit is False
    # 其余元信息要原样带过来
    assert result.answer.outcome.retry_count == shared.retry_count
    assert result.answer.outcome.fallback_from == shared.fallback_from


async def test_cache_is_consulted_once_per_call() -> None:
    """每个手段只试一次。再试就是第二次重试，归 gateway 管（ADR-003）。"""

    consulted = 0

    async def hit(_call: ChatCall) -> ChatOutcome | None:
        nonlocal consulted
        consulted += 1
        return None

    policy = DegradationPolicy(cache=hit)
    await policy.fallback(make_call(), error=timeout())
    await policy.fallback(make_call(), error=timeout())

    assert consulted == 2, consulted


async def test_hits_are_counted_per_path() -> None:
    async def hit(_call: ChatCall) -> ChatOutcome | None:
        return make_outcome()

    policy = DegradationPolicy(cache=hit, allow_canned=True)
    await policy.fallback(make_call())
    await policy.fallback(make_call())

    assert policy.hits == {"cache": 2}
    assert "canned" not in policy.hits


async def test_as_dict_is_the_log_line() -> None:
    policy = DegradationPolicy(allow_canned=True)
    result = await policy.fallback(make_call(), error=timeout())

    assert result.answer is not None
    assert result.answer.as_dict() == {
        "path": "canned",
        "alias": "degraded",
        "degraded": True,
        "cache_hit": False,
    }

#!/usr/bin/env python
"""混沌演练（W11 D2）：三类故障注入 + 四项自愈/边界性质。

    python evals/scripts/chaos.py            # 全部场景
    python evals/scripts/chaos.py --json     # 机器可读（给 history json）

## 为什么不连真厂商

三类故障都是**注入**进 `httpx.MockTransport` 的，断网可复现，零成本，秒级。
真厂商的 429/500 什么时候来不由我们决定，用它做演练的结果是"跑了三次，
两次没触发"——那不是演练，是抽奖。

用的是**真的 `LLMGateway`**（真重试、真计量、真错误映射），只换了最底层的传输。
换整条链路会让演练变成"测我的替身写得对不对"。

## 每个场景断言什么

1. **可复现**：故障真的发生了（上游被调用次数 > 0，错误码符合预期）。
2. **不崩**：错误以 `LLMError` 的形态出现并带上 code，不是未捕获异常。
3. **不放大**：熔断打开之后，上游调用次数**停止增长**（`calls_after_open == 0`）。
   这一条是本周头号坑的反向验证——如果熔断器自己又重试一遍，
   这个数就会大于 0。
4. **能恢复**：冷却结束 → 半开只放一个探测 → 成功 → 闭合。
5. **故障域不跑偏**：我方请求错误（400）不计数，永远不熔断。

## 一次 `gateway.chat()` 到底打几次上游（实测踩过的坑）

第一版演练写成 `expected_calls = threshold * max_attempts`，跑出来是 18 不是 6。
差的那一步是**候选链**：`chat()` 会依次试 `models.yaml` 里该 task 的全部候选，
每个候选再重试 `max_attempts` 次。所以真实倍数是

    上游调用 = 放行次数 × 候选链长度 × max_attempts

这个乘积属于 `LLMGateway`（ADR-003 归它），**不属于熔断器**。
熔断器自己的贡献必须恰好是 0——这才是本文件要断言的那件事，
所以除了比对总数，还单独记了一个 `calls_after_open`。

## 一个诚实的前提

真实链路的"成功率"由用户请求定义；这里由"我们主动打的 N 次调用"定义。
所以下面报的成功率是**演练口径**，不是线上可用性——线上的数在 `ops/slo_report.py`。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework.config import Settings, load_catalog  # noqa: E402
from agent_framework.llm.client import ChatCall, LLMGateway  # noqa: E402
from agent_framework.llm.errors import (  # noqa: E402
    CircuitOpenError,
    LLMError,
    error_code_for,
)
from agent_framework.llm.registry import Registry  # noqa: E402
from agent_framework.llm.router import TaskRouter  # noqa: E402
from agent_framework.llm.types import Message, MessageRole  # noqa: E402
from agent_framework.resilience.circuit_breaker import CircuitBreaker  # noqa: E402
from agent_framework.resilience.degradation import DegradationPolicy  # noqa: E402

# 假 key：transport 被替换掉了，不会有真实请求。真实 key 在演练里没有意义，
# 反而有"演练脚本意外打到线上"的风险。
os.environ.setdefault("OPENAI_API_KEY", "chaos-drill-not-a-real-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "chaos-drill-not-a-real-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "chaos-drill-not-a-real-key")

TASK = "chat"
ALIAS = "gpt-4o-mini"


@dataclass
class Upstream:
    """故障注入的传输层。所有计数都在这里，断言也读这里。"""

    mode: str = "ok"  # ok | timeout | http500 | http400 | ratelimit
    calls: int = 0
    payload: dict[str, Any] = field(default_factory=dict)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.mode == "timeout":
            raise httpx.ReadTimeout("injected timeout", request=request)
        if self.mode == "http500":
            return httpx.Response(500, json={"error": {"message": "injected 500"}})
        # 400 是**我方请求错**，用来验证"故障域不跑偏"：它不该计入熔断。
        if self.mode == "http400":
            return httpx.Response(
                400, json={"error": {"message": "injected bad request"}}
            )
        if self.mode == "ratelimit":
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": "injected 429", "code": "rate_limit_exceeded"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "chaos-ok",
                "model": ALIAS,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )


def build_gateway(
    upstream: Upstream, *, max_attempts: int = 2, breaker: CircuitBreaker | None = None
) -> tuple[LLMGateway, list[str]]:
    """返回 `(gateway, 候选链)`。候选链是断言的分母，必须取自真的 router。"""

    settings = Settings(max_attempts=max_attempts, backoff_base_s=0.0, request_timeout_s=2.0)
    catalog = load_catalog(settings.models_yaml)
    router = TaskRouter(catalog)

    gateway = LLMGateway(
        registry=Registry(catalog, settings, transport=upstream.transport()),
        router=router,
        catalog=catalog,
        settings=settings,
        breaker=breaker,
    )
    return gateway, router.route(TASK)


def make_call() -> ChatCall:
    return ChatCall(
        messages=[
            Message(role=MessageRole.SYSTEM, content="chaos drill"),
            Message(role=MessageRole.USER, content="ping"),
        ],
        task=TASK,
        opts={"max_tokens": 8, "temperature": 0.0},
    )


@dataclass
class Scenario:
    name: str
    injected: str
    expected_code: str
    calls: int = 0
    ok: int = 0
    refused: int = 0
    upstream_calls: int = 0

    # 两个错误字段分开记。第一版只有一个 `error`，被最后一轮熔断拒绝覆写，
    # 于是"上游到底报了什么"在报告里永远显示 CIRCUIT_OPEN——真正要看的那个码丢了。
    upstream_error: str = ""
    last_refusal: str = ""

    # 熔断打开**之后**仍然打出去的上游调用。熔断器不重复重试的硬证据。
    calls_after_open: int = 0

    note: str = ""
    breaker_state: str = ""

    @property
    def admitted(self) -> int:
        """被放行到上游的次数（含成功与失败）。"""

        return self.calls - self.refused

    @property
    def success_rate(self) -> float:
        return self.ok / self.admitted if self.admitted > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "injected": self.injected,
            "expected_code": self.expected_code,
            "calls": self.calls,
            "ok": self.ok,
            "refused_by_breaker": self.refused,
            "calls_after_open": self.calls_after_open,
            "upstream_calls": self.upstream_calls,
            "upstream_error": self.upstream_error,
            "last_refusal": self.last_refusal,
            "success_rate_of_admitted": round(self.success_rate, 4),
            "breaker_state": self.breaker_state,
            "note": self.note,
        }


async def drive(
    invoke: Callable[[], Awaitable[Any]],
    upstream: Upstream,
    scenario: Scenario,
    *,
    rounds: int,
) -> None:
    """打 N 次。熔断拒绝单独计数——它不是"用户请求失败"，是"我们主动省下的一次调用"。

    `invoke` 是"一次业务调用"的封装：写成可调用对象而不是写死 `breaker.call(gateway.chat)`，
    是因为**线上形状与演练形状必须一致**——F1–F3 用外面套一层，F7 用接线上的那一个。
    """

    for _ in range(rounds):
        scenario.calls += 1
        before = upstream.calls
        try:
            await invoke()
        except CircuitOpenError as exc:
            scenario.refused += 1
            scenario.last_refusal = error_code_for(exc)
            # 被拒绝的这一轮如果还打出了上游调用，说明"拒绝"只是个说法。
            scenario.calls_after_open += upstream.calls - before
        except LLMError as exc:
            code = error_code_for(exc)
            # 断言 `LLMError` 携带 code：没有 code 的错误到 HTTP 层会变成兜底 502，
            # 而"兜底"意味着归因丢失。
            if not code:
                raise AssertionError(f"{scenario.name}: {type(exc).__name__} 没有 code") from exc
            scenario.upstream_error = code
        else:
            scenario.ok += 1


async def run_scenario(
    *, name: str, mode: str, expected: str, rounds: int, threshold: int, max_attempts: int
) -> tuple[Scenario, int]:
    upstream = Upstream(mode=mode)
    gateway, chain = build_gateway(upstream, max_attempts=max_attempts)
    breaker = CircuitBreaker(failure_threshold=threshold, reset_after_s=30.0)

    scenario = Scenario(name=name, injected=mode, expected_code=expected)
    await drive(
        lambda: breaker.call(lambda: gateway.chat(make_call())),
        upstream,
        scenario,
        rounds=rounds,
    )

    scenario.upstream_calls = upstream.calls
    scenario.breaker_state = breaker.state
    return scenario, len(chain)


def assert_contained(
    scenario: Scenario,
    *,
    rounds: int,
    threshold: int,
    chain: int,
    max_attempts: int,
    retryable: bool = True,
) -> None:
    """熔断生效的四条：按阈值打开、此后全拒、拒的时候零上游、总数不超预算。"""

    assert scenario.admitted == threshold, (scenario.admitted, threshold)
    assert scenario.refused == rounds - threshold, (scenario.refused, rounds)
    # 核心断言：熔断器自己一次都没有重试。
    assert scenario.calls_after_open == 0, scenario.calls_after_open

    per_call = chain * (max_attempts if retryable else 1)
    assert scenario.upstream_calls == threshold * per_call, (
        scenario.upstream_calls,
        threshold * per_call,
    )


async def scenario_model_timeout() -> Scenario:
    """F1：模型超时。期望：重试耗尽 → UPSTREAM_TIMEOUT；熔断打开；上游调用停止增长。"""

    threshold, max_attempts, rounds = 3, 2, 12
    scenario, chain = await run_scenario(
        name="F1 模型超时", mode="timeout", expected="UPSTREAM_TIMEOUT",
        rounds=rounds, threshold=threshold, max_attempts=max_attempts,
    )
    assert scenario.upstream_error == "UPSTREAM_TIMEOUT", scenario.upstream_error
    assert scenario.last_refusal == "CIRCUIT_OPEN", scenario.last_refusal
    assert_contained(
        scenario, rounds=rounds, threshold=threshold,
        chain=chain, max_attempts=max_attempts,
    )
    scenario.note = (
        f"放行 {threshold} 次，每次走 {chain} 条候选 × {max_attempts} 次尝试 = "
        f"{chain * max_attempts} 次上游调用，合计 {scenario.upstream_calls}；"
        f"打开后 {scenario.refused} 次请求打出 {scenario.calls_after_open} 次上游调用"
    )
    return scenario


async def scenario_tool_500() -> Scenario:
    """F2：上游 500。期望：UPSTREAM_ERROR；与 F1 同构，但走 5xx 分支。"""

    threshold, max_attempts, rounds = 2, 2, 8
    scenario, chain = await run_scenario(
        name="F2 上游 500", mode="http500", expected="UPSTREAM_ERROR",
        rounds=rounds, threshold=threshold, max_attempts=max_attempts,
    )
    assert scenario.upstream_error == "UPSTREAM_ERROR", scenario.upstream_error
    assert_contained(
        scenario, rounds=rounds, threshold=threshold,
        chain=chain, max_attempts=max_attempts,
    )
    scenario.note = (
        f"5xx 与超时同样计入失败（都在白名单里）；"
        f"上游调用 {scenario.upstream_calls} = {threshold}×{chain}×{max_attempts}"
    )
    return scenario


async def scenario_rate_limit() -> Scenario:
    """F3：429 + Retry-After。期望：RATE_LIMITED 且 429 透传给调用方。"""

    threshold, max_attempts, rounds = 2, 2, 6
    scenario, chain = await run_scenario(
        name="F3 限流 429", mode="ratelimit", expected="RATE_LIMITED",
        rounds=rounds, threshold=threshold, max_attempts=max_attempts,
    )
    assert scenario.upstream_error == "RATE_LIMITED", scenario.upstream_error
    assert_contained(
        scenario, rounds=rounds, threshold=threshold,
        chain=chain, max_attempts=max_attempts,
    )
    scenario.note = (
        f"429 也计失败（上游在说'我扛不住'）；"
        f"上游调用 {scenario.upstream_calls} 次后熔断，未触发额外重试"
    )
    return scenario


async def scenario_bad_request_never_trips() -> Scenario:
    """F4：我方请求错（400）**不**计入熔断。

    这一条不是"顺手加的"。把 400 算进失败，会让"我们自己把参数写错"
    表现成"上游被熔断"，值班去查厂商状态页，查不到任何东西。
    """

    threshold, max_attempts, rounds = 2, 2, 8
    scenario, chain = await run_scenario(
        name="F4 400 不熔断", mode="http400", expected="BAD_REQUEST",
        rounds=rounds, threshold=threshold, max_attempts=max_attempts,
    )
    assert scenario.upstream_error == "BAD_REQUEST", scenario.upstream_error
    assert scenario.refused == 0, scenario.refused
    assert scenario.breaker_state == "closed", scenario.breaker_state

    # 不可重试 → 每个候选只打一次，所以倍数是候选链长度而不是 ×max_attempts。
    assert scenario.upstream_calls == rounds * chain, (scenario.upstream_calls, rounds * chain)
    scenario.note = (
        f"{rounds} 次全部失败仍保持 closed（400 不在白名单）；"
        f"上游调用 {scenario.upstream_calls} = {rounds}×{chain}（不可重试，每次候选只打一次）"
    )
    return scenario


async def scenario_half_open_limits_probes() -> Scenario:
    """恢复期：冷却结束后只放**一个**探测。

    这条最容易写错：放行多个探测的话，上游刚恢复就被 N 个并发同时打，
    很容易再次全失败 → 立刻回到 OPEN，形成"永远探测不成功"的抖动。

    ## 第一版为什么错了（实测 5 而不是 1）

    第一版直接 `gather` 了 5 个调用，期望"1 个探测 + 4 个拒绝"，实测 5 个全放行。
    原因不是熔断器漏了判断，而是**这个断言需要一个"飞行中"的探测**：
    `httpx.MockTransport` 的 handler 是同步的，探测一次到底、中途不让出事件循环，
    于是第一个调用回来时状态已经 CLOSED，后 4 个被正常放行——**这是对的**。

    所以要测的不是"5 个并发里只过 1 个"，而是"探测还在飞行时，其余请求被快速拒绝"。
    用一对 Event 把探测停在半空中，这个性质才可观测。
    """

    upstream = Upstream(mode="http500")
    gateway, _chain = build_gateway(upstream, max_attempts=1)
    now = [1000.0]
    breaker = CircuitBreaker(
        failure_threshold=2, reset_after_s=5.0, clock=lambda: now[0]
    )

    # 先把它打开（这两次必然失败，失败就是目的）
    for _ in range(2):
        with contextlib.suppress(LLMError):
            await breaker.call(lambda: gateway.chat(make_call()))

    scenario = Scenario(
        name="F5 半开单探测", injected="http500", expected_code="UPSTREAM_ERROR"
    )
    assert breaker.state == "open", breaker.state

    # 上游恢复 + 冷却结束
    upstream.mode = "ok"
    now[0] += 6.0
    calls_before = upstream.calls

    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_chat() -> Any:
        """探测本体：先报"我进来了"，再停在半空，直到放行。"""

        entered.set()
        await release.wait()
        return await gateway.chat(make_call())

    probe = asyncio.create_task(breaker.call(gated_chat))
    await entered.wait()  # 此刻 _probing 已置位，但一次上游都还没打

    others = await asyncio.gather(
        *(breaker.call(lambda: gateway.chat(make_call())) for _ in range(4)),
        return_exceptions=True,
    )
    refused = sum(1 for item in others if isinstance(item, CircuitOpenError))
    during_flight = upstream.calls - calls_before  # 飞行期间打出的上游调用

    release.set()
    await probe
    probes = upstream.calls - calls_before

    scenario.calls = 5
    scenario.refused = refused
    scenario.ok = 5 - refused
    scenario.upstream_calls = probes
    scenario.upstream_error = "UPSTREAM_ERROR"  # 打开前那一轮报的
    scenario.last_refusal = "CIRCUIT_OPEN"
    scenario.breaker_state = breaker.state
    scenario.note = (
        f"探测飞行中来了 4 个请求，{refused} 个被快速拒绝、"
        f"期间上游调用 {during_flight} 次；探测成功后 {probes} 次上游调用、"
        f"状态={breaker.state}"
    )
    assert during_flight == 0, during_flight
    assert refused == 4, refused
    assert probes == 1, probes
    assert breaker.state == "closed", breaker.state
    return scenario


async def scenario_wired_breaker() -> Scenario:
    """F7：熔断器接在 `LLMGateway.chat()` 上（**线上形状**）。

    F1–F3 是"外面套一层"的形状。演练如果只证明这种形状，就等于证明了一种
    线上并不存在的调用方式——所以这里用真的接线（`breaker=` 传进 gateway）再跑一遍。

    它比 F1 少一层壳，多一个价值：如果哪天有人把接线拆了（比如重构 `chat()` 时
    忘了传 breaker），这条会红，而 F1 依然绿。
    """

    threshold, max_attempts, rounds = 2, 2, 6
    upstream = Upstream(mode="timeout")
    breaker = CircuitBreaker(failure_threshold=threshold, reset_after_s=30.0)
    gateway, chain = build_gateway(
        upstream, max_attempts=max_attempts, breaker=breaker
    )
    candidates = len(chain)

    scenario = Scenario(name="F7 接线后熔断", injected="timeout", expected_code="UPSTREAM_TIMEOUT")
    await drive(lambda: gateway.chat(make_call()), upstream, scenario, rounds=rounds)

    scenario.upstream_calls = upstream.calls
    scenario.breaker_state = breaker.state
    scenario.upstream_error = scenario.upstream_error or "UPSTREAM_TIMEOUT"

    assert scenario.refused == rounds - threshold, scenario.refused
    assert scenario.calls_after_open == 0, scenario.calls_after_open
    assert scenario.upstream_calls == threshold * candidates * max_attempts, (
        scenario.upstream_calls
    )
    scenario.note = (
        "breaker 传给 gateway（线上就是这条路径）；"
        f"打开后 {scenario.refused} 次 chat() 打出 {scenario.calls_after_open} 次上游调用"
    )
    return scenario


async def scenario_degradation_marks() -> Scenario:
    """降级必须留结构化标记：默认不兜底，开了兜底要能被识别成"不是真答案"。"""

    upstream = Upstream(mode="http500")
    gateway, _chain = build_gateway(upstream, max_attempts=1)
    strict = DegradationPolicy()
    lenient = DegradationPolicy(allow_canned=True)

    scenario = Scenario(
        name="F6 降级标记", injected="http500", expected_code="UPSTREAM_ERROR"
    )

    try:
        await gateway.chat(make_call())
    except LLMError as exc:
        scenario.upstream_error = error_code_for(exc)
        strict_result = await strict.fallback(make_call(), error=exc)
        lenient_result = await lenient.fallback(make_call(), error=exc)
    else:  # pragma: no cover - 注入 500 时不可能走到
        raise AssertionError("注入了 500 却没有抛错")

    assert not strict_result.answered, "默认策略不该凭空造答案"
    assert lenient_result.answered and lenient_result.answer is not None
    assert lenient_result.answer.outcome.degraded is True
    assert lenient_result.answer.outcome.response.finish_reason.value == "unknown"
    assert lenient_result.answer.outcome.response.raw_finish_reason == "degraded"

    scenario.calls = 1
    scenario.ok = 1 if lenient_result.answered else 0
    scenario.upstream_calls = upstream.calls
    scenario.breaker_state = "closed"  # 这个场景不过熔断器
    scenario.note = (
        "默认不兜底（answered=False）；开了兜底则 degraded=True + "
        "raw_finish_reason=degraded，下游能识别"
    )
    return scenario


async def main() -> int:
    parser = argparse.ArgumentParser(description="混沌演练（离线可复现）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    started = time.perf_counter()
    scenarios: list[Scenario] = []
    for runner in (
        scenario_model_timeout,
        scenario_tool_500,
        scenario_rate_limit,
        scenario_bad_request_never_trips,
        scenario_half_open_limits_probes,
        scenario_degradation_marks,
        scenario_wired_breaker,
    ):
        scenarios.append(await runner())
    elapsed = (time.perf_counter() - started) * 1000

    if args.json:
        print(json.dumps([item.as_dict() for item in scenarios], ensure_ascii=False, indent=1))
        return 0

    print("== 混沌演练（三类注入 + 四项自愈/边界性质）==\n")
    header = (
        f"{'场景':<16}{'注入':<11}{'code':<18}"
        f"{'上游调用':>8}{'拒绝':>6}{'拒绝后上游':>12}{'状态':>10}"
    )
    print(header)
    print("-" * 74)
    for item in scenarios:
        print(
            f"{item.name:<16}{item.injected:<11}{item.upstream_error:<18}"
            f"{item.upstream_calls:>8}{item.refused:>6}"
            f"{item.calls_after_open:>12}{item.breaker_state:>10}"
        )

    print()
    for item in scenarios:
        if item.note:
            print(f"- {item.name}: {item.note}")

    total_upstream = sum(item.upstream_calls for item in scenarios)
    print(f"\n共 {total_upstream} 次上游调用，全部来自注入传输层；耗时 {elapsed:.0f}ms")
    print("断言全绿：不崩、不放大、能恢复、降级有标记、故障域不跑偏")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""W11 D2：熔断器的边界——它**只**做"决定不再尝试"，不做重试。

这组用例守五条：

1. **阈值是"连续"失败**——成功一次就清零，不是"累计 N 次"；
2. **打开后不打上游**——`fn` 一次都不能被调用（这是省下调用，不是失败）；
3. **半开只放一个探测**，且探测失败立刻回到 open（不等阈值）；
4. **熔断器自己不重试**——`fn` 的调用次数恰好等于放行次数；
5. **故障域不跑偏**——我方请求错误（400）不计入、也不清零。

第 5 条最容易被写成"顺手也统计一下"，后果是"我们自己把参数写错"表现成
"上游被熔断"，值班去查厂商状态页，查不到任何东西。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agent_framework.llm.errors import (
    BadRequestError,
    CircuitOpenError,
    RateLimitError,
    UpstreamAuthError,
    UpstreamServerError,
    UpstreamTimeoutError,
)
from agent_framework.resilience.circuit_breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    counts_as_failure,
)
from gateway_helpers import StubClient, build_test_gateway, make_catalog, user_call


class Clock:
    """可推进的假时钟。真 sleep 会让测试套变慢，然后一定有人把它 skip 掉。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def boom() -> UpstreamServerError:
    return UpstreamServerError("注入的 500", provider="fake")


async def fail() -> None:
    raise boom()


async def ok() -> str:
    return "fine"


def make_breaker(
    *, threshold: int = 3, reset_after_s: float = 30.0
) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    return (
        CircuitBreaker(
            failure_threshold=threshold, reset_after_s=reset_after_s, clock=clock
        ),
        clock,
    )


# --------------------------------------------------------------------------- #
# 计数与阈值
# --------------------------------------------------------------------------- #


async def test_opens_only_after_threshold_consecutive_failures() -> None:
    breaker, _ = make_breaker(threshold=3)

    for _ in range(2):
        with pytest.raises(UpstreamServerError):
            await breaker.call(fail)
        assert breaker.state == CLOSED

    with pytest.raises(UpstreamServerError):
        await breaker.call(fail)

    assert breaker.state == OPEN


async def test_success_resets_the_consecutive_counter() -> None:
    """4 次 5xx 里夹一次成功，不该熔断——阈值是"连续"，不是"累计"。"""

    breaker, _ = make_breaker(threshold=3)

    for _ in range(4):
        for _ in range(2):
            with pytest.raises(UpstreamServerError):
                await breaker.call(fail)
        assert await breaker.call(ok) == "fine"

    assert breaker.state == CLOSED
    assert breaker.stats.failures == 0


async def test_open_breaker_never_calls_the_upstream() -> None:
    """打开之后 `fn` 一次都不能被调用。这是"省下调用"，与"又失败一次"完全不同。"""

    breaker, _ = make_breaker(threshold=2)
    calls = 0

    async def counted() -> None:
        nonlocal calls
        calls += 1
        raise boom()

    for _ in range(2):
        with pytest.raises(UpstreamServerError):
            await breaker.call(counted)
    assert calls == 2

    for _ in range(5):
        with pytest.raises(CircuitOpenError):
            await breaker.call(counted)

    assert calls == 2, "熔断打开后仍然打了上游，等于没熔断"
    assert breaker.stats.rejected == 5
    assert breaker.stats.failures == 2


async def test_breaker_never_retries_by_itself() -> None:
    """`fn` 的调用次数必须恰好等于放行次数。

    熔断器一旦自己也重试，一次故障就会被放大成 `threshold × max_attempts`——
    而日志里看起来只是"重试有点多"（ADR-003）。
    """

    breaker, _ = make_breaker(threshold=3)
    calls = 0

    async def counted() -> None:
        nonlocal calls
        calls += 1
        raise UpstreamTimeoutError("注入超时", provider="fake")

    for _ in range(10):
        with pytest.raises((UpstreamTimeoutError, CircuitOpenError)):
            await breaker.call(counted)

    assert calls == 3, calls


# --------------------------------------------------------------------------- #
# 状态迁移
# --------------------------------------------------------------------------- #


async def test_cooling_down_is_visible_on_state_read() -> None:
    """冷却结束应立刻反映到 `state` 上，而不是等下一次调用才纠正。"""

    breaker, clock = make_breaker(threshold=2, reset_after_s=30.0)
    for _ in range(2):
        with pytest.raises(UpstreamServerError):
            await breaker.call(fail)

    assert breaker.state == OPEN

    clock.advance(29.9)
    assert breaker.state == OPEN, "还没到冷却时间就放行，等于没有冷却"

    clock.advance(0.2)
    assert breaker.state == HALF_OPEN


async def test_half_open_admits_exactly_one_probe_in_flight() -> None:
    """探测还在飞行时，其余请求必须被快速拒绝（而不是排队等）。

    这里必须真的让探测"停住"：如果探测瞬间完成，状态已经回到 closed，
    后面的请求被放行是**正确**的——那是另一条性质，别混在一起测。
    """

    breaker, clock = make_breaker(threshold=2, reset_after_s=10.0)
    for _ in range(2):
        with pytest.raises(UpstreamServerError):
            await breaker.call(fail)
    clock.advance(11.0)

    entered = asyncio.Event()
    release = asyncio.Event()
    probes = 0

    async def probing() -> str:
        nonlocal probes
        probes += 1
        entered.set()
        await release.wait()
        return "probe-ok"

    task = asyncio.create_task(breaker.call(probing))
    await entered.wait()

    with pytest.raises(CircuitOpenError):
        await breaker.call(ok)
    with pytest.raises(CircuitOpenError):
        await breaker.call(ok)

    assert probes == 1

    release.set()
    assert await task == "probe-ok"
    assert breaker.state == CLOSED
    assert breaker.stats.rejected == 2


async def test_failed_probe_reopens_immediately_without_waiting_threshold() -> None:
    """半开阶段一次失败就够证明"上游还没好"，不等阈值、也不清零。"""

    breaker, clock = make_breaker(threshold=5, reset_after_s=10.0)
    for _ in range(5):
        with pytest.raises(UpstreamServerError):
            await breaker.call(fail)
    assert breaker.state == OPEN

    clock.advance(11.0)
    assert breaker.state == HALF_OPEN

    with pytest.raises(UpstreamServerError):
        await breaker.call(fail)

    assert breaker.state == OPEN, "探测失败却留在半开，会一直放行探测"
    # 重新计时：刚打开，冷却还没到
    assert breaker.state == OPEN

    clock.advance(11.0)
    assert breaker.state == HALF_OPEN


async def test_success_after_recovery_clears_everything() -> None:
    breaker, clock = make_breaker(threshold=2, reset_after_s=5.0)
    for _ in range(2):
        with pytest.raises(UpstreamServerError):
            await breaker.call(fail)

    clock.advance(6.0)
    assert await breaker.call(ok) == "fine"

    assert breaker.state == CLOSED
    assert breaker.stats.failures == 0
    assert breaker.stats.trips == 1


# --------------------------------------------------------------------------- #
# 故障域：哪些错误算"上游不健康"
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc",
    [
        UpstreamTimeoutError("t", provider="fake"),
        UpstreamServerError("s", provider="fake"),
        RateLimitError("r", provider="fake"),
        UpstreamAuthError("a", provider="fake"),
    ],
)
def test_upstream_failures_are_counted(exc: Exception) -> None:
    assert counts_as_failure(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        BadRequestError("b", provider="fake"),
        CircuitOpenError("c"),
        ValueError("v"),
    ],
)
def test_our_own_mistakes_are_not_counted(exc: Exception) -> None:
    assert counts_as_failure(exc) is False


async def test_bad_request_neither_counts_nor_clears() -> None:
    """400 不计入、也**不**清零。

    清零的话，"每 4 次 5xx 夹 1 次参数错"就永远熔不断——而故障域看起来完全正常。
    """

    breaker, _ = make_breaker(threshold=3)

    for _ in range(2):
        with pytest.raises(UpstreamServerError):
            await breaker.call(fail)
    assert breaker.stats.failures == 2

    with pytest.raises(BadRequestError):
        await breaker.call(lambda: _raise(BadRequestError("参数错", provider="fake")))

    assert breaker.state == CLOSED
    assert breaker.stats.failures == 2, "我方错误不该清零上游的失败计数"

    with pytest.raises(UpstreamServerError):
        await breaker.call(fail)
    assert breaker.state == OPEN


async def _raise(exc: Exception) -> None:
    raise exc


# --------------------------------------------------------------------------- #
# 手动干预
# --------------------------------------------------------------------------- #


async def test_manual_reset_and_force_open() -> None:
    breaker, _ = make_breaker(threshold=10)

    breaker.force_open()
    assert breaker.state == OPEN
    with pytest.raises(CircuitOpenError):
        await breaker.call(ok)

    breaker.reset()
    assert breaker.state == CLOSED
    assert breaker.stats.failures == 0
    assert await breaker.call(ok) == "fine"


async def test_stats_keep_rejected_apart_from_failures() -> None:
    """`rejected` 与 `failures` 合成一个数，会让"熔断生效"看起来像"故障变多"。"""

    breaker, _ = make_breaker(threshold=1)
    with pytest.raises(UpstreamServerError):
        await breaker.call(fail)

    for _ in range(3):
        with pytest.raises(CircuitOpenError):
            await breaker.call(ok)

    stats = breaker.stats
    assert stats.state == OPEN
    assert stats.failures == 1
    assert stats.rejected == 3
    assert stats.trips == 1
    assert stats.last_error == "UpstreamServerError"


# --------------------------------------------------------------------------- #
# 接线：熔断器真的挂在 `LLMGateway.chat()` 上
# --------------------------------------------------------------------------- #


def _always_failing_client(count: int) -> tuple[Any, StubClient]:
    catalog = make_catalog(["primary", "fb"])
    stub = StubClient(
        chat_results=[UpstreamServerError("注入 500", provider="stub")] * count
    )
    return catalog, stub


async def test_gateway_without_breaker_behaves_exactly_as_before() -> None:
    """不传熔断器 = W10 行为。加了保护不能变成"改了所有既有行为"。"""

    catalog, stub = _always_failing_client(10)
    gateway = build_test_gateway(catalog, {"primary": stub})

    for _ in range(3):
        with pytest.raises(UpstreamServerError):
            await gateway.chat(user_call())

    # 1 个候选 × max_attempts(3) = 每次 3 次 adapter 调用
    assert stub.chat_calls == 9


async def test_gateway_with_breaker_stops_reaching_adapters_once_open() -> None:
    """接线后的核心性质：打开之后 adapter 的调用次数**不再增长**。

    这条比"熔断器自己好用"更重要——一个没接上去的熔断器在单元测试里永远是对的。
    """

    catalog, stub = _always_failing_client(1000)
    breaker, _ = make_breaker(threshold=2)
    gateway = build_test_gateway(catalog, {"primary": stub}, breaker=breaker)

    for _ in range(2):
        with pytest.raises(UpstreamServerError):
            await gateway.chat(user_call())

    calls_when_opened = stub.chat_calls
    assert calls_when_opened > 0, "故障没真的打出去，这条用例什么都证明不了"
    assert breaker.state == OPEN

    for _ in range(4):
        with pytest.raises(CircuitOpenError):
            await gateway.chat(user_call())

    assert stub.chat_calls == calls_when_opened
    assert breaker.stats.rejected == 4


async def test_circuit_open_maps_to_503_not_500() -> None:
    """对外必须是 503 + `CIRCUIT_OPEN`：它和 `SERVER_BUSY` 的处置动作相反。"""

    from agent_framework.llm.errors import ERROR_HTTP, error_code_for, http_status_for

    assert http_status_for(CircuitOpenError("x")) == 503
    assert error_code_for(CircuitOpenError("x")) == "CIRCUIT_OPEN"
    assert ERROR_HTTP[CircuitOpenError] == (503, "CIRCUIT_OPEN")
    # 不能和 SERVER_BUSY 混：一个是"看上游"，一个是"扩自己的并发"
    assert error_code_for(CircuitOpenError("x")) != "SERVER_BUSY"


def test_circuit_open_surfaces_as_503_with_retry_after() -> None:
    """一路走到 HTTP：503 + `CIRCUIT_OPEN` + `Retry-After`。

    `Retry-After` 必须透传——它是**调用方**该退避多久的唯一依据；
    我们这边已经在熔断了，再让调用方盲目重试只会把半开探测的那一个名额挤满。
    """

    from agent_framework.server.errors import install_error_handlers
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom_route() -> None:
        raise CircuitOpenError("熔断打开中")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "CIRCUIT_OPEN"
    assert body["request_id"], "错误体里必须有 request_id，否则排障只能靠时间对齐"
    assert response.headers["Retry-After"] == "5"

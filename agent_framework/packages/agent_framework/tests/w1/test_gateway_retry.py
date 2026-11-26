"""重试归属与 fallback（A9 + 教程 §9 的 test_gateway_retry）。

四条纪律：

1. 重试**只在 gateway**（ADR-003）：adapter 里出现 sleep/retry 就要打回；
2. 重试谓词只依赖 `error.retryable`——再写一遍 `if status == 429` 就是双份真相；
3. 非可重试错误立即切下一个候选，全链只绕一圈；
4. 一次成功调用**只产出一条** RequestLog（候选失败走独立 WARNING）。
"""

from __future__ import annotations

import json
import logging

import pytest
from agent_framework.llm.base import Capability
from agent_framework.llm.errors import (
    BadRequestError,
    CapabilityError,
    RateLimitError,
    UpstreamAuthError,
)
from agent_framework.llm.providers import OpenAICompatClient
from agent_framework.llm.types import ToolSpec
from conftest import load_case
from gateway_helpers import (
    RecordingSleep,
    SequenceTransport,
    StubClient,
    build_test_gateway,
    json_response,
    make_catalog,
    make_settings,
    text_response,
    user_call,
)


def request_logs(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "agent_framework.request"
    ]


# --------------------------------------------------------------------------- #
# 编排层的重试
# --------------------------------------------------------------------------- #


async def test_retries_rate_limit_then_succeeds() -> None:
    client = StubClient(
        "primary",
        chat_results=[
            RateLimitError("slow down", provider="primary"),
            RateLimitError("slow down", provider="primary"),
            text_response("done"),
        ],
    )
    sleep = RecordingSleep()
    gateway = build_test_gateway(
        make_catalog(["primary"]), {"primary": client}, sleep=sleep
    )

    outcome = await gateway.chat(user_call())

    assert outcome.response.text == "done"
    assert outcome.retry_count == 2
    assert client.chat_calls == 3

    # 断言的是"退避序列"，而不是真睡 0.5 + 1.0 秒。
    assert len(sleep.delays) == 2
    assert 0.25 <= sleep.delays[0] <= 0.75
    assert 0.5 <= sleep.delays[1] <= 1.5


async def test_auth_error_switches_candidate_without_retry() -> None:
    """不可重试错误立刻换候选，且**不能**先睡一轮。"""

    primary = StubClient(
        "primary", chat_results=[UpstreamAuthError("bad key", provider="primary")]
    )
    backup = StubClient("backup", chat_results=[text_response("from backup")])
    sleep = RecordingSleep()
    gateway = build_test_gateway(
        make_catalog(["primary", "backup"]),
        {"primary": primary, "backup": backup},
        sleep=sleep,
    )

    outcome = await gateway.chat(user_call())

    assert outcome.alias == "backup"
    assert outcome.fallback_from == "primary"
    assert outcome.retry_count == 0
    assert primary.chat_calls == 1
    assert sleep.delays == []


async def test_all_candidates_fail_raises_last_error() -> None:
    primary = StubClient(
        "primary", chat_results=[UpstreamAuthError("bad key", provider="primary")]
    )
    backup = StubClient(
        "backup", chat_results=[BadRequestError("bad params", provider="backup")]
    )
    gateway = build_test_gateway(
        make_catalog(["primary", "backup"]),
        {"primary": primary, "backup": backup},
    )

    with pytest.raises(BadRequestError):
        await gateway.chat(user_call())


async def test_exhausted_attempts_move_to_next_candidate() -> None:
    """重试次数用尽 → 换候选，不回头、不死循环。"""

    primary = StubClient(
        "primary",
        chat_results=[
            RateLimitError("429", provider="primary"),
            RateLimitError("429", provider="primary"),
            RateLimitError("429", provider="primary"),
        ],
    )
    backup = StubClient("backup", chat_results=[text_response("ok")])
    gateway = build_test_gateway(
        make_catalog(["primary", "backup"]),
        {"primary": primary, "backup": backup},
        settings=make_settings(max_attempts=3),
    )

    outcome = await gateway.chat(user_call())

    assert outcome.alias == "backup"
    assert primary.chat_calls == 3


async def test_unknown_alias_from_caller_is_bad_request() -> None:
    """`call.model` 是调用方输入：拼错应是 400，不该让值班同学收到 500 告警。"""

    client = StubClient("primary", chat_results=[text_response("never")])
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    with pytest.raises(BadRequestError):
        await gateway.chat(user_call(model="ghost"))

    assert client.chat_calls == 0


# --------------------------------------------------------------------------- #
# 能力校验：硬需求抛异常，且不花钱
# --------------------------------------------------------------------------- #


async def test_tools_without_capability_raises_before_calling() -> None:
    client = StubClient("primary", capabilities=frozenset({Capability.STREAM}))
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    with pytest.raises(CapabilityError):
        await gateway.chat(user_call(tools=[ToolSpec(name="search", parameters={})]))

    assert client.chat_calls == 0


async def test_stream_without_capability_raises_in_precheck() -> None:
    client = StubClient("primary", capabilities=frozenset({Capability.TOOLS}))
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    with pytest.raises(CapabilityError):
        gateway.precheck(user_call(), stream=True)


async def test_precheck_returns_alias_and_calls_nothing() -> None:
    client = StubClient("primary")
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    assert gateway.precheck(user_call()) == "primary"
    assert client.chat_calls == 0
    assert client.stream_calls == 0


# --------------------------------------------------------------------------- #
# A9：重试归属 —— adapter 只发 HTTP，重试在 gateway
# --------------------------------------------------------------------------- #


async def test_a9_retry_lives_in_gateway_not_adapter() -> None:
    """两次 429 后成功 ⇒ adapter 的 HTTP 次数 == 3，且 gateway 记录了 2 次退避。"""

    rate_limited = load_case("openai", "error_rate_limit")
    success = load_case("openai", "text_basic")

    transport = SequenceTransport(
        [
            lambda: json_response(rate_limited.response, status_code=429),
            lambda: json_response(rate_limited.response, status_code=429),
            lambda: json_response(success.response),
        ]
    )
    client = OpenAICompatClient(
        name="primary",
        model="gpt-test",
        base_url="https://api.openai.test/v1",
        api_key="sk-test",
        transport=transport.transport(),
    )
    sleep = RecordingSleep()
    gateway = build_test_gateway(
        make_catalog(["primary"]), {"primary": client}, sleep=sleep
    )

    outcome = await gateway.chat(user_call())

    assert outcome.response.text == success.expect["text"]
    assert outcome.retry_count == 2
    # 计数在 transport 上：adapter 每被调一次只发一次 HTTP。
    assert transport.count == 3
    assert len(sleep.delays) == 2


async def test_a9_adapter_has_no_retry_of_its_own() -> None:
    """单次调用只发一次 HTTP —— 这是"adapter 不重试"的可执行证据。"""

    success = load_case("openai", "text_basic")
    transport = SequenceTransport([lambda: json_response(success.response)])
    client = OpenAICompatClient(
        name="primary",
        model="gpt-test",
        base_url="https://api.openai.test/v1",
        api_key="sk-test",
        transport=transport.transport(),
    )

    await client.chat(user_call().messages)

    assert transport.count == 1


# --------------------------------------------------------------------------- #
# 日志：一条请求一条记录
# --------------------------------------------------------------------------- #


async def test_one_request_log_per_successful_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = StubClient(
        "primary",
        chat_results=[
            RateLimitError("slow", provider="primary"),
            text_response("done"),
        ],
    )
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    with caplog.at_level(logging.INFO, logger="agent_framework.request"):
        await gateway.chat(user_call())

    logs = request_logs(caplog)
    assert len(logs) == 1

    entry = logs[0]
    assert entry["model"] == "primary"
    assert entry["retry_count"] == 1
    assert entry["error_code"] is None
    assert float(entry["cost"]) > 0  # type: ignore[arg-type]
    assert entry["prompt_tokens"] == 10
    assert entry["completion_tokens"] == 5
    assert entry["stream"] is False


async def test_failed_request_logs_error_code_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = StubClient(
        "primary", chat_results=[UpstreamAuthError("bad key", provider="primary")]
    )
    gateway = build_test_gateway(make_catalog(["primary"]), {"primary": client})

    with (
        caplog.at_level(logging.INFO, logger="agent_framework.request"),
        pytest.raises(UpstreamAuthError),
    ):
        await gateway.chat(user_call())

    logs = request_logs(caplog)
    assert len(logs) == 1
    assert logs[0]["error_code"] == "UPSTREAM_AUTH"
    assert logs[0]["finish_reason"] == "error"


async def test_candidate_failure_is_warning_not_request_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """候选失败单独记 WARNING：混进 RequestLog 会让请求量与成本翻倍。"""

    primary = StubClient(
        "primary", chat_results=[UpstreamAuthError("bad key", provider="primary")]
    )
    backup = StubClient("backup", chat_results=[text_response("ok")])
    gateway = build_test_gateway(
        make_catalog(["primary", "backup"]),
        {"primary": primary, "backup": backup},
    )

    with (
        caplog.at_level(logging.WARNING, logger="agent_framework.llm.candidate"),
        caplog.at_level(logging.INFO, logger="agent_framework.request"),
    ):
        await gateway.chat(user_call())

    assert len(request_logs(caplog)) == 1
    warnings = [
        record for record in caplog.records if record.name == "agent_framework.llm.candidate"
    ]
    assert len(warnings) == 1
    assert "primary" in warnings[0].getMessage()

"""真机用例（`-m live`）：只有需要验证"我们对协议的理解"时才跑。

为什么默认不跑：一次全量 `pytest` 就会真调四家 API，CI 里没有 key 会红、
有 key 会烧钱，最后大家都学会忽略红叉。

这四条各自守着一条"手写夹具永远测不出来"的东西：

1. `usage` 真的回来了（`stream_options.include_usage` 不是摆设，不发就是全 0 成本）；
2. 首 token 在 `first_token_timeout_s` 之内（10s 这个默认值必须被真机校准过）；
3. 真实模型的 tool_call 落在我们声明的 schema 内（A14 的真机版）；
4. 文本路径真的能通（否则前三条都无意义）。

用法：
    uv run pytest -m live --basetemp=.pytest_tmp -q
    uv run pytest -m live -k deepseek -s      # 只跑一家，看首 token 实测值
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import pytest
from agent_framework.config import ConfigError, Settings, load_catalog
from agent_framework.llm.base import LLMClient
from agent_framework.llm.registry import Registry
from agent_framework.llm.types import FinishReason, StreamEventType
from canonical_input import (
    MAX_TOKENS,
    TOOL_NAME,
    TOOL_PARAMETERS,
    canonical_messages,
    canonical_tools,
)

pytestmark = pytest.mark.live

LIVE_PROVIDERS: tuple[str, ...] = ("openai", "deepseek", "qwen", "anthropic")


@pytest.fixture(scope="session", autouse=True)
def require_explicit_marker(pytestconfig: pytest.Config) -> None:
    """只认显式 `-m live`。

    否则 `uv run pytest` 会把真机用例当普通用例跑——CI 里没 key 是红，
    有 key 就是每次提交都在花钱，两种结果都不对。
    """

    marker_expr = pytestconfig.getoption("-m") or ""

    if "live" not in marker_expr:
        pytest.skip("真机用例需显式执行：uv run pytest -m live")


def resolve_alias(provider: str, settings: Settings) -> str:
    catalog = load_catalog(settings.models_yaml)

    for alias, model in catalog.models.items():
        if model.provider == provider:
            return str(alias)

    pytest.skip(f"models.yaml 里没有 provider '{provider}'")


@pytest.fixture
async def adapter(request: pytest.FixtureRequest) -> AsyncIterator[LLMClient]:
    provider = request.param
    settings = Settings()
    catalog = load_catalog(settings.models_yaml)
    alias = resolve_alias(provider, settings)
    provider_cfg = catalog.get_provider(alias)[1]

    try:
        settings.api_key(provider_cfg.api_key_env)
    except ConfigError as exc:
        pytest.skip(f"缺少凭据：{exc}")

    registry = Registry(catalog, settings)

    try:
        yield registry.get(alias)
    finally:
        await registry.aclose()


live_adapter = pytest.mark.parametrize("adapter", LIVE_PROVIDERS, indirect=True)


@live_adapter
async def test_live_text_returns_usage(adapter: LLMClient) -> None:
    response = await adapter.chat(
        canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
    )

    assert (response.text or "").strip(), f"{adapter.name}: 真实调用返回空文本"
    assert response.usage.estimated is False, "真实厂商应当返回 usage"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.finish_reason in {
        FinishReason.STOP,
        FinishReason.LENGTH,
        FinishReason.TOOL_CALLS,
    }


@live_adapter
async def test_live_stream_returns_usage(adapter: LLMClient) -> None:
    """不发 `stream_options.include_usage` 就没有 usage —— 成本会恒为 0 且不报错。"""

    seen_text = False
    usage = None

    async for event in adapter.stream(
        canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
    ):
        if event.type is StreamEventType.TEXT_DELTA:
            seen_text = True

        if event.usage is not None:
            usage = event.usage

    assert seen_text, f"{adapter.name}: 流式没有产出 text_delta"
    assert usage is not None, f"{adapter.name}: 流式结束仍无 usage（成本将恒为 0）"
    assert usage.estimated is False


@live_adapter
async def test_live_first_token_within_timeout(adapter: LLMClient) -> None:
    """这里用 `perf_counter` 计时是**为了校准阈值**，不是另造指标。

    生产口径仍然是 `RequestLog.first_token_ms`（由 gateway 产出）。
    """

    timeout_s = Settings().first_token_timeout_s
    started = time.perf_counter()
    first_token_ms: float | None = None

    async for event in adapter.stream(
        canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
    ):
        if event.type in (StreamEventType.TEXT_DELTA, StreamEventType.TOOL_CALL):
            first_token_ms = (time.perf_counter() - started) * 1000
            break

    assert first_token_ms is not None, f"{adapter.name}: 没有首个 token"
    assert first_token_ms < timeout_s * 1000, (
        f"{adapter.name}: 首 token {first_token_ms:.0f}ms 超过阈值 {timeout_s}s；"
        "要么调大 first_token_timeout_s，要么换模型"
    )


@live_adapter
async def test_live_tool_call_matches_declared_schema(adapter: LLMClient) -> None:
    """真实模型会跑偏：调错工具名、凭空加参数。跑偏要在这里红，不是放宽断言。"""

    response = await adapter.chat(
        canonical_messages(),
        canonical_tools(),
        max_tokens=MAX_TOKENS,
    )

    assert response.tool_calls, f"{adapter.name}: 给了 tools 但真实模型没有调用工具"

    allowed = set((TOOL_PARAMETERS.get("properties") or {}).keys())
    required = set(TOOL_PARAMETERS.get("required") or [])

    for call in response.tool_calls:
        assert call.name == TOOL_NAME, (
            f"{adapter.name}: 调用了未声明的工具 '{call.name}'"
        )
        assert call.arguments_error is None, f"{adapter.name}: 参数不是合法 JSON"
        assert set(call.arguments) <= allowed
        assert required <= set(call.arguments)

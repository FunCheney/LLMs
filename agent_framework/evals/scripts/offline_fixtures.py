"""离线档的网关工厂（W3 抽出，`run_smoke` 与 `run_eval` 共用）。

抽出来的唯一理由是**避免两份实现漂移**：W2 的 `run_smoke.py` 里原本有一份，
W3 的 `run_eval.py` 再抄一份，某天有人改了退避参数只改一处，两条命令对同一批夹具
给出不同的重试次数——而两者的报告看起来都"正常"。

这里只做三件事，且都以"不联网"为前提：
1. 用 `MockLLMClient` 回放夹具（走真实 adapter 的 parse / assembler）；
2. 错误夹具用 `httpx.MockTransport` 直接返回原始 bytes（错误映射走真实薄壳）；
3. 用 `_SingleClientRegistry` 绕开 `Registry` 读 key 的行为——离线档一个 key 都不该需要。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from agent_framework.config import Catalog, Settings, load_catalog
from agent_framework.llm.client import LLMGateway
from agent_framework.llm.providers.anthropic import AnthropicClient
from agent_framework.llm.providers.mock import MockLLMClient
from agent_framework.llm.providers.openai_compat import OpenAICompatClient
from agent_framework.llm.router import StaticRouter

__all__ = [
    "FIXTURES_ROOT",
    "OFFLINE_ALIAS",
    "build_offline_gateway",
    "delegate_adapter",
    "error_client",
]

FIXTURES_ROOT = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "agent_framework"
    / "tests"
    / "fixtures"
)

OFFLINE_ALIAS = "fixture"


class SingleClientRegistry:
    """最小的 Registry 替身：任何 alias 都返回同一个 client。

    刻意不实现完整接口：离线档只有一个别名 `fixture`，多实现的每个方法都是
    "看起来可以但没人验证过"的死代码。
    """

    def __init__(self, catalog: Catalog, client: Any) -> None:
        self._catalog = catalog
        self._client = client

    def get(self, alias: str) -> Any:
        return self._client

    def get_model(self, alias: str) -> Any:
        return self._catalog.get_model(alias)

    async def aclose(self) -> None:
        await self._client.aclose()


def delegate_adapter(provider: str) -> Any:
    """回放时包裹的真 adapter。

    必须用真 adapter：mock 只喂 wire 原文，parse/assemble 由它们做，
    否则"自己跟自己比对"，真机解析 bug 一个都测不出来（假绿的头号来源）。
    """

    if provider == "anthropic":
        return AnthropicClient(
            name="anthropic",
            model="claude-test",
            base_url="https://api.anthropic.test",
            api_key="offline",
            default_max_tokens=1024,
        )

    return OpenAICompatClient(
        name=provider,
        model=f"{provider}-test",
        base_url=f"https://{provider}.test/v1",
        api_key="offline",
    )


def error_client(provider: str, case_name: str) -> Any:
    """错误夹具：把 `response.json` + `meta.status_code` 原样喂给真 adapter。

    走 MockTransport 而不是伪造异常，是为了让"状态码 → 异常类 → 错误码"这条链
    真的跑一遍。直接 `raise RateLimitError` 只能测到调用方，测不到映射表。
    """

    case_dir = FIXTURES_ROOT / provider / case_name
    meta = json.loads((case_dir / "meta.json").read_text("utf-8"))
    body = (case_dir / "response.json").read_text("utf-8")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            int(meta.get("status_code", 500)),
            content=body.encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    transport = httpx.MockTransport(handler)

    if provider == "anthropic":
        return AnthropicClient(
            name="anthropic",
            model="claude-test",
            base_url="https://api.anthropic.test",
            api_key="offline",
            default_max_tokens=1024,
            transport=transport,
        )

    return OpenAICompatClient(
        name=provider,
        model=f"{provider}-test",
        base_url=f"https://{provider}.test/v1",
        api_key="offline",
        transport=transport,
    )


def build_offline_gateway(
    *,
    provider: str,
    fixture_case: str,
    expect_error: bool = False,
    settings: Settings | None = None,
    catalog: Catalog | None = None,
) -> LLMGateway:
    """按夹具构造一个离线网关。`expect_error=True` 时走错误夹具。"""

    resolved_catalog = catalog or load_catalog()

    if settings is None:
        # 退避压到 1ms：离线档不该真等，否则一条 429 用例就让 CI 多花 1.5 秒。
        # 注意 `_env_file=None`——测试机上的 .env.local 不该影响离线档的数字。
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None, backoff_base_s=0.001, backoff_cap_s=0.002
        )

    client: Any = (
        error_client(provider, fixture_case)
        if expect_error
        else MockLLMClient(
            delegate_adapter(provider),
            fixtures_root=FIXTURES_ROOT,
            case=fixture_case,
        )
    )

    return LLMGateway(
        registry=SingleClientRegistry(resolved_catalog, client),  # type: ignore[arg-type]
        router=StaticRouter([OFFLINE_ALIAS]),
        catalog=resolved_catalog,
        settings=settings,
    )

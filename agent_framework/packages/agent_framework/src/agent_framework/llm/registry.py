"""alias → adapter 工厂（architecture §4.1 / §8）。

三条约束：

1. **懒加载**：`__init__` 里不建任何 httpx client。写在构造里会导致"启动就连 4 家"，
   离线测试与 CI 直接挂掉。
2. **凭据缺失在这里炸，不在启动时炸**（config.py 的纪律 2）。
3. **厂商差异只来自 models.yaml 与 adapter flags**，这里不出现 `if provider == "qwen"`。
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from pydantic import ValidationError

from ..config import Catalog, ConfigError, ModelConfig, ProviderConfig, Settings
from ..observability import mask_secret
from .base import LLMClient
from .providers.anthropic import AnthropicClient
from .providers.mock import MockLLMClient, ReplayAdapter
from .providers.openai_compat import (
    PROVIDER_FLAGS,
    OpenAICompatClient,
    OpenAICompatFlags,
)

_LOG = logging.getLogger("agent_framework.registry")

_FALLBACK_ANTHROPIC_URL = "https://api.anthropic.com"


class Registry:
    """按 alias 造 adapter 并缓存。"""

    def __init__(
        self,
        catalog: Catalog,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._catalog = catalog
        self._settings = settings
        self._clients: dict[str, LLMClient] = {}

        # 注入点只有一个用途：把真 adapter 的流量接进录制器 / 测试替身。
        # 生产路径必须为 None —— 有它才能"录一份真机流量"，而不用改 adapter 一行。
        self._transport = transport

        # mock 的 delegate 是"借来解析夹具"的真 adapter，也要在 aclose 时释放。
        self._extra: list[LLMClient] = []

    # -- 查询 ------------------------------------------------------------ #

    def get_model(self, alias: str) -> ModelConfig:
        return self._catalog.get_model(alias)

    @property
    def created(self) -> tuple[str, ...]:
        """已经实例化的 alias（测试用来验证懒加载）。"""

        return tuple(self._clients)

    def get(self, alias: str) -> LLMClient:
        client = self._clients.get(alias)

        if client is None:
            client = self._build(alias)
            self._clients[alias] = client

        return client

    # -- 构造 ------------------------------------------------------------ #

    def _build(self, alias: str) -> LLMClient:
        model = self._catalog.get_model(alias)
        provider_name, provider = self._catalog.get_provider(alias)

        match provider.type:
            case "openai_compat":
                client: LLMClient = self._build_openai_compat(
                    alias, provider_name, provider, model
                )
            case "anthropic":
                client = self._build_anthropic(alias, provider_name, provider, model)
            case "mock":
                client = self._build_mock(alias, provider, model)
            case _:  # pragma: no cover - Literal 已限制取值，兜底防未来新增类型忘记实现
                raise ConfigError(f"provider '{provider_name}' 的 type 无法识别")

        _LOG.info(
            "built adapter alias=%s provider=%s type=%s wire_model=%s",
            alias,
            provider_name,
            provider.type,
            model.model or "<empty>",
        )

        return client

    def _build_openai_compat(
        self,
        alias: str,
        provider_name: str,
        provider: ProviderConfig,
        model: ModelConfig,
    ) -> LLMClient:
        if not provider.base_url:
            raise ConfigError(f"provider '{provider_name}' 缺少 base_url")

        api_key = self._settings.api_key(provider.api_key_env)

        _LOG.debug("credential provider=%s key=%s", provider_name, mask_secret(api_key))

        return OpenAICompatClient(
            name=provider_name,
            model=model.model,
            base_url=provider.base_url,
            api_key=api_key,
            flags=self._flags(provider_name, provider, model),
            timeout=self._timeout(provider),
            transport=self._transport,
        )

    def _build_anthropic(
        self,
        alias: str,
        provider_name: str,
        provider: ProviderConfig,
        model: ModelConfig,
    ) -> LLMClient:
        api_key = self._settings.api_key(provider.api_key_env)

        _LOG.debug("credential provider=%s key=%s", provider_name, mask_secret(api_key))

        return AnthropicClient(
            name=provider_name,
            model=model.model,
            base_url=provider.base_url or _FALLBACK_ANTHROPIC_URL,
            api_key=api_key,
            # Anthropic 的 max_tokens 必填：配置里配好，缺了就由 adapter 报 400。
            default_max_tokens=model.max_tokens,
            timeout=self._timeout(provider),
            transport=self._transport,
        )

    def _build_mock(
        self,
        alias: str,
        provider: ProviderConfig,
        model: ModelConfig,
    ) -> LLMClient:
        """离线回放：借一个真 adapter 的解析逻辑，只把 wire 数据换成夹具。

        `MockLLMClient` 不自己构造 LLMResponse（那是"自己跟自己对答案"），
        所以这里必须给出一个真 adapter —— 它只调用 parse_response / make_assembler，
        因此假 key、不可达 base_url 都不影响。
        """

        flags = provider.flags or {}
        delegate_name = str(flags.get("delegate", "openai"))
        case = str(flags.get("case", "text_basic"))
        raw_root = flags.get("fixtures_root")

        if not raw_root:
            raise ConfigError(
                f"provider '{alias}' 是 mock 类型，但 flags 里缺少 fixtures_root"
            )

        root = Path(str(raw_root))

        if not root.is_absolute():
            # 相对路径按仓库根解析（models.yaml 里的路径都是人读的，不是包内路径）。
            root = Path.cwd() / root

        if not (root / delegate_name / case).is_dir():
            raise ConfigError(
                f"夹具不存在：{root / delegate_name / case}；"
                f"请检查 mock provider 的 fixtures_root / case"
            )

        delegate = self._delegate(delegate_name, model)

        return MockLLMClient(delegate, fixtures_root=root, case=case)

    def _delegate(self, provider_name: str, model: ModelConfig) -> ReplayAdapter:
        provider = self._catalog.providers.get(provider_name)

        if provider is None:
            raise ConfigError(f"mock 的 delegate '{provider_name}' 不是已知 provider")

        if provider.type == "anthropic":
            client: AnthropicClient | OpenAICompatClient = AnthropicClient(
                name=provider_name,
                model=model.model,
                base_url=provider.base_url or _FALLBACK_ANTHROPIC_URL,
                api_key="fixture",
                default_max_tokens=model.max_tokens or 1024,
                timeout=self._timeout(provider),
            )
        else:
            client = OpenAICompatClient(
                name=provider_name,
                model=model.model,
                base_url=provider.base_url or "http://localhost",
                api_key="fixture",
                flags=self._flags(provider_name, provider, model),
                timeout=self._timeout(provider),
            )

        self._extra.append(client)

        return client

    # -- 辅助 ------------------------------------------------------------ #

    def _timeout(self, provider: ProviderConfig) -> float:
        return provider.timeout_s or self._settings.request_timeout_s

    def _flags(
        self,
        provider_name: str,
        provider: ProviderConfig,
        model: ModelConfig,
    ) -> OpenAICompatFlags:
        """三层合并：内置预设 → yaml 覆盖 → 按模型覆盖。

        第三层是必须的：Qwen 同一 provider 下 3.7 不能回传 reasoning、3.8 必须回传，
        放在 provider 级会必然写错一半。
        """

        base = PROVIDER_FLAGS.get(provider_name) or OpenAICompatFlags()
        merged = base.model_dump()
        merged.update(provider.flags or {})

        if model.reasoning_replay is not None:
            merged["reasoning_replay_override"] = model.reasoning_replay

        try:
            return OpenAICompatFlags(**merged)
        except ValidationError as exc:
            raise ConfigError(
                f"provider '{provider_name}' 的 flags 非法：{exc}"
            ) from exc

    # -- 生命周期 -------------------------------------------------------- #

    async def aclose(self) -> None:
        """关闭所有已建的 httpx client。

        只关 gateway 不管 registry，W2 压测时会出现 `Too many open files`——
        这类泄漏在功能测试里永远看不到。
        """

        for client in (*self._clients.values(), *self._extra):
            await client.aclose()

        self._clients.clear()
        self._extra.clear()

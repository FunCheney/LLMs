"""Day4 链路层测试的共用构造物。

为什么单开一个模块而不是塞进 conftest：adapter 测试（A1–A8）只需要"夹具 + MockTransport"，
链路层测试需要另外三样东西——**假 Registry**（绕开 credential 读取）、
**按序回放的 transport**（构造"两次 429 后成功"）、**可脚本化的 adapter**（构造"提交后报错"）。
混进 conftest 会让 adapter 测试背上一堆用不到的依赖。

这里的每个替身都刻意做成"不继承真实现"：
`FakeRegistry` 不继承 `Registry`（否则拖进 api_key 读取与 httpx client 构造），
`StubClient` 不继承真 adapter（gateway 的职责是编排，与厂商协议无关）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import httpx
from agent_framework.config import (
    Catalog,
    ModelConfig,
    Pricing,
    ProviderConfig,
    RouteConfig,
    Settings,
)
from agent_framework.llm.base import Capability, LLMClient
from agent_framework.llm.client import ChatCall, LLMGateway
from agent_framework.llm.router import StaticRouter
from agent_framework.llm.types import (
    FinishReason,
    LLMResponse,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolChoice,
    ToolSpec,
    Usage,
)

# --------------------------------------------------------------------------- #
# 配置与 gateway
# --------------------------------------------------------------------------- #


def make_settings(**overrides: Any) -> Settings:
    """不读 `.env.local`、不进真实网络的 Settings。

    `_env_file=None` 是必需的：否则测试会读到开发者本机的密钥，
    出现"我这儿过、CI 不过"的经典问题。
    """

    params: dict[str, Any] = {
        "_env_file": None,
        "max_attempts": 3,
        "backoff_base_s": 0.5,
        "backoff_cap_s": 8.0,
        "first_token_timeout_s": 10.0,
        "request_timeout_s": 5.0,
    }
    params.update(overrides)

    return Settings(**params)


def make_catalog(
    aliases: Sequence[str],
    *,
    provider: str = "openai",
    tier: str = "cheap",
    task: str = "chat",
) -> Catalog:
    """造一份最小可用 catalog：第一个 alias 是 primary，其余是 fallback。"""

    if not aliases:
        raise ValueError("至少给一个 alias")

    models = {
        alias: ModelConfig(
            provider=provider,
            model=f"{alias}-wire",
            tier=tier,
            pricing=Pricing(input=1.0, output=2.0, currency="USD"),
        )
        for alias in aliases
    }
    route = RouteConfig(primary=aliases[0], fallback=list(aliases[1:]))

    return Catalog(
        providers={
            provider: ProviderConfig(
                type="openai_compat", base_url="https://example.test/v1"
            )
        },
        models=models,
        routes={"default": route, tier: route},
        tasks={task: tier},
    )


class FakeRegistry:
    """只实现 gateway 用到的四个成员。

    不继承 `Registry` 的原因：构造真 Registry 不会建 adapter（懒加载），
    但 `get()` 会去读 `api_key`，测试就得准备四个假密钥——替身更干净。
    """

    def __init__(self, catalog: Catalog, clients: dict[str, LLMClient]) -> None:
        self._catalog = catalog
        self._clients = dict(clients)

    @property
    def created(self) -> tuple[str, ...]:
        return tuple(self._clients)

    def get(self, alias: str) -> LLMClient:
        return self._clients[alias]

    def get_model(self, alias: str) -> ModelConfig:
        return self._catalog.get_model(alias)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()


def build_test_gateway(
    catalog: Catalog,
    clients: dict[str, LLMClient],
    *,
    settings: Settings | None = None,
    sleep: Any = None,
) -> LLMGateway:
    return LLMGateway(
        registry=FakeRegistry(catalog, clients),  # type: ignore[arg-type]
        router=StaticRouter(list(clients)),
        catalog=catalog,
        settings=settings or make_settings(),
        sleep=sleep or RecordingSleep(),
    )


def user_call(
    text: str = "帮我查一下 MCP 协议的最新进展",
    *,
    tools: list[ToolSpec] | None = None,
    model: str | None = None,
) -> ChatCall:
    return ChatCall(
        messages=[Message(role=MessageRole.USER, content=text)],
        tools=tools,
        model=model,
    )


# --------------------------------------------------------------------------- #
# 可观测的替身
# --------------------------------------------------------------------------- #


class RecordingSleep:
    """记录退避时长而不是真睡。

    没有它，重试测试会真的睡 0.5 + 1.0 秒；三个用例就让整个套件慢 10 秒，
    而"测试变慢"的下一步一定是"把重试逻辑用 mock 掉"，重试就再也测不到了。
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --------------------------------------------------------------------------- #
# HTTP 替身
# --------------------------------------------------------------------------- #


def json_response(payload: dict[str, Any], *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code, json=payload, headers={"content-type": "application/json"}
    )


def sse_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=text.encode("utf-8"),
        headers={"content-type": "text/event-stream"},
    )


class SequenceTransport:
    """按调用序回放；用尽后重复最后一个（模拟"一直失败"）。

    **每一项都必须是工厂**（`lambda: json_response(...)`）：httpx 的 Response 只能被
    消费一次，复用同一个对象在第二次请求时会报 "response already read"。
    """

    def __init__(
        self, responses: Sequence[Callable[[], httpx.Response]]
    ) -> None:
        if not responses:
            raise ValueError("至少给一个响应工厂")

        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)

        return self._responses[index]()

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    @property
    def count(self) -> int:
        return len(self.requests)


class FailingByteStream(httpx.AsyncByteStream):
    """先吐若干字节再断连，用来构造"已经输出内容后上游挂掉"。"""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = list(chunks)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

        raise httpx.ReadError("connection dropped mid-stream")


# --------------------------------------------------------------------------- #
# 可脚本化的 adapter
# --------------------------------------------------------------------------- #


class StubClient(LLMClient):
    """按脚本产出响应或抛错的假 adapter。

    脚本元素可以是异常实例——这就是"adapter 内部失败"的建模方式。
    用它可以在不碰 HTTP 的前提下，精确控制失败发生在第几个事件之前。
    """

    def __init__(
        self,
        name: str = "stub",
        *,
        model: str = "stub-model",
        chat_results: Sequence[LLMResponse | Exception] | None = None,
        stream_scripts: Sequence[Sequence[StreamEvent | Exception]] | None = None,
        capabilities: frozenset[Capability] | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._chat_results = list(chat_results or [])
        self._stream_scripts = list(stream_scripts or [])
        self._capabilities = capabilities or frozenset(
            {Capability.STREAM, Capability.TOOLS}
        )
        self.chat_calls = 0
        self.stream_calls = 0
        self.closed = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: object,
    ) -> LLMResponse:
        self.chat_calls += 1

        if not self._chat_results:
            raise AssertionError(f"{self._name}: 没有更多脚本化的非流式响应")

        item = self._chat_results.pop(0)

        if isinstance(item, Exception):
            raise item

        return item

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: object,
    ) -> AsyncIterator[StreamEvent]:
        self.stream_calls += 1

        if not self._stream_scripts:
            raise AssertionError(f"{self._name}: 没有更多脚本化的流式响应")

        for item in self._stream_scripts.pop(0):
            if isinstance(item, Exception):
                raise item

            yield item

    async def aclose(self) -> None:
        self.closed += 1


# --------------------------------------------------------------------------- #
# 事件与响应构造
# --------------------------------------------------------------------------- #


def text_response(
    text: str = "ok",
    *,
    provider: str = "stub",
    input_tokens: int = 10,
    output_tokens: int = 5,
    estimated: bool = False,
) -> LLMResponse:
    return LLMResponse(
        id="resp-test",
        model="stub-model",
        provider=provider,
        message=Message(role=MessageRole.ASSISTANT, content=text),
        finish_reason=FinishReason.STOP,
        raw_finish_reason="stop",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated=estimated,
        ),
    )


def text_delta(text: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=text)


def message_end(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    estimated: bool = False,
) -> StreamEvent:
    return StreamEvent(
        type=StreamEventType.MESSAGE_END,
        finish_reason=FinishReason.STOP,
        raw_finish_reason="stop",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated=estimated,
        ),
    )

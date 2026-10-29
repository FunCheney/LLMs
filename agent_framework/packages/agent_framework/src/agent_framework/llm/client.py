"""客户端层：重试 / 计量 / 日志 / fallback 的唯一归属（ADR-003 / ADR-008）。

**ADR-008：这里用编排方法，不用装饰器链。** 原因不是风格：
流式重试的正确性依赖一个运行时状态 `committed`（是否已把事件吐给用户），
装饰器把 `stream()` 当黑盒拿不到它；而在 async generator 边界上，
`yield` 之后抛出的异常也无法被外层装饰器捕获重放（§3.7 明令禁止）。

分清两个对象：

- `LLMClient`（base.py）= **adapter**，只做 wire ↔ internal 翻译；
- `LLMGateway`（本文件）= 编排者，持有路由、重试、计量、日志、fallback。

把这两者混成一个类，是这类项目最常见的返工来源。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from ..config import Catalog, ConfigError, Settings
from ..observability import (
    RequestLog,
    compute_cost,
    current_request_id,
    emit,
    emit_candidate_failure,
)
from .base import Capability, LLMClient
from .errors import (
    BadRequestError,
    CapabilityError,
    LLMError,
    UpstreamServerError,
    UpstreamTimeoutError,
    error_code_for,
)
from .registry import Registry
from .router import Router
from .types import (
    FinishReason,
    LLMResponse,
    Message,
    StreamEvent,
    StreamEventType,
    ToolChoice,
    ToolSpec,
    Usage,
)

_LOG = logging.getLogger("agent_framework.llm")

# 这些事件一旦发出，就**不能再重试**：用户已经看见内容了（§3.7 红线）。
_COMMITTING_EVENTS = frozenset(
    {
        StreamEventType.TEXT_DELTA,
        StreamEventType.REASONING_DELTA,
        StreamEventType.TOOL_CALL,
    }
)

# 首 token 计时只认这两个：reasoning_delta 不算，否则思考型模型的 p95 会虚高到 5s+。
_FIRST_TOKEN_EVENTS = frozenset(
    {StreamEventType.TEXT_DELTA, StreamEventType.TOOL_CALL}
)


@dataclass(slots=True)
class ChatCall:
    """一次调用的意图。业务层只构造这个，不接触任何厂商字段。"""

    messages: list[Message]
    task: str = "chat"
    tools: list[ToolSpec] | None = None
    tool_choice: ToolChoice | None = None

    # 覆盖路由，仅调试用（评测集钉模型时也会用到）。
    model: str | None = None

    opts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatOutcome:
    """响应 + 路由/重试元信息。

    为什么不把这些字段塞进 `LLMResponse`：那是契约层（§3.4），
    `retry_count` / `fallback_from` 是客户端层的运营信息，混进去会让
    契约层为客户端实现背锅。
    """

    response: LLMResponse
    alias: str
    retry_count: int = 0
    fallback_from: str | None = None


class LLMGateway:
    """把 adapter 组织成一条可运营的链路。"""

    def __init__(
        self,
        *,
        registry: Registry,
        router: Router,
        catalog: Catalog,
        settings: Settings,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._registry = registry
        self._router = router
        self._catalog = catalog
        self._settings = settings
        self._sleep = sleep
        self._clock = clock

    # -- 预检 ------------------------------------------------------------ #

    def precheck(self, call: ChatCall, *, stream: bool = False) -> str:
        """路由 + 取 adapter + 能力校验，返回选中的 alias。

        **必须在构造 `StreamingResponse` 之前调用。** 因为 StreamingResponse 是
        先发 header 再迭代 body，一旦进了生成器，任何错误都只能变成
        200 + `event: error`——调用方会以为请求成功了。
        """

        aliases = self._plan(call)
        alias = aliases[0]
        self._ensure_capabilities(alias, call, stream=stream)

        return alias

    # -- 非流式 ---------------------------------------------------------- #

    async def chat(self, call: ChatCall) -> ChatOutcome:
        aliases = self._plan(call)
        tier = self._tier(call)
        fallback_from: str | None = None
        last_error: LLMError | None = None
        started_all = self._clock()

        for alias in aliases:
            adapter = self._ensure_capabilities(alias, call, stream=False)
            model = self._registry.get_model(alias)
            started = self._clock()

            for attempt in range(1, self._settings.max_attempts + 1):
                try:
                    response = await adapter.chat(
                        call.messages,
                        call.tools,
                        tool_choice=call.tool_choice,
                        **call.opts,
                    )
                except (LLMError, httpx.HTTPError) as exc:
                    error = self._as_llm_error(exc, adapter)

                    if not self._should_retry(error, attempt):
                        last_error = error
                        emit_candidate_failure(
                            alias, attempt, error_code_for(error), str(error)
                        )
                        break

                    _LOG.info(
                        "retry alias=%s attempt=%d code=%s",
                        alias,
                        attempt,
                        error_code_for(error),
                    )
                    await self._sleep(self._delay(attempt - 1, error.retry_after))
                    continue

                latency_ms = (self._clock() - started) * 1000
                response.provider = response.provider or adapter.name
                response.latency_ms = latency_ms
                cost = compute_cost(model.pricing, response.usage)

                emit(
                    RequestLog(
                        request_id=current_request_id(),
                        task=call.task,
                        tier=tier,
                        provider=response.provider,
                        model=alias,
                        fallback_from=fallback_from,
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                        cost=cost,
                        currency=model.pricing.currency,
                        latency_ms=round(latency_ms, 3),
                        first_token_ms=None,
                        retry_count=attempt - 1,
                        finish_reason=str(response.finish_reason),
                        error_code=None,
                        stream=False,
                        usage_estimated=response.usage.estimated,
                    )
                )

                return ChatOutcome(
                    response=response,
                    alias=alias,
                    retry_count=attempt - 1,
                    fallback_from=fallback_from,
                )

            # 这个候选彻底失败 → 进入下一个（全链只绕一圈，绝不回头）
            fallback_from = alias

        error = last_error or BadRequestError(
            f"task '{call.task}' 没有可用候选",
            provider="gateway",
        )

        emit(
            RequestLog(
                request_id=current_request_id(),
                task=call.task,
                tier=tier,
                provider=error.provider,
                model=fallback_from or "",
                fallback_from=fallback_from,
                latency_ms=round((self._clock() - started_all) * 1000, 3),
                finish_reason="error",
                error_code=error_code_for(error),
                stream=False,
            )
        )

        raise error

    # -- 流式 ------------------------------------------------------------ #

    async def stream(self, call: ChatCall) -> AsyncIterator[StreamEvent]:
        aliases = self._plan(call)
        tier = self._tier(call)
        fallback_from: str | None = None
        last_error: LLMError | None = None
        started_all = self._clock()

        for alias in aliases:
            adapter = self._ensure_capabilities(alias, call, stream=True)
            model = self._registry.get_model(alias)
            for attempt in range(1, self._settings.max_attempts + 1):
                started = self._clock()
                agen = cast(
                    AsyncGenerator[StreamEvent, None],
                    adapter.stream(
                        call.messages,
                        call.tools,
                        tool_choice=call.tool_choice,
                        **call.opts,
                    ),
                )

                committed = False
                first_token_ms: float | None = None
                usage: Usage | None = None
                finish_reason: FinishReason = FinishReason.UNKNOWN
                error: LLMError | None = None

                try:
                    try:
                        event = await self._first_event(agen, adapter, alias)
                    except LLMError as exc:
                        # 首个事件之前失败 = 用户什么都还没看到 → 允许重试
                        error = exc
                    else:
                        while True:
                            committed = committed or event.type in _COMMITTING_EVENTS

                            if first_token_ms is None and event.type in _FIRST_TOKEN_EVENTS:
                                first_token_ms = (self._clock() - started) * 1000

                            if event.usage is not None:
                                usage = event.usage

                            if event.finish_reason is not None:
                                finish_reason = event.finish_reason

                            yield event

                            try:
                                event = await agen.__anext__()
                            except StopAsyncIteration:
                                break
                            except (LLMError, httpx.HTTPError) as exc:
                                error = self._as_llm_error(exc, adapter)
                                break
                finally:
                    # 放弃半开的流必须显式关闭：不关会把连接留在池里，
                    # 功能测试一天都发现不了，W2 压测立刻 `Too many open files`。
                    await agen.aclose()

                if error is None:
                    latency_ms = (self._clock() - started) * 1000
                    final_usage = usage or Usage(estimated=True)
                    cost = compute_cost(model.pricing, final_usage)

                    emit(
                        RequestLog(
                            request_id=current_request_id(),
                            task=call.task,
                            tier=tier,
                            provider=adapter.name,
                            model=alias,
                            fallback_from=fallback_from,
                            prompt_tokens=final_usage.input_tokens,
                            completion_tokens=final_usage.output_tokens,
                            cost=cost,
                            currency=model.pricing.currency,
                            latency_ms=round(latency_ms, 3),
                            first_token_ms=(
                                round(first_token_ms, 3)
                                if first_token_ms is not None
                                else None
                            ),
                            retry_count=attempt - 1,
                            finish_reason=str(finish_reason),
                            error_code=None,
                            stream=True,
                            usage_estimated=final_usage.estimated,
                        )
                    )

                    return

                if committed:
                    # 已经吐过内容：只能以 error 帧终止，绝不重放（§3.7 红线）。
                    code = error_code_for(error)
                    _LOG.warning(
                        "stream aborted after commit alias=%s code=%s", alias, code
                    )
                    emit(
                        RequestLog(
                            request_id=current_request_id(),
                            task=call.task,
                            tier=tier,
                            provider=adapter.name,
                            model=alias,
                            fallback_from=fallback_from,
                            latency_ms=round((self._clock() - started) * 1000, 3),
                            first_token_ms=(
                                round(first_token_ms, 3)
                                if first_token_ms is not None
                                else None
                            ),
                            retry_count=attempt - 1,
                            finish_reason="error",
                            error_code=code,
                            stream=True,
                        )
                    )

                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        data={
                            "code": code,
                            "message": str(error),
                            "request_id": current_request_id() or "-",
                        },
                    )

                    return

                last_error = error

                if not self._should_retry(error, attempt):
                    emit_candidate_failure(
                        alias, attempt, error_code_for(error), str(error)
                    )
                    break

                _LOG.info(
                    "retry stream alias=%s attempt=%d code=%s",
                    alias,
                    attempt,
                    error_code_for(error),
                )
                await self._sleep(self._delay(attempt - 1, error.retry_after))

            fallback_from = alias

        error = last_error or BadRequestError(
            f"task '{call.task}' 没有可用候选",
            provider="gateway",
        )

        emit(
            RequestLog(
                request_id=current_request_id(),
                task=call.task,
                tier=tier,
                provider=error.provider,
                model=fallback_from or "",
                fallback_from=fallback_from,
                latency_ms=round((self._clock() - started_all) * 1000, 3),
                finish_reason="error",
                error_code=error_code_for(error),
                stream=True,
            )
        )

        raise error

    # -- 生命周期 -------------------------------------------------------- #

    async def aclose(self) -> None:
        await self._registry.aclose()

    # -- 内部：路由与能力 ------------------------------------------------ #

    def _plan(self, call: ChatCall) -> list[str]:
        aliases = [call.model] if call.model else self._router.route(call.task)

        if not aliases:
            raise BadRequestError(
                f"task '{call.task}' 没有任何候选模型", provider="gateway"
            )

        # 结构错误在这里就暴露，别等到请求发出后。
        #
        # 但两类错误的**归属不同**：`call.model` 来自调用方（调试/评测钉模型），
        # alias 打错是调用方的输入错误 → 400；路由里的 alias 来自 models.yaml，
        # 那是我们自己的配置坏了 → ConfigError → 500。混成一个会让值班同学
        # 为"调用方拼错模型名"收到 500 告警。
        for alias in aliases:
            if call.model is not None:
                try:
                    self._registry.get_model(alias)
                except ConfigError as exc:
                    raise BadRequestError(
                        f"未知的模型 alias '{alias}'", provider="gateway"
                    ) from exc
            else:
                self._registry.get_model(alias)

        return aliases

    def _tier(self, call: ChatCall) -> str:
        return self._catalog.tasks.get(call.task, "default")

    def _ensure_capabilities(
        self, alias: str, call: ChatCall, *, stream: bool
    ) -> LLMClient:
        """硬需求不满足就抛；软策略（tool_choice）由 adapter 降级、不抛。

        两者不要混：`tools` 是调用方的硬需求（拿不到工具调用，整条链路就没意义），
        `tool_choice=required` 只是策略意图（模型仍可能返回纯文本）。
        """

        adapter = self._registry.get(alias)
        caps = adapter.capabilities

        if call.tools and Capability.TOOLS not in caps:
            raise CapabilityError(
                f"模型 '{alias}' 不支持工具调用",
                provider=adapter.name,
                model=alias,
            )

        if stream and Capability.STREAM not in caps:
            raise CapabilityError(
                f"模型 '{alias}' 不支持流式输出",
                provider=adapter.name,
                model=alias,
            )

        return adapter

    # -- 内部：重试 ------------------------------------------------------ #

    def _should_retry(self, error: LLMError, attempt: int) -> bool:
        """只依赖 `error.retryable` 一个字段。

        retryable 的判定（429 / 402 / 422 / code 白名单）已经在 errors.py 里做完，
        这里再写一遍 `if status == 429` 就是把同一份逻辑放两个地方，
        W11 一定改一处漏一处。
        """

        return error.retryable and attempt < self._settings.max_attempts

    def _delay(self, retry_index: int, retry_after: float | None) -> float:
        """限流听上游的，其余指数退避 + 抖动。

        抖动乘 `0.5 + rand` 而不是 `1 + rand` 是刻意的：多个实例同时被限流时，
        只有**向下**抖动才能把重试打散；向上抖动会让尖峰更高。
        """

        if retry_after is not None:
            return min(retry_after, self._settings.backoff_cap_s) * (
                1 + self._jitter() * 0.1
            )

        base = self._settings.backoff_base_s * (2**retry_index)
        capped = min(base, self._settings.backoff_cap_s)

        return capped * (0.5 + self._jitter())

    @staticmethod
    def _jitter() -> float:
        return random.random()

    # -- 内部：错误与首事件 ---------------------------------------------- #

    def _as_llm_error(self, exc: Exception, adapter: LLMClient) -> LLMError:
        """把 httpx 的传输层异常翻译成内部异常。

        adapter 不负责这件事（它只管 HTTP 语义），所以必须在 client 层做，
        否则 ConnectError 会以原始形态冒到 HTTP 层变成 500。
        """

        if isinstance(exc, LLMError):
            return exc

        provider = adapter.name
        model = adapter.model or None

        if isinstance(exc, httpx.TimeoutException):
            return UpstreamTimeoutError(
                f"上游超时：{exc}", provider=provider, model=model, raw_error=exc
            )

        return UpstreamServerError(
            f"传输层错误：{exc}", provider=provider, model=model, raw_error=exc
        )

    async def _first_event(
        self,
        agen: AsyncGenerator[StreamEvent, None],
        adapter: LLMClient,
        alias: str,
    ) -> StreamEvent:
        """取首个事件，并用 `first_token_timeout` 作为边界（§3.7）。

        空流（立刻 StopAsyncIteration）必须当错误：当成正常结束会返回
        空字符串 + finish_reason=stop，这是最难排查的一类事故
        （表现为"模型偶尔不回答"）。
        """

        timeout = self._settings.first_token_timeout_s

        try:
            return await asyncio.wait_for(agen.__anext__(), timeout=timeout)
        except StopAsyncIteration as exc:
            raise UpstreamServerError(
                "上游返回了空流", provider=adapter.name, model=alias
            ) from exc
        except TimeoutError as exc:
            raise UpstreamTimeoutError(
                f"首 token 超时（{timeout}s）",
                provider=adapter.name,
                model=alias,
            ) from exc
        except httpx.HTTPError as exc:
            raise self._as_llm_error(exc, adapter) from exc

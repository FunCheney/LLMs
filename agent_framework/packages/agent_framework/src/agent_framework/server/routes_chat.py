"""POST /v1/chat。

这一层只做四件事：校验 body、投影成内部类型、调 gateway、把事件编码成响应。

**禁止**：拼 prompt、做重试、判断厂商、在流式分支里写业务逻辑。

W2 D2 加了一道并发闸门（`ConcurrencyLimiter`），位置很讲究：
- **在 gateway 之前**：gateway 的重试逻辑（ADR-003）只对上游错误生效，
  入口过载绝不能变成"再打一遍"；
- **在 `StreamingResponse` 之前**：进了生成器就只能 200 + error 帧，
  过载会伪装成"请求成功但没内容"。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import ConfigError
from ..llm.client import LLMGateway
from ..observability import current_request_id
from ..resilience.concurrency import ConcurrencyLimiter
from .schemas import ChatRequest, ChatResponse
from .sse import SSE_HEADERS, encode_event

router = APIRouter()


def _gateway(request: Request) -> LLMGateway:
    gateway = getattr(request.app.state, "gateway", None)

    if not isinstance(gateway, LLMGateway):
        raise ConfigError("gateway 未初始化：create_app() 必须注入 gateway")

    return gateway


def _limiter(request: Request) -> ConcurrencyLimiter | None:
    limiter = getattr(request.app.state, "limiter", None)

    return limiter if isinstance(limiter, ConcurrencyLimiter) else None


@router.post("/v1/chat")
async def chat(body: ChatRequest, request: Request) -> Response:
    gateway = _gateway(request)
    call = body.to_call()
    request_id = current_request_id()

    # 必须在构造 StreamingResponse **之前**做：StreamingResponse 先发 header
    # 再迭代 body，一旦进了生成器，能力错误就只能是 200 + event: error。
    alias = gateway.precheck(call, stream=body.stream)
    limiter = _limiter(request)

    if not body.stream:
        if limiter is None:
            outcome = await gateway.chat(call)
        else:
            # 拿不到槽位 → ServerBusyError → 503 + Retry-After（不会被 retry 放大）。
            async with limiter.slot(alias):
                outcome = await gateway.chat(call)

        payload = ChatResponse.from_outcome(outcome, request_id=request_id)

        return JSONResponse(
            payload.model_dump(mode="json"),
            headers={"X-Model-Alias": outcome.alias},
        )

    async def frames() -> AsyncIterator[str]:
        # request_id 在进入生成器前就固定下来：contextvar 在流式迭代期间
        # 虽然可见，但显式捕获能避免以后换成后台任务时静默丢号。
        try:
            async for event in gateway.stream(call):
                yield encode_event(event, request_id=request_id)
        finally:
            # 槽位必须在流结束时还回去：无论是正常结束、上游报错，
            # 还是客户端断开导致生成器被关闭——漏还一次就永久少一个并发额度。
            if limiter is not None:
                await limiter.release(alias)

    if limiter is not None:
        # 先占槽位再返回响应：过载要表现为 503，而不是"200 但流是空的"。
        await limiter.acquire(alias)

        try:
            return StreamingResponse(
                frames(),
                media_type="text/event-stream",
                headers={**SSE_HEADERS, "X-Model-Alias": alias},
            )
        except BaseException:
            await limiter.release(alias)
            raise

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "X-Model-Alias": alias},
    )

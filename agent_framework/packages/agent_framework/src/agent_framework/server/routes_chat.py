"""POST /v1/chat。

这一层只做四件事：校验 body、投影成内部类型、调 gateway、把事件编码成响应。

**禁止**：拼 prompt、做重试、判断厂商、在流式分支里写业务逻辑。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import ConfigError
from ..llm.client import LLMGateway
from ..observability import current_request_id
from .schemas import ChatRequest, ChatResponse
from .sse import SSE_HEADERS, encode_event

router = APIRouter()


def _gateway(request: Request) -> LLMGateway:
    gateway = getattr(request.app.state, "gateway", None)

    if not isinstance(gateway, LLMGateway):
        raise ConfigError("gateway 未初始化：create_app() 必须注入 gateway")

    return gateway


@router.post("/v1/chat")
async def chat(body: ChatRequest, request: Request) -> Response:
    gateway = _gateway(request)
    call = body.to_call()
    request_id = current_request_id()

    # 必须在构造 StreamingResponse **之前**做：StreamingResponse 先发 header
    # 再迭代 body，一旦进了生成器，能力错误就只能是 200 + event: error。
    alias = gateway.precheck(call, stream=body.stream)

    if not body.stream:
        outcome = await gateway.chat(call)
        payload = ChatResponse.from_outcome(outcome, request_id=request_id)

        return JSONResponse(
            payload.model_dump(mode="json"),
            headers={"X-Model-Alias": outcome.alias},
        )

    async def frames() -> AsyncIterator[str]:
        # request_id 在进入生成器前就固定下来：contextvar 在流式迭代期间
        # 虽然可见，但显式捕获能避免以后换成后台任务时静默丢号。
        async for event in gateway.stream(call):
            yield encode_event(event, request_id=request_id)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "X-Model-Alias": alias},
    )

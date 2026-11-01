"""`POST /v1/tasks` 与 `GET /v1/tasks/{task_id}`（W2 D2）。

入口只做三件事：**校验 → 入队 → 立即返回**。长任务绝不能在 HTTP 请求里跑完：
客户端一断开、网关一超时，协程被取消，几十秒的工作全白做，而调用方还看不到任何结果。

为什么入队前还要 `precheck()`
---------------------------
"模型 alias 打错""模型不支持 tools"这类错误**注定失败**，让它们占一个队列槽位、
排 30 秒、再以 failed 结束，是把调用方的输入错误变成我们的容量问题。
所以先用 gateway 的预检挡掉（400），通过的才入队。
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config import ConfigError
from ..llm.client import ChatCall, LLMGateway
from ..llm.errors import BadRequestError
from ..observability import current_request_id, current_trace_id
from ..queue.base import TaskHandler, TaskQueue
from .schemas import ChatRequest, ChatResponse

router = APIRouter()


def make_task_handler(gateway: LLMGateway) -> TaskHandler:
    """把"任务"还原成一次 `ChatCall`，再走 gateway。

    投影（`ChatResponse`）放在这里而不是队列里：队列不该知道对外 DTO 长什么样，
    否则 W9 换成 workflow 时要把序列化再抄一遍。
    """

    async def handler(call: ChatCall) -> dict[str, object]:
        outcome = await gateway.chat(call)
        payload = ChatResponse.from_outcome(
            outcome, request_id=current_request_id()
        ).model_dump(mode="json")

        return dict(payload)

    return handler


def _queue(request: Request) -> TaskQueue:
    queue = getattr(request.app.state, "task_queue", None)

    if queue is None:
        raise ConfigError("task_queue 未初始化：create_app() 必须注入 task_queue")

    return queue


def _gateway(request: Request) -> LLMGateway:
    gateway = getattr(request.app.state, "gateway", None)

    if not isinstance(gateway, LLMGateway):
        raise ConfigError("gateway 未初始化：create_app() 必须注入 gateway")

    return gateway


@router.post("/v1/tasks", status_code=202)
async def submit_task(body: ChatRequest, request: Request) -> JSONResponse:
    if body.stream:
        # 任务队列的语义是"稍后取结果"，流式没有承载它的地方（SSE 连接不可能挂几分钟）。
        raise BadRequestError(
            "任务队列不支持 stream=true；实时输出请直接用 /v1/chat",
            provider="gateway",
        )

    call = body.to_call()
    _gateway(request).precheck(call, stream=False)

    record = await _queue(request).enqueue(
        call,
        request_id=current_request_id(),
        trace_id=current_trace_id(),
    )

    return JSONResponse(
        {
            "task_id": record.task_id,
            "status": record.status.value,
            "request_id": record.request_id,
            "trace_id": record.trace_id,
        },
        status_code=202,
        headers={"X-Task-ID": record.task_id},
    )


@router.get("/v1/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> JSONResponse:
    record = await _queue(request).get(task_id)

    return JSONResponse(record.to_dict())

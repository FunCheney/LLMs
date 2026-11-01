"""request_id / trace_id 注入中间件（纯 ASGI）。

为什么单独一个文件：产品应用（`apps/growth_agent`）也要用同一套贯穿规则。
如果它只能 `from agent_framework.server.app import RequestIDMiddleware`，
就会顺带执行 app.py 顶层的 `app = create_app()`——为了拿一个中间件而启动一个框架应用，
是典型的边界画错（Day5 正是要在这个阶段暴露这类问题）。

request_id vs trace_id（W2 D3 起）
----------------------------------
- `request_id`：**一次 HTTP 请求**。响应头回给调用方，用于对账"我看到的那次"。
- `trace_id`：**一次用户任务**，可能跨多次 HTTP / 一次队列消费。
  调用方可以透传（跨服务串联），不传就新建。

两者都校验字符集：不校验等于给日志注入与 key 注入开了口子。
"""

from __future__ import annotations

import re
from collections.abc import Callable

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..observability import new_request_id, new_trace_id, request_id_var, trace_id_var

# 调用方可以传 x-request-id / x-trace-id 便于跨服务串联，但必须校验字符集。
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 兼容旧名（W1 的测试与文档里用的是这个）。
REQUEST_ID_PATTERN = ID_PATTERN


class RequestIDMiddleware:
    """纯 ASGI 中间件，而不是 `@app.middleware("http")`。

    两个原因：

    1. `BaseHTTPMiddleware` 会把下游放进独立 task，`StreamingResponse` 的 body
       迭代发生在 `call_next` 返回**之后**，contextvar 是否可见要依赖实现细节；
       纯 ASGI 中间件与业务在同一个 task 里，行为确定。
    2. 只有在 `http.response.start` 时才能 append header —— 用 `send_wrapper`
       正好在这个时机插 `X-Request-ID` / `X-Trace-ID`。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = self._resolve(headers.get("x-request-id"), new_request_id)
        trace_id = self._resolve(headers.get("x-trace-id"), new_trace_id)

        request_token = request_id_var.set(request_id)
        trace_token = trace_id_var.set(trace_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                outgoing = MutableHeaders(scope=message)
                outgoing.append("X-Request-ID", request_id)
                outgoing.append("X-Trace-ID", trace_id)

            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(request_token)
            trace_id_var.reset(trace_token)

    @staticmethod
    def _resolve(incoming: str | None, factory: Callable[[], str]) -> str:
        if incoming and ID_PATTERN.fullmatch(incoming):
            return incoming

        return factory()

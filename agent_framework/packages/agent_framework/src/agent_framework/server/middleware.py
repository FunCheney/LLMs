"""request_id 注入中间件（纯 ASGI）。

为什么单独一个文件：产品应用（`apps/growth_agent`）也要用同一套 request_id
贯穿规则。如果它只能 `from agent_framework.server.app import RequestIDMiddleware`，
就会顺带执行 app.py 顶层的 `app = create_app()`——为了拿一个中间件而启动一个框架应用，
是典型的边界画错（Day5 正是要在这个阶段暴露这类问题）。
"""

from __future__ import annotations

import re

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..observability import new_request_id, request_id_var

# 调用方可以传 x-request-id 便于跨服务串联，但必须校验字符集：
# 不校验等于给日志注入开了口子。
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RequestIDMiddleware:
    """纯 ASGI 中间件，而不是 `@app.middleware("http")`。

    两个原因：

    1. `BaseHTTPMiddleware` 会把下游放进独立 task，`StreamingResponse` 的 body
       迭代发生在 `call_next` 返回**之后**，contextvar 是否可见要依赖实现细节；
       纯 ASGI 中间件与业务在同一个 task 里，行为确定。
    2. 只有在 `http.response.start` 时才能 append header —— 用 `send_wrapper`
       正好在这个时机插 `X-Request-ID`。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id")
        request_id = (
            incoming
            if incoming and REQUEST_ID_PATTERN.fullmatch(incoming)
            else new_request_id()
        )

        token = request_id_var.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("X-Request-ID", request_id)

            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)

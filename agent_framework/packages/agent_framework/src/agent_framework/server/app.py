"""框架的 HTTP 入口。

职责只有四件：组装 gateway / 会话层 / 并发闸门 / 任务队列，装 request_id 与 trace_id
中间件和错误契约，暴露 `/v1/chat` 与 `/v1/tasks`。

**禁止**：拼 prompt、直连厂商 HTTP、在路由层做重试、把 `/v1/chat` 塞进任务队列。

Day5 的两处抽离（都是"产品层要复用"逼出来的）：
`RequestIDMiddleware` → `server/middleware.py`、错误契约 → `server/errors.py`。
不抽的话，`apps/growth_agent` 要么复制一份错误码、要么 import 本模块时
连带执行顶层 `app = create_app()`（为了拿一个中间件而启动一个应用）。

W2 的两处：`session_store` 与 `limiter` 都做成**可注入**参数。
测试用内存实现、生产用 Redis + yaml 限额，走同一条装配代码——
"测试走一条路、生产走另一条"是这类项目最常见的假绿来源。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from agent_framework import __version__
from agent_framework.bootstrap import build_gateway, build_limiter
from agent_framework.llm.client import LLMGateway
from agent_framework.observability import configure_logging
from agent_framework.queue.base import TaskQueue
from agent_framework.queue.inmemory import InMemoryTaskQueue
from agent_framework.resilience.concurrency import ConcurrencyLimiter
from agent_framework.server.errors import install_error_handlers
from agent_framework.server.middleware import RequestIDMiddleware
from agent_framework.server.routes_chat import router as chat_router
from agent_framework.server.routes_task import make_task_handler
from agent_framework.server.routes_task import router as task_router
from agent_framework.session.base import SessionStore
from agent_framework.session.factory import build_session_store


def create_app(
    gateway: LLMGateway | None = None,
    *,
    session_store: SessionStore | None = None,
    limiter: ConcurrencyLimiter | None = None,
    task_queue: TaskQueue | None = None,
) -> FastAPI:
    # 应用边界才配日志：库本身不擅自改宿主的 logging 配置。
    configure_logging()

    resolved_gateway = gateway if gateway is not None else build_gateway()
    settings = resolved_gateway.settings

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # 谁建谁关：队列消费者、Redis 连接池、httpx 连接池。
            # 不关的后果在功能测试里永远看不到，压测时会以 `Too many open files`
            # 或"任务卡在 running"的形式出现。
            await app.state.task_queue.aclose()
            await app.state.session_store.aclose()
            await app.state.gateway.aclose()

    app = FastAPI(title="agent-framework", version=__version__, lifespan=lifespan)

    # 启动时只做结构校验，不建 adapter（懒加载）；测试可直接注入替身。
    app.state.gateway = resolved_gateway
    app.state.session_store = session_store or build_session_store(settings)
    app.state.limiter = limiter or build_limiter(resolved_gateway.catalog, settings)
    app.state.task_queue = task_queue or InMemoryTaskQueue(
        make_task_handler(resolved_gateway),
        max_size=settings.task_queue_size,
        workers=settings.task_workers,
    )

    app.add_middleware(RequestIDMiddleware)
    app.include_router(chat_router)
    app.include_router(task_router)
    install_error_handlers(app)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        limiter_state = app.state.limiter
        queue_state = app.state.task_queue

        return {
            "status": "ok",
            "service": "agent-framework",
            "version": __version__,
            "session_backend": settings.session_backend,
            "in_flight": (
                limiter_state.in_flight()
                if isinstance(limiter_state, ConcurrencyLimiter)
                else {}
            ),
            "queue": (
                queue_state.snapshot()
                if isinstance(queue_state, InMemoryTaskQueue)
                else {"backend": type(queue_state).__name__}
            ),
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("agent_framework.server.app:app", host="0.0.0.0", port=8000)

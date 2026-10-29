"""产品 HTTP 入口。

Day5 之后它验证四件事：
1. 产品应用能独立启动；
2. 能 import 并消费 `agent_framework`（依赖方向单向，无循环）；
3. `/health` 报出框架版本，便于确认部署时两边版本一致；
4. `/v1/draft` 真的走一遍框架（路由 → adapter → 计量 → 日志），
   证明"框架可被消费"，而不是只能自测。

两处细节都是 Day5 才补的，且都不是"写法偏好"：

- **lifespan 里 `gateway.aclose()`**：不关 httpx client，长跑进程会
  `Too many open files`——功能测试一天都发现不了，W2 压测立刻炸。
- **错误契约复用 `install_error_handlers`**：本文件不写任何 `try/except`，
  否则同一个上游故障在框架与产品两侧会返回不同 code。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from agent_framework import __version__ as framework_version
from agent_framework.bootstrap import build_gateway
from agent_framework.llm.client import LLMGateway
from agent_framework.observability import configure_logging
from agent_framework.server.errors import install_error_handlers
from agent_framework.server.middleware import RequestIDMiddleware
from fastapi import FastAPI

import growth_agent
from growth_agent.routes_draft import router as draft_router


def create_app(gateway: LLMGateway | None = None) -> FastAPI:
    # 产品应用同样要配：否则本进程里的 RequestLog 会被静默丢弃，
    # 表现为"框架说它记了日志，产品侧一条都查不到"。
    configure_logging()

    resolved_gateway = gateway if gateway is not None else build_gateway()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await app.state.gateway.aclose()

    app = FastAPI(
        title="growth-agent", version=growth_agent.__version__, lifespan=lifespan
    )

    app.state.gateway = resolved_gateway

    app.add_middleware(RequestIDMiddleware)
    install_error_handlers(app)
    app.include_router(draft_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "growth-agent",
            "version": growth_agent.__version__,
            "framework_version": framework_version,
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("growth_agent.app:app", host="0.0.0.0", port=8100)

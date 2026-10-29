"""框架的 HTTP 入口。

职责只有三件：组装 gateway、装 request_id 中间件与错误契约、暴露 `/v1/chat`。

**禁止**：拼 prompt、直连厂商 HTTP、在路由层做重试。

Day5 的两处抽离（都是"产品层要复用"逼出来的）：
`RequestIDMiddleware` → `server/middleware.py`、错误契约 → `server/errors.py`。
不抽的话，`apps/growth_agent` 要么复制一份错误码、要么 import 本模块时
连带执行顶层 `app = create_app()`（为了拿一个中间件而启动一个应用）。
"""

from __future__ import annotations

from fastapi import FastAPI

from agent_framework import __version__
from agent_framework.bootstrap import build_gateway
from agent_framework.llm.client import LLMGateway
from agent_framework.observability import configure_logging
from agent_framework.server.errors import install_error_handlers
from agent_framework.server.middleware import RequestIDMiddleware
from agent_framework.server.routes_chat import router as chat_router


def create_app(gateway: LLMGateway | None = None) -> FastAPI:
    # 应用边界才配日志：库本身不擅自改宿主的 logging 配置。
    configure_logging()

    app = FastAPI(title="agent-framework", version=__version__)

    # 启动时只做结构校验，不建 adapter（懒加载）；测试可直接注入替身。
    app.state.gateway = gateway if gateway is not None else build_gateway()

    app.add_middleware(RequestIDMiddleware)
    app.include_router(chat_router)
    install_error_handlers(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "agent-framework", "version": __version__}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("agent_framework.server.app:app", host="0.0.0.0", port=8000)

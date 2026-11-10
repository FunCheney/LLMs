"""产品 HTTP 入口。

验证四件事：
1. 产品应用能独立启动；
2. 能 import 并消费 `agent_framework`（依赖方向单向，无循环）；
3. `/health` 报出框架版本，便于确认部署时两边版本一致；
4. 端点真的走一遍框架（路由 → adapter → 计量 → 日志），
   证明"框架可被消费"，而不是只能自测。

两处细节都是 Day5 才补的，且都不是"写法偏好"：

- **lifespan 里 `gateway.aclose()`**：不关 httpx client，长跑进程会
  `Too many open files`——功能测试一天都发现不了，W2 压测立刻炸。
- **错误契约复用 `install_error_handlers`**：本文件不写任何 `try/except`，
  否则同一个上游故障在框架与产品两侧会返回不同 code。

W6 新增：检索链路（`/v1/answer`）**不在启动时装配**。理由与代价见 `runtime.py`：
模型加载 18–20s，放进 lifespan 会让 healthcheck 超时；所以第一个 `/v1/answer`
请求承担冷启动，上线后应显式打一次预热请求（RUNBOOK §6）。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from agent_framework import __version__ as framework_version
from agent_framework.bootstrap import build_gateway
from agent_framework.config import Settings
from agent_framework.llm.client import LLMGateway
from agent_framework.observability import configure_logging
from agent_framework.server.errors import install_error_handlers
from agent_framework.server.middleware import RequestIDMiddleware
from fastapi import FastAPI

import growth_agent
from growth_agent.routes_answer import router as answer_router
from growth_agent.routes_draft import router as draft_router
from growth_agent.routes_memory import router as memory_router
from growth_agent.runtime import close_runtime


def create_app(
    gateway: LLMGateway | None = None,
    *,
    settings: Settings | None = None,
    answerer: Any | None = None,
) -> FastAPI:
    """`answerer` 是给测试的注入口：传一个假回答器就不必加载模型。

    它必须是"显式注入"而不是"环境变量开关"——测试若靠 env 关掉检索，
    就永远测不到接线的真实行为（这正是 W5 留下的那道缺口）。
    """

    # 产品应用同样要配：否则本进程里的 RequestLog 会被静默丢弃，
    # 表现为"框架说它记了日志，产品侧一条都查不到"。
    configure_logging()

    resolved_settings = settings or Settings()
    resolved_gateway = gateway if gateway is not None else build_gateway(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # 先放检索（线程池 + 模型），再放网关（httpx 连接池）。
            # 顺序反了不会报错，但会让"退出时挂住"的排查从线程池开始——
            # 而真正的原因可能是网关连接没关。
            await close_runtime(app)
            await app.state.gateway.aclose()

    app = FastAPI(
        title="growth-agent", version=growth_agent.__version__, lifespan=lifespan
    )

    app.state.gateway = resolved_gateway
    app.state.settings = resolved_settings
    # 惰性装配用。Lock 在没有运行中的事件循环时也能构造（3.10+）。
    app.state.rag_lock = asyncio.Lock()
    app.state.rag = None
    app.state.answerer = answerer
    # W7：记忆运行时（MEMORY=governed 时由 ensure_answerer 惰性装配）。
    app.state.memory = None

    app.add_middleware(RequestIDMiddleware)
    install_error_handlers(app)
    app.include_router(draft_router)
    app.include_router(answer_router)
    app.include_router(memory_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "growth-agent",
            "version": growth_agent.__version__,
            "framework_version": framework_version,
            # 检索是惰性加载的，所以要能看出"现在到底就绪没有"：
            # 否则 p95 里那条 20s 的首请求只能靠猜是冷启动还是模型真慢。
            "rag": "ready" if getattr(app.state, "rag", None) is not None else "cold",
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("growth_agent.app:app", host="0.0.0.0", port=8100)

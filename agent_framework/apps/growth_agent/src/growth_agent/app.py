"""产品 HTTP 入口。

W1 阶段只验证三件事：
1. 产品应用能独立启动
2. 能 import 并消费 agent_framework（依赖方向正确、无循环）
3. /health 能报出框架版本，便于确认部署时两边版本一致

W1 Day5 增加透传端点；W9 起挂载真实业务 pipeline。
"""

from agent_framework import __version__ as framework_version
from fastapi import FastAPI

import growth_agent


def create_app() -> FastAPI:
    app = FastAPI(title="growth-agent", version=growth_agent.__version__)

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

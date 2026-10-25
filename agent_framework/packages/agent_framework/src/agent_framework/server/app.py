"""框架的 HTTP 入口。

当前只提供 /health，用于确认项目可运行。
W1 Day4 在此挂载 /v1/chat：

1. 新增 ``routes_chat.py`` 并在 create_app() 里 ``include_router``
2. 异常处理器按 docs/architecture.md 3.6 节把 LLMError 映射为 {code, message, request_id}
3. request_id 用 contextvar 注入，贯穿到 provider 日志

约束：本文件不拼 prompt、不做重试、不出现任何厂商字段。
"""

from fastapi import FastAPI

from agent_framework import __version__


def create_app() -> FastAPI:
    app = FastAPI(title="agent-framework", version=__version__)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "agent-framework", "version": __version__}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("agent_framework.server.app:app", host="0.0.0.0", port=8000)

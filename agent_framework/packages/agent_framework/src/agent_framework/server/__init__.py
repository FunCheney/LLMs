"""HTTP / SSE 传输层（W1 Day4）。

- app.py           FastAPI 实例、依赖注入、异常处理器（当前仅 /health）
- sse.py           StreamEvent -> SSE 帧编码
- routes_chat.py   POST /v1/chat（stream=true/false）

禁止：拼 prompt、直连厂商 HTTP、业务分支。
"""

__all__: list[str] = []

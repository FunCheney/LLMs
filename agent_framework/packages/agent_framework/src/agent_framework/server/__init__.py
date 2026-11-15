"""HTTP / SSE 传输层（W1 Day4）。

- app.py           FastAPI 实例、依赖注入、异常处理器
- sse.py           StreamEvent -> SSE 帧编码
- routes_chat.py   POST /v1/chat（stream=true/false）
- routes_task.py   长任务端点
- auth.py          API Key 鉴权（默认关闭 = 只跑本地）
- errors.py        异常 → HTTP 响应体的唯一投影处
- middleware.py    请求 ID 贯穿

禁止：拼 prompt、直连厂商 HTTP、业务分支。

## 这一层导出 `create_app`，但**不**导出 `app`

原来的写法是「只导出纯函数」：`app.py` 在模块顶层 `app = create_app()`，
所以这里一旦 `from .app import create_app`，任何人 `from agent_framework.server.auth import ...`
都会连带把 FastAPI 实例建出来——import 变成一次看不见的初始化。

W12 D2 把那个副作用改成了 PEP 562 惰性属性（`app.py` 的 `__getattr__`），
`import` 不再构造任何东西。于是 `create_app` 可以正常导出，
而 `app`（那个全局单例）仍然只在 `uvicorn ...:app` 或显式 `from ... import app`
时才被建出来——想装配自己的服务的调用方，不必先生成一份用不上的。
"""

from .app import create_app
from .auth import (
    AUTH_INVALID,
    AUTH_REQUIRED,
    APIKeyAuth,
    AuthError,
    install_api_key_auth,
    parse_api_keys,
)
from .errors import error_body, install_error_handlers
from .middleware import RequestIDMiddleware

__all__ = [
    "APIKeyAuth",
    "AUTH_INVALID",
    "AUTH_REQUIRED",
    "AuthError",
    "RequestIDMiddleware",
    "create_app",
    "error_body",
    "install_api_key_auth",
    "install_error_handlers",
    "parse_api_keys",
]

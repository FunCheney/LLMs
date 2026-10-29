"""对外错误契约（architecture §3.6）：一张映射表 + 一组 handler。

**为什么抽成公共函数**：产品应用（`apps/growth_agent`）必须复用同一套错误契约。
如果它自己再写一遍 `try/except` 把异常转成 `{code, message}`，W12 提取框架时就会发现
"错误码有两份定义"，而这类分叉在线上表现为"同一个上游故障，两个服务返回不同 code"。

纪律：这里只做**翻译**，不判断业务；`http_status_for` / `error_code_for` 的归属
仍然是 `llm/errors.py`（唯一真相源），本文件不重复定义状态码。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import ConfigError
from ..llm.errors import LLMError, error_code_for, http_status_for
from ..observability import current_request_id


def error_body(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """统一的错误体：`{code, message, request_id}` + 可选上下文。"""

    body: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": current_request_id() or "-",
    }
    body.update({key: value for key, value in extra.items() if value is not None})

    return body


def install_error_handlers(app: FastAPI) -> None:
    """把三类异常挂到 app 上。框架与产品应用都调这一个函数。"""

    @app.exception_handler(LLMError)
    async def _on_llm_error(request: Request, exc: LLMError) -> JSONResponse:
        headers: dict[str, str] = {}

        if exc.retry_after is not None:
            # 透传 Retry-After：调用方按它退避，比我们自己睡够更有效。
            headers["Retry-After"] = str(int(exc.retry_after))

        return JSONResponse(
            error_body(
                error_code_for(exc),
                str(exc),
                provider=exc.provider,
                model=exc.model,
                upstream_status=exc.status_code,
            ),
            status_code=http_status_for(exc),
            headers=headers,
        )

    @app.exception_handler(ConfigError)
    async def _on_config_error(request: Request, exc: ConfigError) -> JSONResponse:
        # 配置错误是我们自己的问题，对外统一 500：绝不能伪装成 400
        # 让调用方以为是自己参数写错了。
        return JSONResponse(error_body("CONFIG_ERROR", str(exc)), status_code=500)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "loc": ".".join(str(part) for part in item.get("loc", ())),
                "msg": str(item.get("msg", "")),
            }
            for item in exc.errors()
        ]

        return JSONResponse(
            error_body("BAD_REQUEST", "请求参数校验失败", details=details),
            status_code=400,
        )

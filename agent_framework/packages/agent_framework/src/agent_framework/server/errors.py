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
    """把四类异常挂到 app 上。框架与产品应用都调这一个函数。

    第四类是 W10 加的安全类（`BlockedInput` / `OutputRejected` / `ApprovalRequired`）。
    加在这里而不是让每个路由自己 `try/except`，理由与文件头那条一样：
    安全拦截若各自翻译，同一次拦截在不同端点会得到不同状态码，
    而值班同学是按状态码分流的。**安全类的 400/422/202 只能有一处定义。**
    """

    # 局部导入：`guardrails` 依赖 `skills`（拿 risk 级），把它提到模块顶层
    # 会让 `server.errors` 变成一条通往工具链的路径，而错误表本身该是最小依赖。
    from ..guardrails.hitl import ApprovalRequired
    from ..guardrails.input import BlockedInput
    from ..guardrails.output import OutputRejected

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

    @app.exception_handler(BlockedInput)
    async def _on_blocked_input(request: Request, exc: BlockedInput) -> JSONResponse:
        # 400 而不是 422：输入在**处理之前**就被拒了，还没到"能不能处理"。
        return JSONResponse(
            error_body(exc.code, exc.message, violations=list(exc.violations)),
            status_code=400,
        )

    @app.exception_handler(OutputRejected)
    async def _on_output_rejected(request: Request, exc: OutputRejected) -> JSONResponse:
        # 422：请求合法、处理也跑了，是**产物**不能交付。与 SEOError 的
        # `draft/evidence/outline` 阶段失败同一个语义档位。
        return JSONResponse(
            error_body(exc.code, "生成结果未通过输出检查", violations=list(exc.violations)),
            status_code=422,
        )

    @app.exception_handler(ApprovalRequired)
    async def _on_approval_required(
        request: Request, exc: ApprovalRequired
    ) -> JSONResponse:
        # 202：动作**合法但尚未执行**，等等审人。用 202 而不是 403——
        # 403 会让调用方以为"这事儿不能做"，而它其实只是"还没批"。
        return JSONResponse(
            error_body(
                exc.code,
                str(exc),
                approval_id=exc.request.id,
                summary=exc.request.summary,
                skill=exc.request.skill,
            ),
            status_code=202,
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

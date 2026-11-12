"""`/v1/tools` —— 工具面的运维与演示入口（W8）。

为什么单独开路由而不是把工具塞进 `/v1/answer`：

- 回答链路只该"用"检索，工具执行是**另一条链路**（提议 → 校验 → 授权 →
  审批 → 执行 → 验真）。塞进回答里会让一次 HTTP 请求同时承担两种失败语义
  （"检索没召回"与"发布被审批挂起"）。
- 执行循环（模型→工具→再问模型）属于 W9 的编排层。本周先把**单次受控执行**
  暴露出来，这样验收项（三个 badcase、审计字段、可见面收敛）都能用 curl
  复现，不依赖模型是否配合。

主体解析与 `/v1/answer` 有一处**故意的**不同：这里**总是**返回一个主体
（无请求头 = 匿名），而不是"ACL 没开就返回 None"。理由：检索 ACL 是一个
可以不开的开关，而工具权限是长期生效的规矩——`None` 会让 `permission_denied`
永远不出现，等于把白名单装成了摆设。
"""

from __future__ import annotations

from typing import Any

from agent_framework.llm.types import ToolCall
from agent_framework.observability import current_request_id
from agent_framework.retrieval import Principal
from agent_framework.skills import TaskStage, registry_snapshot
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from growth_agent.runtime import ensure_tools

router = APIRouter()


class ToolCallRequest(BaseModel):
    # 工具名（= Skill 名，与模型看到的完全一致）。
    name: str = Field(min_length=1, max_length=64)
    # 参数原样透传：schema 校验在执行器里做，路由不做第二套判断——
    # 两处校验迟早会给出不一致的答案。
    arguments: dict[str, Any] = Field(default_factory=dict)
    # 任务阶段。默认 draft：调研阶段调写工具会被可见面直接挡掉，
    # 而"误挡"比"误放"难排查得多。
    stage: str = Field(default=TaskStage.DRAFT.value)
    # 幂等意图 id。同一 (name, arguments, intent_id) 只会执行一次。
    intent_id: str = Field(default="", max_length=128)


@router.get("/v1/tools")
async def list_tools(
    request: Request,
    stage: str = Query(default=TaskStage.RESEARCH.value),
) -> JSONResponse:
    runtime = await ensure_tools(request.app)
    principal = _principal(request)
    decision = runtime.surface(principal=principal, stage=stage)

    return JSONResponse(
        {
            "request_id": current_request_id() or "-",
            "principal": {"subject_id": principal.subject_id, "groups": sorted(principal.groups)},
            **decision.as_dict(),
            # 渐进式加载的注入文本：调用方可以直接拿它拼进 system prompt。
            # 只有 name + description（约 20 token/个），选中后再取 details。
            "describe": decision.describe_all(),
            "skills": registry_snapshot(decision.visible),
            "reload_errors": list(runtime.registry.errors),
        }
    )


@router.get("/v1/tools/{name}")
async def describe_tool(name: str, request: Request) -> JSONResponse:
    """单个 Skill 的完整定义（选中之后才注入的正文）。"""
    runtime = await ensure_tools(request.app)
    spec = runtime.registry.get(name)
    if spec is None:
        return JSONResponse({"error": "tool_not_found", "name": name}, status_code=404)

    return JSONResponse(
        {
            "request_id": current_request_id() or "-",
            "name": spec.name,
            "description": spec.description,
            "mission": spec.mission,
            "principles": list(spec.principles),
            "instructions": spec.instructions,
            "parameters": spec.parameters,
            "permissions": sorted(spec.permissions),
            "sandbox": spec.sandbox,
            "stages": sorted(spec.stages),
            "dangerous": spec.dangerous,
            "done_criteria": [
                {"kind": check.kind, "target": check.target, "hint": check.hint,
                 "count": check.count}
                for check in spec.done_criteria
            ],
            "source": spec.source,
            # 可运行但可疑的写法：Review 时要看到。
            "lint": list(spec.lint()),
        }
    )


@router.post("/v1/tools/call")
async def call_tool(body: ToolCallRequest, request: Request) -> JSONResponse:
    runtime = await ensure_tools(request.app)
    principal = _principal(request)

    call = ToolCall(
        id=f"http-{current_request_id() or 'anon'}-{body.name}",
        name=body.name,
        arguments=body.arguments,
    )
    result = await runtime.actions.submit(
        call,
        principal=principal,
        stage=body.stage,
        intent_id=body.intent_id,
    )

    return JSONResponse(
        {
            "request_id": current_request_id() or "-",
            "action_id": result.action_id,
            "status": result.status,
            "detail": result.detail,
            "duplicate_of": result.duplicate_of,
            "loop_warning": result.loop_warning,
            "approval": result.approval,
            "notes": list(result.notes),
            "verify": None
            if result.verify is None
            else {"status": result.verify.status, "detail": result.verify.detail},
            # 回注给模型的那条 tool 消息。调用方把它塞回 messages 就完成闭环。
            "tool_message": None
            if result.message is None
            else {
                "role": "tool",
                "tool_call_id": result.message.tool_call_id,
                "name": result.message.name,
                "content": result.message.content,
            },
            "content": None
            if result.execution is None or result.execution.outcome is None
            else result.execution.outcome.result.content,
            "replayed": None
            if result.execution is None or result.execution.outcome is None
            else result.execution.outcome.replayed,
        }
    )


@router.post("/v1/tools/reload")
async def reload_tools(request: Request) -> JSONResponse:
    """热重载：新增 Schema 文件后调一次，主流程零改动。

    返回全部 Skill 名而不是"重载了几个"：调用方要能对比前后差异，
    而这只能靠完整清单（"重载了 0 个"无法区分"文件没变"和"文件写错了"）。
    """
    runtime = await ensure_tools(request.app)

    names = await runtime.reload()

    return JSONResponse(
        {
            "request_id": current_request_id() or "-",
            "skills": list(names),
            "errors": list(runtime.registry.errors),
            "generation": runtime.registry.generation,
        }
    )


def _principal(request: Request) -> Principal:
    """总是返回一个主体。无请求头 = 匿名（只能跑 read/net）。"""
    subject = (request.headers.get("X-Principal-Id") or "").strip()
    raw_groups = request.headers.get("X-Principal-Groups") or ""
    groups = frozenset(item.strip() for item in raw_groups.split(",") if item.strip())

    if not subject:
        return Principal.anonymous()
    return Principal(subject_id=subject, groups=groups)

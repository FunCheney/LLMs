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

import time
from typing import Any

from agent_framework.guardrails import (
    EVENT_APPROVAL_PENDING,
    EVENT_DANGEROUS_CALL,
    EVENT_TOOL_CALL,
    ApprovalError,
)
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

    # W10 行为监控：记录调用频次。**只记录不拦断**——批量场景下频率高是正常的，
    # 拦断会误伤；告警的消费者是值班与 W11 的熔断器，不是这里的状态码。
    spec = runtime.registry.get(body.name)
    runtime.behavior.record(
        EVENT_TOOL_CALL, scope=principal.subject_id, skill=body.name
    )
    if spec is not None and getattr(spec, "dangerous", False):
        runtime.behavior.record(
            EVENT_DANGEROUS_CALL, scope=principal.subject_id, skill=body.name
        )

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

    # 停在等审批也是一类"卡住"，单独计一次：它的告警阈值更宽（10 分钟 3 次），
    # 因为它不是安全问题，是流程吞吐问题。
    if result.status == "pending":
        runtime.behavior.record(
            EVENT_APPROVAL_PENDING, scope=principal.subject_id, skill=body.name
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
    """身份来源：**先看鉴权中间件写的 `request.state.principal`**，其次请求头。

    为什么要有第二条：`AUTH_MODE=off` 时（本地/测试）没有中间件，
    此时用 `X-Principal-Id` 方便调试。但这意味着**开鉴权后请求头不再被信任**——
    顺序必须是"可信来源优先"，否则带一个 `X-Principal-Id: admin` 就能
    绕过 key 冒充管理员。先查 state，正是为了让请求头只在
    "本来就没设防"的档位下起作用。
    """

    from_state = getattr(request.state, "principal", None)
    if isinstance(from_state, Principal):
        return from_state

    subject = (request.headers.get("X-Principal-Id") or "").strip()
    raw_groups = request.headers.get("X-Principal-Groups") or ""
    groups = frozenset(item.strip() for item in raw_groups.split(",") if item.strip())

    if not subject:
        return Principal.anonymous()
    return Principal(subject_id=subject, groups=groups)


# --------------------------------------------------------------------------- #
# HITL 审批通道（W10 D4）
# --------------------------------------------------------------------------- #


class ApprovalDecisionRequest(BaseModel):
    """人工决策。`by` 必填：一条没有决策人的审批记录在审计里等于没有。"""

    approved: bool
    by: str = Field(min_length=1, max_length=64, description="审批人标识")


@router.get("/v1/approvals")
async def list_approvals(request: Request) -> JSONResponse:
    """待审清单 + 审计尾部。

    顺带返回 `audit_tail` 是为了让"谁批的、什么时候批的"能在同一个响应里看到——
    分成两个端点之后，排障时必然要开两个 tab 对照。
    """

    runtime = await ensure_tools(request.app)
    pending = runtime.approvals.pending()

    return JSONResponse(
        {
            "request_id": current_request_id() or "-",
            "pending": [
                {
                    "id": item.id,
                    "skill": item.skill,
                    "arguments_digest": item.arguments_digest,
                    "summary": item.summary,
                    "reason": item.reason,
                    "created_at": item.created_at,
                    "expires_at": item.expires_at,
                    # 剩余秒数给前端做倒计时；人看不到"还有多久过期"就不会去处理。
                    "expires_in_s": round(max(0.0, item.expires_at - _now()), 1),
                }
                for item in pending
            ],
            "audit_tail": [
                {
                    "at": entry.at,
                    "action": entry.action,
                    "request_id": entry.request_id,
                    "skill": entry.skill,
                    "by": entry.by,
                    "note": entry.note,
                }
                for entry in runtime.approvals.audit_log[-20:]
            ],
            "behavior": runtime.behavior.snapshot(),
        }
    )


@router.post("/v1/approvals/{request_id}")
async def decide_approval(
    request_id: str, body: ApprovalDecisionRequest, request: Request
) -> JSONResponse:
    """批准或拒绝。

    三种失败语义分开：

    - 未知 id → 404（可能是过期后被清理，也可能是拼错）
    - 已终态 → 409（已经批过/拒过，**不能改历史**）
    - 成功 → 200 + 新状态
    """

    runtime = await ensure_tools(request.app)
    existing = runtime.approvals.get(request_id)
    if existing is None:
        return JSONResponse(
            {"error": "approval_not_found", "request_id_field": request_id},
            status_code=404,
        )

    try:
        decided = await runtime.approvals.decide(
            request_id, approved=body.approved, by=body.by
        )
    except ApprovalError as exc:
        return JSONResponse(
            {"error": "approval_conflict", "message": str(exc), "state": existing.status},
            status_code=409,
        )

    return JSONResponse(
        {
            "request_id": current_request_id() or "-",
            "id": decided.id,
            "skill": decided.skill,
            "status": decided.status,
            "by": decided.decided_by,
            "decided_at": decided.decided_at,
        }
    )


def _now() -> float:
    """当前时间。抽出来是为了让测试能 monkeypatch 单点，
    而不是在每个断言里算 `time.time() - created_at` 的浮点误差。"""

    return time.time()

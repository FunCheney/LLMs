"""Skill 执行器（W8 D4）：五道门，顺序固定，一步都不能换。

    A14 参数校验 → 可见面 → 权限 → 沙箱边界 → 执行 → 回注

为什么顺序是这个顺序（每一步都在回答"后面那些还要不要跑"）：

1. **可见面在权限之前？** 不，这里把"工具存不存在"放最前（`tool_not_found`），
   然后**可见面**（`tool_not_visible`），再**权限**（`permission_denied`）。
   两个码必须能分开：前者是"模型不该知道这个工具"（安全事件），
   后者是"看得见但没授权"（运营事件）。混淆它们，两类工单进同一个队列。
2. **参数校验在权限之前。** 参数不合法时连"要不要给权限"都不用问——
   而且先判参数能让"模型乱传参数"的日志更干净（不会每个都带一条权限记录）。
3. **沙箱在权限之后。** 沙箱判的是"这个动作能碰什么"，前提是主体有权做它。
   顺序反了会出现"越权用户拿到沙箱拒绝理由"的信息泄漏。
4. **沙箱必须在执行之前**（这是本周第 8 个返工点）：文件已经写了、
   进程已经起了才判断越界，等于只有日志没有边界。
5. **回注最后**，且失败也回注——模型需要知道"这次没成"，否则它会以为
   调用成功了然后接着编。

三个刻意的实现选择：

- **失败不抛异常**，一律返回带 `code` 的 `ToolExecution`。抛异常会逼调用方
  写 `except`，而那个 except 迟早会吞掉真正的编程错误。
- **审计在每条出口都记一次**（成功与四类失败都是）。只记成功的日志
  回答不了"为什么它没调"。
- **沙箱维度由参数名决定**（`path` / `url` / `command`）：这三个名字就是
  三道边界的入口。用命名约定而不是给每个 Skill 再加一份 `sandbox_spec`，
  是因为后者会和 `parameters` 里的同名参数漂移——两份"这个参数是路径"
  的声明，迟早有一份是错的。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..context.schema import ContextItem, ContextKind, TrustLevel
from ..llm.types import Message, MessageRole, ToolCall
from ..retrieval.acl import Principal
from .audit import AuditLog, ToolAudit, args_digest
from .contract import (
    CODE_INVALID_ARGUMENTS,
    CODE_NOT_FOUND,
    CODE_OK,
    CODE_PERMISSION_DENIED,
    CODE_SANDBOX_DENIED,
    CODE_TOOL_NOT_VISIBLE,
    ToolOutcome,
)
from .guard import PermissionGuard
from .loader import SkillRegistry
from .mcp.client import MCPClient
from .mcp.types import internal_name
from .sandbox import SandboxDecision, SandboxPolicy
from .spec import SkillSpec, TaskStage, validate_arguments
from .visibility import VisibilityDecision, VisibilityPolicy

logger = logging.getLogger(__name__)

__all__ = [
    "BOUNDARY_ARGS",
    "ResultTruncator",
    "SkillExecutor",
    "ToolExecution",
]

# 参数名 → 沙箱边界维度。**这三个名字是约定**：参数叫什么，就按哪道边界判。
BOUNDARY_ARGS: dict[str, str] = {
    "path": "path",
    "url": "network",
    "command": "command",
}

# 工具结果在 `tool_result` 槽位里的相关性基线。高于 history（0.1–0.5）
# 低于"用户刚问的那句话"（在 prompt 尾部，不参与排序）：工具结果是
# 本轮刚查到的，理应比旧对话更该留下。
TOOL_RESULT_SCORE = 0.7

_TRUNCATION_NOTE = "\n…（结果已截断，保留前 {kept} 字符；完整内容句柄：{handle}）"


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """一次执行的完整记录。**成功与失败同构**，这样调用方不用分两条路。"""

    code: str
    call: ToolCall
    skill: SkillSpec | None = None
    outcome: ToolOutcome | None = None
    # 回注用的消息（role=tool + tool_call_id）。失败时也有，内容是错误摘要。
    message: Message | None = None
    # 进 `tool_result` 槽位的候选（W7 的 ContextItem 五件套）。
    item: ContextItem | None = None
    detail: str = ""
    truncated: bool = False
    handle: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.code == CODE_OK


class ResultTruncator:
    """超长结果截断 + 句柄。

    与 W7 `WorkingMemory.truncate_tool_result` 同一套语义（截断必须留句柄），
    但**不共用实现**：那个在会话记忆里，生命周期跟着会话；工具结果的句柄
    跟着这次任务。共用会把"会话被清理"变成"工具结果取不回来"。
    """

    def __init__(self, *, max_chars: int = 2000) -> None:
        self._max_chars = max_chars
        self._handles: dict[str, str] = {}

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def truncate(self, text: str, *, handle: str) -> tuple[str, bool]:
        if len(text) <= self._max_chars:
            return text, False
        self._handles[handle] = text
        kept = text[: self._max_chars]
        return kept + _TRUNCATION_NOTE.format(kept=self._max_chars, handle=handle), True

    def refetch(self, handle: str) -> str | None:
        return self._handles.get(handle)


class SkillExecutor:
    """执行器。构造一次、全程复用（无请求级状态，主体与阶段都从参数进来）。"""

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        clients: Mapping[str, MCPClient],
        guard: PermissionGuard | None = None,
        sandbox: SandboxPolicy | None = None,
        visibility: VisibilityPolicy | None = None,
        audit: AuditLog | None = None,
        truncator: ResultTruncator | None = None,
    ) -> None:
        self._registry = registry
        # server 名 → client。执行器不认识 MCP，只认识"哪个 server 提供这个工具"。
        self._clients = dict(clients)
        self._guard = guard or PermissionGuard()
        self._sandbox = sandbox or SandboxPolicy()
        self._visibility = visibility or VisibilityPolicy(permissions=self._guard.policy)
        self._audit = audit or AuditLog()
        self._truncator = truncator or ResultTruncator()

    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @property
    def clients(self) -> Mapping[str, MCPClient]:
        """server 名 → client。给验真回读用（`action.py` 只读它，不改它）。"""
        return dict(self._clients)

    @property
    def sandbox(self) -> SandboxPolicy:
        return self._sandbox

    @property
    def guard(self) -> PermissionGuard:
        return self._guard

    def surface(
        self, *, principal: Principal | None = None, stage: TaskStage | str = TaskStage.RESEARCH
    ) -> VisibilityDecision:
        """当前 (principal, stage) 的可见面。`/v1/tools` 与 prompt 组装都用它。"""
        return self._visibility.surface(
            self._registry.all(), stage=stage, principal=principal
        )

    async def execute(
        self,
        call: ToolCall,
        *,
        principal: Principal | None = None,
        stage: TaskStage | str = TaskStage.RESEARCH,
        action_id: str = "",
        approval: str = "",
    ) -> ToolExecution:
        """执行一次工具调用。**任何失败都返回确定结果**，绝不静默。"""
        subject = principal or Principal.anonymous()
        resolved_stage = stage.value if isinstance(stage, TaskStage) else str(stage)

        # -- 1 工具存在性 -------------------------------------------------- #
        skill = self._registry.get(call.name)
        if skill is None:
            return self._fail(
                call,
                CODE_NOT_FOUND,
                f"没有名为 {call.name!r} 的 Skill",
                subject=subject,
                stage=resolved_stage,
                action_id=action_id,
            )

        # -- 2 可见面（阶段 + 主体粗筛） ------------------------------------ #
        decision = self.surface(principal=subject, stage=resolved_stage)
        if not decision.is_visible(skill.name):
            # 这里是安全事件：模型调了一个它不该知道的工具。可见面之外
            # 还调得到，说明 prompt 里的工具清单与执行层不一致，或被注入。
            logger.warning(
                "tool_not_visible: skill=%s stage=%s reason=%s",
                skill.name,
                resolved_stage,
                decision.reason(skill.name),
            )
            return self._fail(
                call,
                CODE_TOOL_NOT_VISIBLE,
                f"{skill.name} 在当前可见面之外（{decision.reason(skill.name)}）",
                subject=subject,
                stage=resolved_stage,
                action_id=action_id,
                skill=skill,
                notes=(f"visibility:{decision.reason(skill.name)}",),
            )

        # -- 3 A14 参数校验 ------------------------------------------------- #
        problems = validate_arguments(skill.parameters, call.arguments)
        if call.arguments_error:
            # 参数连 JSON 都没解开：这是最该被看见的一类跑偏，单独记。
            problems.insert(0, f"参数解析失败：{call.arguments_error}")
        if problems:
            return self._fail(
                call,
                CODE_INVALID_ARGUMENTS,
                "；".join(problems),
                subject=subject,
                stage=resolved_stage,
                action_id=action_id,
                skill=skill,
            )

        # -- 4 权限 -------------------------------------------------------- #
        permission = self._guard.check(subject, skill)
        if permission.denied:
            return self._fail(
                call,
                CODE_PERMISSION_DENIED,
                f"缺少权限 {list(permission.missing)}",
                subject=subject,
                stage=resolved_stage,
                action_id=action_id,
                skill=skill,
                risk=permission.risk,
                notes=tuple(f"missing:{item}" for item in permission.missing),
            )

        # -- 5 沙箱边界（副作用之前） --------------------------------------- #
        sandbox_decision = self._check_sandbox(call.arguments)
        if sandbox_decision is not None and sandbox_decision.denied:
            return self._fail(
                call,
                CODE_SANDBOX_DENIED,
                f"{sandbox_decision.code}：{sandbox_decision.detail}",
                subject=subject,
                stage=resolved_stage,
                action_id=action_id,
                skill=skill,
                risk=permission.risk,
                notes=(f"sandbox:{sandbox_decision.code}",),
            )

        # -- 6 执行（走 MCP client，回放/真调用由传输层决定） ---------------- #
        server, _, tool = skill.entrypoint.partition(".")
        client = self._clients.get(server)
        if client is None:
            return self._fail(
                call,
                CODE_NOT_FOUND,
                f"entrypoint 的 server {server!r} 未装配",
                subject=subject,
                stage=resolved_stage,
                action_id=action_id,
                skill=skill,
                risk=permission.risk,
            )

        outcome = await client.call_tool(internal_name(server, tool), call.arguments)
        if outcome.code != CODE_OK:
            return self._fail(
                call,
                outcome.code,
                json_safe_message(outcome),
                subject=subject,
                stage=resolved_stage,
                action_id=action_id,
                skill=skill,
                risk=permission.risk,
                outcome=outcome,
            )

        # -- 7 回注（截断留句柄） ------------------------------------------- #
        return self._succeed(
            call,
            skill,
            outcome,
            subject=subject,
            stage=resolved_stage,
            action_id=action_id,
            risk=permission.risk,
            approval=approval,
        )

    # -- 内部 -------------------------------------------------------------- #

    def _check_sandbox(self, arguments: Mapping[str, object]) -> SandboxDecision | None:
        """按参数名挑边界维度。多个维度同时出现时逐个判，返回第一个拒绝。"""
        for argument_name, dimension in BOUNDARY_ARGS.items():
            if argument_name not in arguments:
                continue
            value = arguments[argument_name]
            if isinstance(value, bytes):
                text = value.decode("utf-8", errors="replace")
            elif isinstance(value, str):
                text = value
            else:
                # 参数类型已经在 A14 校验过；这里遇到非字符串说明 schema 声明
                # 与边界约定不一致（比如 `path` 被声明成 integer）。不猜，跳过。
                continue

            if dimension == "path":
                decision = self._sandbox.check(path=text)
            elif dimension == "network":
                decision = self._sandbox.check(url=text)
            else:
                decision = self._sandbox.check(command=text)

            if decision.denied:
                return decision
        return None

    def _succeed(
        self,
        call: ToolCall,
        skill: SkillSpec,
        outcome: ToolOutcome,
        *,
        subject: Principal,
        stage: str,
        action_id: str,
        risk: str,
        approval: str,
    ) -> ToolExecution:
        content = str(outcome.result.content or "")
        handle = f"{skill.name}:{call.id}"
        text, truncated = self._truncator.truncate(content, handle=handle)

        message = Message(
            role=MessageRole.TOOL,
            tool_call_id=call.id,
            name=skill.name,
            content=text,
        )
        item = ContextItem(
            id=f"tool-{call.id or skill.name}",
            content=text,
            kind=ContextKind.TOOL_RESULT,
            source=f"tool:{skill.name}",
            version=skill.version,
            # **信任级 external**：工具/网页返回的内容永远不是指令。
            # 这是 W10 注入防护在 Context 层的同一条原则（W7 已立）。
            trust=TrustLevel.EXTERNAL,
            score=TOOL_RESULT_SCORE,
            message=message,
            note=",".join(outcome.notes) if outcome.notes else "",
        )

        self._audit.record(
            ToolAudit(
                action_id=action_id or call.id,
                tool=f"{skill.entrypoint}",
                skill=skill.name,
                status=CODE_OK,
                risk=risk,
                principal=subject.subject_id,
                args_digest=args_digest(call.arguments),
                arg_keys=tuple(sorted(call.arguments)),
                approval=approval,
                stage=stage,
                notes=(("truncated",) if truncated else ()) + outcome.notes,
            ),
            outcome=outcome,
        )

        return ToolExecution(
            code=CODE_OK,
            call=call,
            skill=skill,
            outcome=outcome,
            message=message,
            item=item,
            truncated=truncated,
            handle=handle if truncated else "",
        )

    def _fail(
        self,
        call: ToolCall,
        code: str,
        detail: str,
        *,
        subject: Principal,
        stage: str,
        action_id: str,
        skill: SkillSpec | None = None,
        risk: str = "read",
        outcome: ToolOutcome | None = None,
        notes: tuple[str, ...] = (),
    ) -> ToolExecution:
        """失败出口。**也要回注**：模型必须知道这次没成。

        回注内容用 `{"error": code}` 形态（不是自然语言）：模型对结构化错误
        的"改参数重试"行为明显好于对一段散文，而验真层也要按字段判。
        """
        self._audit.record(
            ToolAudit(
                action_id=action_id or call.id,
                tool=skill.entrypoint if skill else call.name,
                skill=skill.name if skill else "",
                status=code,
                risk=risk,
                principal=subject.subject_id,
                args_digest=args_digest(call.arguments),
                arg_keys=tuple(sorted(call.arguments)),
                stage=stage,
                notes=notes,
            ),
            outcome=outcome,
        )

        message = Message(
            role=MessageRole.TOOL,
            tool_call_id=call.id,
            name=call.name,
            content=f'{{"error": "{code}", "message": "{detail}"}}',
        )
        return ToolExecution(
            code=code,
            call=call,
            skill=skill,
            outcome=outcome,
            message=message,
            detail=detail,
            notes=notes,
        )


def json_safe_message(outcome: ToolOutcome) -> str:
    """错误结果的简短说明（用于 `ToolExecution.detail` 与回注）。

    只取我们自己的错误码，不带上游原文——上游原文在 `outcome.result` 里
    已经是规范化过的形态了。
    """
    return f"{outcome.code}（{outcome.tool or outcome.server}）"

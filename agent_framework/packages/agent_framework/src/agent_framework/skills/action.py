"""Action Runtime（W8 D5）：把 `ToolCall` 从"意图"变成"受控事实"。

**为什么 `ToolCall` 不能直接等于执行。** 它是模型的意图，不是事实。模型可能
在同一任务里重复发起同一笔写操作，可能被上下文里的恶意文本诱导改参数，
也可能遇到"接口返回 200 但其实没生效"。如果 `ToolCall` 一产生副作用，
这三类问题都没有拦截点——所以中间隔一层 Action：

| 阶段 | 谁负责 | 做什么 | 失败时 |
|---|---|---|---|
| propose | 模型 | 生成 `ToolCall`（只是意图） | — |
| validate | 代码 | schema（A14）+ 业务规则（上限、互斥） | `invalid_action` |
| authorize | 代码 | 权限白名单 + 风险分级 | `forbidden` |
| approve | **人**（仅高风险） | 二次确认，展示"将要发生什么" | `pending` / `rejected` |
| execute | 代码 | **带幂等键**执行，超时受控 | `failed`，可重试 |
| verify | 代码 | 回读真实状态确认生效（**不信 200**） | `unverified`，告警 |

三条设计纪律：

1. **幂等键 = hash(tool + 规范化 args + 意图 id)，TTL 24h。** 键里带意图 id
   而不是 `ToolCall.id`：`ToolCall.id` 每次重试都会变，用它做键等于"重试
   就是新动作"，那幂等就失效了。意图 id 由调用方按"这一轮要干的事"给。
2. **并发重复不能靠"先查后写"。** 三个并发请求同时查缓存都 miss，然后三个
   都执行了——用 per-key 锁 + 双重检查，而不是靠时序。
3. **验真只做"读回状态"一种。** 复杂对账要真实业务系统（W11 才有）。
   抓到 `{"ok": false}` 与"回读状态不对"就够了：这两类覆盖了线上绝大多数
   "假成功"。

**死循环预警复用 W7 的 Drift。** 最近 3 步里同一 `(tool, args)` 出现 ≥2 次
就置 `loop_warning`——但**只报警不阻断**（W7 ADR：终止策略是 W9 多 Agent
的职责）。阻断会把一次误报变成一次任务失败，而这比多花一点 token 贵得多。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from ..context.drift import REPEAT_ACTION_COUNT, REPEAT_ACTION_WINDOW, check_drift
from ..llm.types import Message, ToolCall
from ..retrieval.acl import Principal
from .audit import ToolAudit, args_digest
from .executor import SkillExecutor, ToolExecution
from .guard import RISK_READ, PermissionGuard, risk_of
from .mcp.types import internal_name
from .replay import canonical_args
from .spec import SkillSpec, TaskStage

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TTL_S",
    "STATUS_DUPLICATE",
    "STATUS_FAILED",
    "STATUS_FORBIDDEN",
    "STATUS_INVALID",
    "STATUS_NOT_VISIBLE",
    "STATUS_PENDING",
    "STATUS_REJECTED",
    "STATUS_UNVERIFIED",
    "STATUS_VERIFIED",
    "Action",
    "ActionResult",
    "ActionRuntime",
    "NumericCapRule",
    "VerifyOutcome",
    "make_action_id",
]

# 幂等缓存存活时间。24h 对应"人的一次工作日"：跨天重放同一动作，
# 更可能是新需求而不是重复提交。
DEFAULT_TTL_S = 24 * 60 * 60.0

STATUS_VERIFIED = "verified"
STATUS_UNVERIFIED = "unverified"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"
STATUS_REJECTED = "rejected"
STATUS_FORBIDDEN = "forbidden"
STATUS_INVALID = "invalid_action"
STATUS_DUPLICATE = "duplicate"
# 工具不在当前 (principal, stage) 的可见面内。与 STATUS_FORBIDDEN 分开：
# 前者是"不该知道它存在"（安全事件），后者是"看得见但没授权"（运营事件）。
STATUS_NOT_VISIBLE = "tool_not_visible"

VERIFY_VERIFIED = "verified"
VERIFY_UNVERIFIED = "unverified"
VERIFY_SKIPPED = "skipped"


class BusinessRule(Protocol):
    """业务规则。与 schema 分工：schema 管"形状"，规则管"语义"。"""

    name: str

    def check(self, skill: SkillSpec, arguments: Mapping[str, object]) -> str | None:
        """返回违规说明；None = 通过。"""
        ...


@dataclass(frozen=True, slots=True)
class NumericCapRule:
    """数值上限。这是"参数篡改"badcase 的拦截点。

    为什么不用 schema 的 `maximum`：schema 的 `maximum` 是**给模型看的契约**
    （它会照着改），而业务上限是**我们的规矩**（超了必须拒，不管模型怎么想）。
    两者可以不同：比如 schema 允许 `budget` ≤ 100 万（历史遗留），
    而业务只批到 10 万。混在一起，改业务上限就得动 schema，
    等于把"给模型看的说明书"当成了"授权系统"。
    """

    field: str
    maximum: float
    name: str = "numeric_cap"

    def check(self, skill: SkillSpec, arguments: Mapping[str, object]) -> str | None:
        value = arguments.get(self.field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{self.field} 必须是数字，收到 {type(value).__name__}"
        if value > self.maximum:
            return (
                f"{self.field}={value} 超过业务上限 {self.maximum:g}"
                f"（schema 允许不等于业务批准）"
            )
        return None


@dataclass(frozen=True, slots=True)
class Action:
    """一个被提议的动作（还没执行）。`frozen` 是为了让"审批时看到的东西"
    与"执行的东西"是同一个对象——中间被改过参数，审批就白批了。"""

    action_id: str
    call: ToolCall
    skill: SkillSpec
    principal: Principal
    risk: str
    idempotency_key: str
    stage: str = ""
    intent_id: str = ""


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    status: str = VERIFY_SKIPPED
    detail: str = ""

    @property
    def failed(self) -> bool:
        """只有**明确判定没生效**才算失败。

        `skipped` 不是失败：只读动作没有"生效"可验，把它算成失败会让每一次
        搜索都变成 `unverified`，而这个信号一旦处处出现就等于没有信号。
        代价是"未声明 verify 的写动作"也会被算成成功——但它会在
        `notes` 里留 `verify:skipped`，报告能数出来，不是无痕的。
        """
        return self.status == VERIFY_UNVERIFIED


@dataclass(frozen=True, slots=True)
class ActionResult:
    """一次动作的终局。`status` 是**唯一**可被上层分支的字段。"""

    action_id: str
    status: str
    execution: ToolExecution | None = None
    verify: VerifyOutcome | None = None
    detail: str = ""
    # 命中幂等：指向首次执行的 action_id，`execution` 也是首次的那份结果。
    duplicate_of: str | None = None
    # 最近窗口内重复调用（死循环前兆）。只报警，不阻断。
    loop_warning: bool = False
    approval: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """只有"执行成功 **且** 验真通过"才算成功。"""
        return self.status == STATUS_VERIFIED

    @property
    def message(self) -> Message | None:
        """要回注给模型的那条消息（可能为 None）。"""
        return self.execution.message if self.execution else None


@dataclass(slots=True)
class _CacheEntry:
    result: ActionResult
    stored_at: float


def make_action_id(call: ToolCall) -> str:
    """动作 id。优先用模型给的 `ToolCall.id`（对账时能直接对上），
    缺了才自己生成——**不覆盖**模型给的值，否则日志里两套 id 对不上。"""
    return call.id or f"act-{uuid.uuid4().hex[:12]}"


class ActionRuntime:
    """提议 → 校验 → 授权 → 审批 → 执行 → 验真。"""

    def __init__(
        self,
        *,
        executor: SkillExecutor,
        guard: PermissionGuard | None = None,
        rules: Sequence[BusinessRule] = (),
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._executor = executor
        self._guard = guard or executor.guard
        self._rules = tuple(rules)
        # 注入时钟而不是直接用 time.time：TTL 测试不能靠 sleep 24 小时。
        self._clock: Callable[[], float] = clock or time.time
        self._ttl_s = ttl_s
        self._cache: dict[str, _CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # 最近动作序列（死循环预警用）。窗口与 Drift 的常量对齐。
        self._recent: deque[tuple[str, str]] = deque(maxlen=REPEAT_ACTION_WINDOW)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # -- 提议 -------------------------------------------------------------- #

    def propose(
        self,
        call: ToolCall,
        *,
        principal: Principal | None = None,
        stage: TaskStage | str = TaskStage.DRAFT,
        intent_id: str = "",
    ) -> Action | None:
        """把 `ToolCall` 包成 `Action`。未知 Skill 返回 None（由 `submit` 转成
        `invalid_action`）——**不**在这里抛异常：未知工具名是模型的常见跑偏，
        不是程序错误。
        """
        skill = self._executor.registry.get(call.name)
        if skill is None:
            return None

        subject = principal or Principal.anonymous()
        return Action(
            action_id=make_action_id(call),
            call=call,
            skill=skill,
            principal=subject,
            risk=risk_of(skill),
            idempotency_key=idempotency_key(call, intent_id=intent_id),
            stage=stage.value if isinstance(stage, TaskStage) else str(stage),
            intent_id=intent_id,
        )

    async def submit(
        self,
        call: ToolCall,
        *,
        principal: Principal | None = None,
        stage: TaskStage | str = TaskStage.DRAFT,
        intent_id: str = "",
    ) -> ActionResult:
        """提议 + 执行。产品层（HTTP 路由）用这个入口就够。"""
        action = self.propose(call, principal=principal, stage=stage, intent_id=intent_id)
        if action is None:
            self._audit(
                ToolAudit(
                    action_id=make_action_id(call),
                    tool=call.name,
                    status=STATUS_INVALID,
                    principal=(principal or Principal.anonymous()).subject_id,
                    args_digest=args_digest(call.arguments),
                    arg_keys=tuple(sorted(call.arguments)),
                    stage=stage.value if isinstance(stage, TaskStage) else str(stage),
                )
            )
            return ActionResult(
                action_id=make_action_id(call),
                status=STATUS_INVALID,
                detail=f"未知 Skill {call.name!r}",
            )
        return await self.run(action)

    # -- 执行 -------------------------------------------------------------- #

    async def run(self, action: Action) -> ActionResult:
        self._prune()

        # 0) 记录本次尝试（死循环预警）。**必须在幂等检查之前**：
        #    "同一动作连发 3 次"正是死循环最典型的样子，而它在幂等层
        #    就已经被拦成 duplicate 了——把计数放在执行之后，这个信号
        #    永远不会出现（幂等生效的同时把预警能力关掉了）。
        loop_warning = self._note_action(action)

        # 1) 幂等命中：**在一切校验之前**。重复请求不该重新走审批，
        #    也不该因为"这次参数刚好不合规"而返回一个不同答案。
        cached = self._cache.get(action.idempotency_key)
        if cached is not None:
            return replace(
                cached.result,
                status=STATUS_DUPLICATE,
                duplicate_of=cached.result.action_id,
                detail=f"命中幂等键（首次 {cached.result.action_id}），未重复执行",
                loop_warning=loop_warning,
            )

        lock = self._locks.setdefault(action.idempotency_key, asyncio.Lock())
        async with lock:
            # 双重检查：等锁期间别的并发请求可能已经执行完。
            cached = self._cache.get(action.idempotency_key)
            if cached is not None:
                return replace(
                    cached.result,
                    status=STATUS_DUPLICATE,
                    duplicate_of=cached.result.action_id,
                    detail=f"命中幂等键（首次 {cached.result.action_id}），未重复执行",
                    loop_warning=loop_warning,
                )

            result = await self._execute(action, loop_warning=loop_warning)
            if result.status in {STATUS_VERIFIED, STATUS_UNVERIFIED}:
                # 只有"真的执行过"才进缓存。失败/待审批不缓存：
                # 缓存一个 pending 会让审批通过后的重试直接拿到 pending。
                self._cache[action.idempotency_key] = _CacheEntry(
                    result=result, stored_at=float(self._clock())
                )
            return result

    async def _execute(self, action: Action, *, loop_warning: bool = False) -> ActionResult:
        # -- validate：业务规则 -------------------------------------------- #
        for rule in self._rules:
            problem = rule.check(action.skill, action.call.arguments)
            if problem:
                self._audit(
                    ToolAudit(
                        action_id=action.action_id,
                        tool=action.skill.entrypoint,
                        skill=action.skill.name,
                        status=STATUS_INVALID,
                        risk=action.risk,
                        principal=action.principal.subject_id,
                        args_digest=args_digest(action.call.arguments),
                        arg_keys=tuple(sorted(action.call.arguments)),
                        stage=action.stage,
                        notes=(f"rule:{rule.name}",),
                    )
                )
                return ActionResult(
                    action_id=action.action_id,
                    status=STATUS_INVALID,
                    detail=f"业务规则 {rule.name} 拒绝：{problem}",
                    notes=(f"rule:{rule.name}",),
                )

        # -- visibility（必须在授权之前） ------------------------------------ #
        # 先判"看不看得见"再判"有没有权"：一个被可见面裁掉的工具被强行调用，
        # 要的是 `tool_not_visible`（模型/注入不该知道它存在），而不是
        # `forbidden`（主体缺权限）。两个码在值班队列里不是同一类工单。
        surface_decision = self._executor.surface(
            principal=action.principal, stage=action.stage
        )
        if not surface_decision.is_visible(action.skill.name):
            reason = surface_decision.reason(action.skill.name)
            self._audit(
                ToolAudit(
                    action_id=action.action_id,
                    tool=action.skill.entrypoint,
                    skill=action.skill.name,
                    status=STATUS_NOT_VISIBLE,
                    risk=action.risk,
                    principal=action.principal.subject_id,
                    args_digest=args_digest(action.call.arguments),
                    arg_keys=tuple(sorted(action.call.arguments)),
                    stage=action.stage,
                    notes=(f"visibility:{reason}",),
                )
            )
            return ActionResult(
                action_id=action.action_id,
                status=STATUS_NOT_VISIBLE,
                detail=f"{action.skill.name} 在当前可见面之外（{reason}）",
                notes=(f"visibility:{reason}",),
            )

        # -- authorize ------------------------------------------------------ #
        permission = self._guard.check(action.principal, action.skill)
        if permission.denied:
            # 真正的权限判定发生在执行器里（那里也要判，因为执行器可被单独使用）。
            # 这里提前判一次是为了让"没权限"在**审批之前**就被拒——
            # 不能让审批人去批一件主体根本无权做的事。
            return ActionResult(
                action_id=action.action_id,
                status=STATUS_FORBIDDEN,
                detail=f"缺少权限 {list(permission.missing)}",
                notes=tuple(f"missing:{item}" for item in permission.missing),
            )

        # -- approve（仅高风险） -------------------------------------------- #
        approval = ""
        if self._guard.needs_approval(action.risk):
            decision = await self._guard.approve(action.skill, action.call.arguments)
            approval = decision.status
            if not decision.approved:
                status = STATUS_REJECTED if decision.status == "rejected" else STATUS_PENDING
                self._audit(
                    ToolAudit(
                        action_id=action.action_id,
                        tool=action.skill.entrypoint,
                        skill=action.skill.name,
                        status=status,
                        risk=action.risk,
                        principal=action.principal.subject_id,
                        args_digest=args_digest(action.call.arguments),
                        arg_keys=tuple(sorted(action.call.arguments)),
                        approval=approval,
                        stage=action.stage,
                        notes=("approval_required",),
                    )
                )
                return ActionResult(
                    action_id=action.action_id,
                    status=status,
                    detail=decision.reason or decision.summary,
                    approval=approval,
                )

        # -- execute -------------------------------------------------------- #
        execution = await self._executor.execute(
            action.call,
            principal=action.principal,
            stage=action.stage,
            action_id=action.action_id,
            approval=approval,
        )

        if not execution.ok:
            return ActionResult(
                action_id=action.action_id,
                status=STATUS_FAILED,
                execution=execution,
                verify=VerifyOutcome(VERIFY_SKIPPED, "执行失败，无需验真"),
                detail=execution.detail or execution.code,
                loop_warning=loop_warning,
                approval=approval,
                notes=(f"code:{execution.code}",),
            )

        # -- verify --------------------------------------------------------- #
        verify = await self._verify(action, execution)
        status = STATUS_UNVERIFIED if verify.failed else STATUS_VERIFIED
        if verify.failed:
            logger.warning(
                "action unverified: action_id=%s skill=%s detail=%s",
                action.action_id,
                action.skill.name,
                verify.detail,
            )

        return ActionResult(
            action_id=action.action_id,
            status=status,
            execution=execution,
            verify=verify,
            detail=verify.detail,
            loop_warning=loop_warning,
            approval=approval,
            # verify 的结果必须可数：`verify:skipped` 让"这次其实没验"
            # 在报告里是可见的，而不是悄悄算作通过。
            notes=execution.notes + (f"verify:{verify.status}",),
        )

    # -- 验真 -------------------------------------------------------------- #

    async def _verify(self, action: Action, execution: ToolExecution) -> VerifyOutcome:
        if action.risk == RISK_READ:
            # 只读动作没有"生效"这回事。给它造一个验真步骤，
            # 结果是每次都 verified 的空动作——那会稀释"unverified"这个信号。
            return VerifyOutcome(VERIFY_SKIPPED, "只读动作无需验真")

        payload = execution.outcome.payload if execution.outcome else None

        # 1) 响应体自己说没生效（badcase #3：接口 200 但 ok=false）
        if isinstance(payload, Mapping) and payload.get("ok") is False:
            reason = str(payload.get("reason") or "")
            return VerifyOutcome(
                VERIFY_UNVERIFIED,
                f"工具返回 ok=false{f'（{reason}）' if reason else ''}：HTTP 成功不等于生效",
            )

        spec = action.skill.verify
        if spec is None:
            # 有副作用却没声明验真方式：如实标成"没验"，而不是默认通过。
            return VerifyOutcome(VERIFY_SKIPPED, "Skill 未声明 verify：本次没有回读状态")

        read_back_args = spec.read_back_args(payload)
        if read_back_args is None:
            return VerifyOutcome(
                VERIFY_UNVERIFIED,
                f"执行结果里没有 {spec.id_field}，无法回读确认",
            )

        server, _, _tool = spec.entrypoint.partition(".")
        client = self._executor.clients.get(server)
        if client is None:
            return VerifyOutcome(VERIFY_UNVERIFIED, f"验真用的 server {server!r} 未装配")

        try:
            # 回读是**系统动作**，不受模型可见面约束：它读的是我们刚写的东西，
            # 且目标 entrypoint 来自 Skill 定义（不是模型输入），没有越权面。
            read_back = await client.call_tool(
                internal_name(server, _tool), read_back_args
            )
        except Exception as exc:  # noqa: BLE001 - 验真失败不能让动作变成"成功"
            logger.warning("verify read-back failed: %s", exc)
            return VerifyOutcome(VERIFY_UNVERIFIED, f"回读失败：{type(exc).__name__}")

        if read_back.code != "ok" or not isinstance(read_back.payload, Mapping):
            return VerifyOutcome(VERIFY_UNVERIFIED, f"回读失败：{read_back.code}")

        if not spec.is_ok(read_back.payload):
            status_value = read_back.payload.get(spec.status_field)
            return VerifyOutcome(
                VERIFY_UNVERIFIED,
                f"回读状态为 {status_value!r}，不在 {list(spec.ok_status)} 内",
            )

        return VerifyOutcome(VERIFY_VERIFIED, "回读状态确认生效")

    # -- 内部 -------------------------------------------------------------- #

    def _note_action(self, action: Action) -> bool:
        """记录动作并返回"是否疑似死循环"。"""
        entry = (action.skill.name, action.idempotency_key)
        self._recent.append(entry)

        report = check_drift(goal="", payload_texts=(), recent_actions=list(self._recent))
        return report.max_repeat >= REPEAT_ACTION_COUNT

    def _prune(self) -> None:
        now = float(self._clock())
        expired = [
            key for key, entry in self._cache.items() if now - entry.stored_at > self._ttl_s
        ]
        for key in expired:
            self._cache.pop(key, None)
            self._locks.pop(key, None)
        if len(self._cache) > 1000:
            # 兜底：极端情况下（大量不同动作）不留无限内存。
            for key in list(self._cache)[: len(self._cache) - 1000]:
                self._cache.pop(key, None)
                self._locks.pop(key, None)

    def _audit(self, entry: ToolAudit) -> None:
        self._executor.audit.record(entry)

    def reset(self) -> None:
        """清空幂等缓存与最近动作序列（测试与"换一个任务"用）。"""
        self._cache.clear()
        self._locks.clear()
        self._recent.clear()


def idempotency_key(call: ToolCall, *, intent_id: str = "") -> str:
    """`hash(tool + 规范化 args + 意图 id)`。

    为什么带 `intent_id` 而不是 `ToolCall.id`：见模块 docstring 纪律 1。
    默认空串 = "同一份参数只该执行一次"，这是最安全也最有用的默认。
    """
    raw = f"{call.name}\x00{canonical_args(call.arguments)}\x00{intent_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

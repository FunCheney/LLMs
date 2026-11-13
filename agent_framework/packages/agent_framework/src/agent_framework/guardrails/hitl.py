"""HITL 审批门（W10 D4）：把 W8 的 `dangerous` 占位升级成真的人工确认。

W8 的 `PermissionGuard` 在 `irreversible` 风险下返回 `pending` 并**不执行**，
那是对的默认值，但它只是"停在原地"——没有通道、没有状态、没有人能把它放行。
本章补上这条通道，并且把红线写进代码：

**未审批的 dangerous 动作绝不执行。**
不是"记一条 warning 然后继续"，是抛 `ApprovalRequired`。

## 为什么只有 `dangerous` 需要审批

把审批加到所有调用上，产品会在第一天就被关掉：审批疲劳之后，人会不看
summary 直接点同意，那时审批门等于一个延时器。所以只有三类动作过门：

- 发布（`publish` 权限）——发出去收不回；
- 花钱（显式声明的 `dangerous`）；
- 外发（对外发消息/文件）。

判据在 W8 的 `risk_of()` 里按**权限**推导，不按工具名——理由见 `skills/guard.py`。

## 过期为什么默认拒绝

`pending → expired` 的转移里，"过期"结论只能是拒绝。反过来（过期后放行）
意味着"没人看就等于批准"，那么攻击者只需要挑凌晨提交。

## 幂等：审批过的动作重复请求不再执行第二次

`as_approver()` 会把"同一 (skill, 参数摘要) 且已批准的请求"直接判成
`approved`。这**不是**放行两次：W8 的 `ActionRuntime` 有幂等缓存，
第二次调用命中缓存返回第一次的结果。两层一起，才既不重复执行、
也不重复要人批。

## 为什么审计里只存摘要不存参数全文

审批摘要里已经只放前 80 字符（W8 `PermissionGuard.summarize`），
这里再进一步：审计只存 `arguments_digest`。理由是这类动作的参数里
经常带目标地址、账号、私钥片段——一份"为了审计"而存下全部参数的日志，
本身就是新的泄露面。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "APPROVED",
    "EXPIRED",
    "PENDING",
    "REJECTED",
    "ApprovalAuditEntry",
    "ApprovalError",
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalRequired",
    "arguments_digest",
    "skill_of",
]

logger = logging.getLogger(__name__)

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
EXPIRED = "expired"

TERMINAL = frozenset({APPROVED, REJECTED, EXPIRED})

# 默认 15 分钟。太短会让人来不及处理（尤其是异步通知场景），
# 太长则"pending"会在系统里堆积，W11 的 approval_backlog 告警会一直响。
DEFAULT_TTL_S = 900.0


class ApprovalError(RuntimeError):
    """审批流程里的状态错误（重复决策、未知 id、已过期仍要批）。"""


class ApprovalRequired(RuntimeError):
    """**安全红线**：该动作需要审批但还没有获批。

    单独一个异常类型而不是复用 `ApprovalError`：调用方要能把"需要人批"
    与"审批流程出错了"分开处理——前者是正常业务分支（返回 202 + 待审 id），
    后者是故障（500）。
    """

    code = "APPROVAL_REQUIRED"

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__(f"动作需要人工审批（{request.id}）：{request.summary}")
        self.request = request


def arguments_digest(skill: str, arguments: Mapping[str, object]) -> str:
    """参数摘要。**规范化**：键排序 + 紧凑分隔符。

    为什么不用 `repr(dict)`：字典顺序不同会被算成两个摘要，于是"同一个动作
    重复请求"变成两次审批，人会被同一件事问两遍。
    """

    payload = json.dumps(
        {"skill": skill, "arguments": dict(arguments)},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ApprovalRequest:
    """一次待审批请求。可变：状态会流转，但只通过 `ApprovalGate` 改。"""

    id: str
    skill: str
    arguments_digest: str
    # 给人看的"将要发生什么"。没有它，审批就是盲签。
    summary: str
    reason: str = ""
    status: str = PENDING
    created_at: float = 0.0
    expires_at: float = 0.0
    decided_at: float | None = None
    decided_by: str = ""

    @property
    def approved(self) -> bool:
        return self.status == APPROVED

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


@dataclass(frozen=True)
class ApprovalAuditEntry:
    """审计条目。只追加，不改写——审批历史被改写等于没有审计。"""

    at: float
    action: str  # requested | approved | rejected | expired | blocked
    request_id: str
    skill: str
    arguments_digest: str
    by: str = ""
    note: str = ""


class AuditSink(Protocol):
    def __call__(self, entry: ApprovalAuditEntry) -> None: ...


class ApprovalGate:
    """审批门。内存实现（单实例够用；多副本要换成共享存储，见 runbook 的缺口说明）。"""

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self._ttl = ttl_s
        self._clock = clock or time.time
        self._audit_sink = audit
        self._requests: dict[str, ApprovalRequest] = {}
        self._log: list[ApprovalAuditEntry] = []

    # -- 查询 -------------------------------------------------------------- #

    def get(self, request_id: str) -> ApprovalRequest | None:
        request = self._requests.get(request_id)
        if request is not None:
            self._expire_if_due(request)
        return request

    def pending(self) -> tuple[ApprovalRequest, ...]:
        """当前待审清单。顺带做过期扫描——**读的时候扫**，
        比后台定时任务可靠：没有后台任务在跑的环境（测试、CLI）也一样对。
        """

        for request in self._requests.values():
            self._expire_if_due(request)
        return tuple(
            request for request in self._requests.values() if request.status == PENDING
        )

    @property
    def audit_log(self) -> tuple[ApprovalAuditEntry, ...]:
        return tuple(self._log)

    # -- 申请 / 决策 -------------------------------------------------------- #

    async def require(
        self,
        skill: Any,
        arguments: Mapping[str, object],
        *,
        summary: str = "",
        reason: str = "",
    ) -> ApprovalRequest:
        """创建（或复用）一条待审批请求。

        `skill` 可以是 `SkillSpec`，也可以是带 `name` 属性/字符串的名字——
        这里只用它取名字，不碰权限逻辑（那是 W8 `PermissionGuard` 的事）。
        """

        name = skill_of(skill)
        digest = arguments_digest(name, arguments)

        existing = self._find_pending(name, digest)
        if existing is not None:
            # 同一动作再次请求 → 复用同一条，不新开一条。
            # 否则人会在待审列表里看到 5 条一模一样的请求。
            return existing

        now = self._clock()
        request = ApprovalRequest(
            id=uuid.uuid4().hex[:12],
            skill=name,
            arguments_digest=digest,
            summary=summary or f"{name}（参数摘要 {digest}）",
            reason=reason or "需要人工确认",
            created_at=now,
            expires_at=now + self._ttl,
        )
        self._requests[request.id] = request
        self._emit("requested", request)
        return request

    async def decide(
        self, request_id: str, *, approved: bool, by: str
    ) -> ApprovalRequest:
        """人工决策。已终态的请求不能再改——**改历史等于没有审计**。"""

        request = self._requests.get(request_id)
        if request is None:
            raise ApprovalError(f"未知审批请求：{request_id}")

        self._expire_if_due(request)
        if request.status != PENDING:
            raise ApprovalError(
                f"请求 {request_id} 已是 {request.status}，不能再决策"
            )

        request.status = APPROVED if approved else REJECTED
        request.decided_at = self._clock()
        request.decided_by = by or "-"
        self._emit("approved" if approved else "rejected", request, by=request.decided_by)
        return request

    # -- 执行前的门 --------------------------------------------------------- #

    async def guard_execute(
        self,
        skill: Any,
        arguments: Mapping[str, object],
        *,
        summary: str = "",
    ) -> ApprovalRequest | None:
        """执行前的最后一道。

        返回值：None = 这个动作不需要审批（可以执行）；有值 = 已获批的请求。
        **不是 `pending` 就一定抛 `ApprovalRequired`**——这是本周的红线，
        没有任何参数能把它关掉（没有 `skip_approval`、没有 `force`）。
        """

        name = skill_of(skill)
        if not _is_dangerous(skill):
            # 非 dangerous 动作直接放行，不产生审批记录：
            # 给每次调用都留一条 pending 记录，等于把审计淹掉。
            return None

        digest = arguments_digest(name, arguments)
        existing = self._find_pending(name, digest)
        if existing is not None:
            self._emit("blocked", existing, note="未审批即尝试执行")
            raise ApprovalRequired(existing)

        approved = self._find_approved(name, digest)
        if approved is not None:
            return approved

        request = await self.require(skill, arguments, summary=summary)
        self._emit("blocked", request, note="未审批即尝试执行")
        raise ApprovalRequired(request)

    # -- 接 W8 的 approver 钩子 --------------------------------------------- #

    def as_approver(self) -> Callable[[Any, Mapping[str, object], str], Awaitable[str]]:
        """返回符合 `PermissionGuard` 期望签名的审批人回调。

        `PermissionGuard(approver=gate.as_approver())` 之后，W8 的
        `risk_of → approve()` 流程就接到了真正的通道上：第一次返回
        `pending`（人还没批），批完之后同一动作返回 `approved`。
        """

        async def approver(
            skill: Any, arguments: Mapping[str, object], summary: str
        ) -> str:
            name = skill_of(skill)
            digest = arguments_digest(name, arguments)
            approved = self._find_approved(name, digest)
            if approved is not None:
                return APPROVED
            request = await self.require(skill, arguments, summary=summary)
            return request.status if request.status != PENDING else PENDING

        return approver

    # -- 内部 -------------------------------------------------------------- #

    def _find_pending(self, skill: str, digest: str) -> ApprovalRequest | None:
        for request in self._requests.values():
            if (
                request.skill == skill
                and request.arguments_digest == digest
                and request.status == PENDING
            ):
                self._expire_if_due(request)
                if request.status == PENDING:
                    return request
        return None

    def _find_approved(self, skill: str, digest: str) -> ApprovalRequest | None:
        for request in self._requests.values():
            if (
                request.skill == skill
                and request.arguments_digest == digest
                and request.status == APPROVED
            ):
                return request
        return None

    def _expire_if_due(self, request: ApprovalRequest) -> None:
        if request.status != PENDING:
            return
        if self._clock() < request.expires_at:
            return
        request.status = EXPIRED
        request.decided_at = self._clock()
        self._emit("expired", request, note="超时未审批，按拒绝处理")

    def _emit(self, action: str, request: ApprovalRequest, *, by: str = "", note: str = "") -> None:
        entry = ApprovalAuditEntry(
            at=self._clock(),
            action=action,
            request_id=request.id,
            skill=request.skill,
            arguments_digest=request.arguments_digest,
            by=by,
            note=note,
        )
        self._log.append(entry)
        if self._audit_sink is not None:
            try:
                self._audit_sink(entry)
            except Exception:  # noqa: BLE001 - 审计通道故障不该让请求失败
                logger.warning("审批审计写入失败：%s", action, exc_info=True)


def skill_of(skill: Any) -> str:
    """从 `SkillSpec` / 字符串 / 任何带 `name` 的对象取名字。

    宽容是刻意的：这个模块要被 W8 的 `PermissionGuard`（传 SkillSpec）、
    W11 的 `runtime`（传名字）和测试（传字符串）三处调用，
    要求它们都构造一个 SkillSpec 只是为了取名字，没有意义。
    """

    if isinstance(skill, str):
        return skill
    name = getattr(skill, "name", None)
    if isinstance(name, str) and name:
        return name
    raise ApprovalError(f"无法从 {skill!r} 取出技能名")


def _is_dangerous(skill: Any) -> bool:
    """判定是否需要人批。

    两个来源任一成立即算：显式 `dangerous=True`，或风险级是 `irreversible`
    （W8 的 `risk_of` 按权限推导，`publish` 权限自动落到最高级）。
    字符串形式的技能名没有元数据，**按不需要审批处理**——
    它只可能来自测试或内部工具，真实执行路径一定拿得到 SkillSpec。
    """

    if isinstance(skill, str):
        return False
    if bool(getattr(skill, "dangerous", False)):
        return True
    try:
        from ..skills.guard import RISK_IRREVERSIBLE, risk_of

        return risk_of(skill) == RISK_IRREVERSIBLE
    except Exception:  # noqa: BLE001 - 取不到元数据就当不需要审批（见 docstring）
        return False

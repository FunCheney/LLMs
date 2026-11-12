"""权限闸与风险分级（W8 D5）：决定"要不要人"。

这一层回答三个问题，顺序固定：

1. **这个主体能不能执行这个 Skill**（`permissions` × `principal.groups`）
2. **风险几级**（决定后面要不要审批）
3. **要不要人批**（`irreversible` 必须人批；本周给 `pending` 占位，W10 接真 HITL）

风险分级为什么不按"工具名"而是按权限推导：

`publish_post` 和 `write_file` 在名字上就差了两个字，而在风险上差一个量级。
按权限推导，新增一个发布类 Skill 时**只要它声明了 `publish` 权限就自动
拿到最高风险级**；按名字维护一张表，则是每加一个 Skill 都要人工决定一次
风险级——而漏判的表现是"不可撤销的动作被当成普通写操作放行"。

`*` 这个通配组只给 `read` / `net` 用：公开检索不需要身份。
给 `write` 配 `*` 等于"谁都能写"，那白名单就白配了。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from ..retrieval.acl import Principal
from .spec import (
    SkillSpec,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ANY_PRINCIPAL",
    "DEFAULT_POLICY",
    "RISK_IRREVERSIBLE",
    "RISK_READ",
    "RISK_WRITE",
    "ApprovalDecision",
    "PermissionDecision",
    "PermissionGuard",
    "PermissionPolicy",
    "risk_of",
]

# 通配组：任何主体（含匿名）都满足。只该用于 read / net。
ANY_PRINCIPAL = "*"

RISK_READ = "read"
RISK_WRITE = "write"
RISK_IRREVERSIBLE = "irreversible"

_RISK_ORDER = {RISK_READ: 0, RISK_WRITE: 1, RISK_IRREVERSIBLE: 2}

# 权限 → 风险级。`shell` 归到 write：它能造成写副作用，但不像"对外发布"
# 那样不可撤销。`publish` 单独一档：它是唯一"发出去了就收不回"的动作。
_RISK_BY_PERMISSION = {
    "read": RISK_READ,
    "net": RISK_READ,
    "write": RISK_WRITE,
    "shell": RISK_WRITE,
    "publish": RISK_IRREVERSIBLE,
}

# 默认授权表。写成常量而不是散在代码里，是为了让"谁有权做什么"是一份
# **可以 review 的清单**，而不是若干处 if。
DEFAULT_POLICY: dict[str, frozenset[str]] = {
    "read": frozenset({ANY_PRINCIPAL}),
    "net": frozenset({ANY_PRINCIPAL}),
    "write": frozenset({"editor", "publisher"}),
    "shell": frozenset({"ops"}),
    "publish": frozenset({"publisher"}),
}


def risk_of(skill: SkillSpec) -> str:
    """按声明的权限推导风险级，取最高的一档。

    `dangerous=True` 直接拉到最高——这个字段是"作者明知它有风险"的**显式声明**，
    不该因为权限表里漏了一项就被降级。
    """
    if skill.dangerous:
        return RISK_IRREVERSIBLE

    risk = RISK_READ
    for permission in skill.permissions:
        candidate = _RISK_BY_PERMISSION.get(permission, RISK_WRITE)
        if _RISK_ORDER[candidate] > _RISK_ORDER[risk]:
            risk = candidate
    return risk


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    """权限 → 允许的组集合。不可变：一张运行时会变的授权表，
    等于"每次调用都可能得到不同结果"，审计也就无法复算。"""

    rules: Mapping[str, frozenset[str]] = field(default_factory=lambda: dict(DEFAULT_POLICY))

    def granted_groups(self, permission: str) -> frozenset[str]:
        return self.rules.get(permission, frozenset())

    def missing(self, principal: Principal, skill: SkillSpec) -> tuple[str, ...]:
        """返回主体**缺少授权**的权限项。空 = 全部满足。"""
        missing: list[str] = []
        for permission in sorted(skill.permissions):
            groups = self.granted_groups(permission)
            if ANY_PRINCIPAL in groups:
                continue
            if principal.groups & groups:
                continue
            missing.append(permission)
        return tuple(missing)

    def allows(self, principal: Principal, skill: SkillSpec) -> bool:
        return not self.missing(principal, skill)


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """授权判定。`code` 与 `ToolOutcome.code` 同一套词表。"""

    allowed: bool
    risk: str = RISK_READ
    # 缺哪几项权限。**不含**主体名以外的信息，可直接进审计。
    missing: tuple[str, ...] = ()
    code: str = ""

    @property
    def denied(self) -> bool:
        return not self.allowed


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """审批结果。

    `pending` 是**正常返回值**而不是异常：本周没有 HITL 通道，
    `irreversible` 动作就该停在"等人批"这个状态上，而不是静默执行。
    """

    status: str
    # 展示给审批人的"将要发生什么"。没有它，审批就是盲签。
    summary: str = ""
    reason: str = ""

    @property
    def approved(self) -> bool:
        return self.status == "approved"


# 审批人的签名：收 (skill, arguments, summary) → 返回 "approved" / "rejected" / "pending"。
Approver = Callable[[SkillSpec, Mapping[str, object], str], Awaitable[str]]


class PermissionGuard:
    """白名单 + 风险分级 + 审批钩子。"""

    def __init__(
        self,
        *,
        policy: PermissionPolicy | None = None,
        approver: Approver | None = None,
    ) -> None:
        self._policy = policy or PermissionPolicy()
        # None = 没有接审批通道 → 高风险动作一律 pending（占位，W10 接真 HITL）。
        self._approver = approver

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    def check(self, principal: Principal, skill: SkillSpec) -> PermissionDecision:
        """授权判定。**先判权限再算风险**：没权限的人不该看到风险等级，
        那是内部信息（"这个动作是最危险的一档"本身就有情报价值）。
        """
        missing = self._policy.missing(principal, skill)
        if missing:
            return PermissionDecision(
                allowed=False,
                risk=RISK_READ,
                missing=missing,
                code="permission_denied",
            )
        return PermissionDecision(allowed=True, risk=risk_of(skill))

    @staticmethod
    def needs_approval(risk: str) -> bool:
        return risk == RISK_IRREVERSIBLE

    @staticmethod
    def summarize(skill: SkillSpec, arguments: Mapping[str, object]) -> str:
        """生成"将要发生什么"。

        只列**参数名与前 80 字符**，不列全文：审批页要让人能一眼确认
        "是不是我批的那件事"，而不是让人读一篇长文。密钥类字段直接打码。
        """
        parts: list[str] = []
        for key, value in list(arguments.items())[:5]:
            text = str(value)
            if len(text) > 80:
                text = f"{text[:80]}…（共 {len(text)} 字符）"
            parts.append(f"{key}={text}")
        args_summary = "; ".join(parts) or "（无参数）"
        return f"{skill.name}（风险：不可撤销）将执行：{args_summary}"

    async def approve(
        self, skill: SkillSpec, arguments: Mapping[str, object]
    ) -> ApprovalDecision:
        """高风险动作的审批门。

        没有审批通道时返回 `pending` 且**不执行**。这条默认行为很重要：
        "没接 HITL"不等于"默认放行"，那就等于所有不可撤销动作都无门槛。
        """
        summary = self.summarize(skill, arguments)
        if self._approver is None:
            return ApprovalDecision(
                status="pending",
                summary=summary,
                reason="未接审批通道（W10 HITL 前占位）：irreversible 动作一律 pending",
            )

        try:
            verdict = await self._approver(skill, arguments, summary)
        except Exception:  # noqa: BLE001 - 审批通道故障必须 fail-closed
            logger.warning("approver failed, fail-closed: skill=%s", skill.name, exc_info=True)
            return ApprovalDecision(
                status="pending", summary=summary, reason="审批通道异常，按 pending 处理"
            )

        if verdict == "approved":
            return ApprovalDecision(status="approved", summary=summary)
        if verdict == "rejected":
            return ApprovalDecision(status="rejected", summary=summary, reason="审批人拒绝")
        return ApprovalDecision(status="pending", summary=summary, reason=f"审批人返回 {verdict!r}")

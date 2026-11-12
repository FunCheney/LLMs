"""最小模型可见面（W8 D4）：工具越多，选错越多。

把所有 Skill 一次性塞进 system prompt 会有三个后果：token 涨、模型选错、
越权面变大。第三个最隐蔽——**模型能看见的工具，就是它可能去调用的工具**，
而一次被诱导的调用比一次越权访问更容易发生（提示注入的目标就是"让它调"）。

两个裁剪维度：

| 维度 | 依据 | 例子 |
|---|---|---|
| 阶段 | `skill.stages` | 调研阶段不需要 `publish_post` |
| 主体（粗粒度） | `skill.permissions` ∩ 主体组 | 只读访客看不到 `publish_post` |

**为什么主体维度只做"粗粒度"裁剪。** 细粒度的授权判定在 `guard.py`，
两边都做全量判定会让 `permission_denied` 这个码永远不可能出现——
工具要么可见且可执行，要么根本不可见。而这两件事在日志里的含义完全不同：

- `tool_not_visible`：模型调了一个**它不该知道存在**的工具 → 提示注入、
  过期可见面、或者模型在编工具名。这是**安全事件**。
- `permission_denied`：模型调了一个它看得见的工具，但主体没被授权 →
  正常的权限生效，是**运营事件**（该去加组还是该去收权限）。

混成一个码，上述两类工单会进同一个队列。

粗粒度只裁 `publish` / `shell`：这两类"看得见但完全用不了"最伤体验
（模型会反复尝试）。`write` 这类留着可见——"能写文件但不能联网"
是有意义的中间态，该由 guard 在执行时说清楚。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..retrieval.acl import Principal
from .guard import ANY_PRINCIPAL, PermissionPolicy
from .spec import PERMISSION_PUBLISH, PERMISSION_SHELL, SkillSpec, TaskStage

__all__ = [
    "COARSE_PERMISSIONS",
    "HIDDEN_BY_STAGE",
    "HIDDEN_BY_PRINCIPAL",
    "VisibilityDecision",
    "VisibilityPolicy",
]

# 参与"主体粗筛"的权限：这两类被裁掉时，模型即使看见也 100% 会失败。
COARSE_PERMISSIONS = frozenset({PERMISSION_PUBLISH, PERMISSION_SHELL})

HIDDEN_BY_STAGE = "stage"
HIDDEN_BY_PRINCIPAL = "principal"


@dataclass(frozen=True, slots=True)
class VisibilityDecision:
    """一次可见面计算的结果。**必须可打印**——可见面进日志，才能回答
    "模型当时为什么没选那个工具"。"""

    stage: str
    # 模型这次能看见的 Skill（顺序按注册顺序，保证可复现）。
    visible: tuple[SkillSpec, ...] = ()
    # 被裁掉的：`(name, reason)`。reason 只有 `stage` / `principal` 两种，
    # 与 `permission_denied` 严格区分。
    hidden: tuple[tuple[str, str], ...] = ()
    # 全集大小，用来算收敛比例（Gate 8 #8 的判据）。
    total: int = 0

    @property
    def ratio(self) -> float:
        """可见比例。全集为 0 时返回 0.0（不是 1.0：没有工具不等于收敛达标）。"""
        return len(self.visible) / self.total if self.total else 0.0

    def names(self) -> tuple[str, ...]:
        return tuple(skill.name for skill in self.visible)

    def is_visible(self, name: str) -> bool:
        return any(skill.name == name for skill in self.visible)

    def reason(self, name: str) -> str:
        """这个工具为什么不可见。可见返 `visible`，根本不存在返 `unknown_skill`。"""
        if self.is_visible(name):
            return "visible"
        for hidden_name, reason in self.hidden:
            if hidden_name == name:
                return reason
        return "unknown_skill"

    def describe_all(self) -> str:
        """渐进式加载的注入文本：**只**有 `name + description`（约 20 token/个）。

        `instructions` 等模型选中之后再注入。实跑证据见
        `docs/w8/w8-acceptance.md` #8：6 个内置 Skill 全量注入 instructions 是
        2778 字符，只注入描述是 531 字符（**19.1%**，判据 < 20%）。
        """
        return "\n".join(skill.describe() for skill in self.visible)

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "visible": [skill.name for skill in self.visible],
            "hidden": [{"name": name, "reason": reason} for name, reason in self.hidden],
            "total": self.total,
            "ratio": round(self.ratio, 3),
        }


class VisibilityPolicy:
    """按 (principal, stage) 计算可见面。纯函数，无状态。"""

    def __init__(
        self,
        *,
        permissions: PermissionPolicy | None = None,
        coarse: frozenset[str] = COARSE_PERMISSIONS,
    ) -> None:
        # 复用 guard 的授权表：两处各存一份"谁能做什么"，
        # 迟早在某次改动后给出不一致的答案。
        self._permissions = permissions or PermissionPolicy()
        self._coarse = coarse

    def surface(
        self,
        skills: list[SkillSpec] | tuple[SkillSpec, ...],
        *,
        stage: TaskStage | str,
        principal: Principal | None = None,
    ) -> VisibilityDecision:
        resolved_stage = stage.value if isinstance(stage, TaskStage) else str(stage)
        visible: list[SkillSpec] = []
        hidden: list[tuple[str, str]] = []

        for skill in skills:
            stage_ok = not skill.stages or resolved_stage in skill.stages
            if not stage_ok:
                hidden.append((skill.name, HIDDEN_BY_STAGE))
                continue

            if principal is not None and not self._coarse_allows(principal, skill):
                hidden.append((skill.name, HIDDEN_BY_PRINCIPAL))
                continue

            visible.append(skill)

        return VisibilityDecision(
            stage=resolved_stage,
            visible=tuple(visible),
            hidden=tuple(hidden),
            total=len(skills),
        )

    # -- 内部 -------------------------------------------------------------- #

    def _coarse_allows(self, principal: Principal, skill: SkillSpec) -> bool:
        for permission in sorted(skill.permissions & self._coarse):
            groups = self._permissions.granted_groups(permission)
            if ANY_PRINCIPAL in groups:
                continue
            if principal.groups & groups:
                continue
            return False
        return True

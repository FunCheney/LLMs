"""完成标准校验（W8 D3）：第 6 层的执行者。

它只回答一个问题：**这次输出够不够格算"完成了"**。

三条纪律，都来自"不确定就暴露，不粉饰"这一条（和 W4 judge 解析失败给 None
不补分、W7 Drift 只报警不自动改是同一条）：

1. **判不了的标准不许写进 Skill。** `Check` 只接受四种机器可判的形态，
   写"质量要高"在 `spec.py` 就被拒掉——否则校验器会退化成"读一遍然后点头"。
2. **失败只重试一次。** 第二次仍不满足就如实返回 `passed=False` 加"缺什么"。
   无限重试会把一次明确的失败变成一次昂贵的、最终仍然失败的循环。
3. **没标准 ≠ 通过。** 没有 `done_criteria` 的 Skill 返回 `checked=0`，
   报告里能数出"多少次'完成'其实没被验证过"。把它当通过是一种沉默的谎。

`enforce()` 需要重生成时接受一个回调而不是自己调模型：校验器不该拥有
LLM 客户端——那会让它出现在网关的重试/计费/trace 链路之外，
成本与延迟都记不上账（W7 的教训：不记账的优化会被报表推荐）。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .spec import Check, SkillSpec

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CITE_PATTERN",
    "DEFAULT_ITEM_PATTERN",
    "SkillValidator",
    "Verification",
]

# 默认引用形态：`[1]` / `[12]`。与 W6 的引用编号一致。
DEFAULT_CITE_PATTERN = r"\[\d+\]"
# 默认条目形态：行首 `- ` / `* ` / `1. ` / `1)`。
DEFAULT_ITEM_PATTERN = r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S"


@dataclass(frozen=True, slots=True)
class Verification:
    """一次校验的结果。字段直接进报告与审计。"""

    # 是否承认"完成"。第二次仍不满足时为 False，且**不**给出通过。
    passed: bool
    # 未满足的标准（原文 hint，可直接回喂模型）。
    unmet: tuple[str, ...] = ()
    # 实际跑了多少条标准。0 = 这个 Skill 没有完成标准（"通过"其实是没验）。
    checked: int = 0
    # 调用了几次生成（1 或 2）。>1 说明第一次没过。
    attempts: int = 1
    # 最终用于判定的输出（重试后是新的那份）。
    output: str = ""
    # 供报告用的细节：每条标准是过了还是没过。
    details: tuple[str, ...] = field(default_factory=tuple)


class SkillValidator:
    """纯同步判定 + 一个异步 `enforce`（带一次重试）。"""

    def verify(self, skill: SkillSpec, output: str) -> tuple[list[str], list[str]]:
        """返回 `(未满足的 hint, 明细)`。明细形如 `contains:OK` / `cites:1<2`。"""
        unmet: list[str] = []
        details: list[str] = []

        for check in skill.done_criteria:
            ok, detail = self._check(check, output or "")
            details.append(f"{check.kind}:{'OK' if ok else detail}")
            if not ok:
                unmet.append(check.hint or f"未满足 {check.kind}")

        return unmet, details

    async def enforce(
        self,
        skill: SkillSpec,
        output: str,
        *,
        regenerate: Callable[[str], Awaitable[str]] | None = None,
    ) -> Verification:
        """判定；不满足且有 `regenerate` 时**重试一次**。

        `regenerate` 收一段"缺什么"的说明，返回新的输出。第二次判定的结果
        就是最终结果——即使仍然不满足，也不会再调第三次。
        """
        unmet, details = self.verify(skill, output)

        if not unmet:
            return Verification(
                passed=True,
                checked=len(skill.done_criteria),
                attempts=1,
                output=output,
                details=tuple(details),
            )

        if regenerate is None:
            # 不重试只报错：调用方没给重生成通道，硬编一个"再试一次"
            # 只会让我们自己骗自己。
            return Verification(
                passed=False,
                unmet=tuple(unmet),
                checked=len(skill.done_criteria),
                attempts=1,
                output=output,
                details=tuple(details),
            )

        logger.info(
            "skill verification failed, retrying once: skill=%s unmet=%s",
            skill.name,
            [item[:40] for item in unmet],
        )
        retried = await regenerate(_feedback(skill, unmet))
        unmet_retry, details_retry = self.verify(skill, retried)

        return Verification(
            passed=not unmet_retry,
            unmet=tuple(unmet_retry),
            checked=len(skill.done_criteria),
            attempts=2,
            output=retried,
            details=tuple(details_retry),
        )

    # -- 内部 -------------------------------------------------------------- #

    def _check(self, check: Check, output: str) -> tuple[bool, str]:
        if check.kind == "contains":
            return check.target in output, f"missing:{check.target[:20]}"

        if check.kind == "cites":
            pattern = check.target or DEFAULT_CITE_PATTERN
            found = len(re.findall(pattern, output))
            return found >= check.count, f"{found}<{check.count}"

        if check.kind == "min_items":
            pattern = check.target or DEFAULT_ITEM_PATTERN
            found = len(re.findall(pattern, output))
            return found >= check.count, f"{found}<{check.count}"

        # regex
        return bool(re.search(check.target, output)), f"no_match:{check.target[:20]}"


def _feedback(skill: SkillSpec, unmet: list[str]) -> str:
    """回喂给模型的说明。

    只给"缺什么"而不给"重写全文"，是因为重写会让它顺手改掉已经对的部分；
    而且这条说明本身要短——它进的是同一份 context。
    """
    lines = [f"上一次输出未通过 {skill.name} 的完成标准，请只补足以下缺口后重新输出："]
    lines.extend(f"- {item}" for item in unmet)
    return "\n".join(lines)

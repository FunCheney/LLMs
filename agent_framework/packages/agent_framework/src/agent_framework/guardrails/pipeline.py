"""四层护栏的**唯一入口**（W10 D4）：输入 → 审批 → 执行 → 输出 → 行为。

## 为什么要有这个文件

`input.py` / `output.py` / `behavior.py` / `hitl.py` 各自都能单测，但它们**不构成一道门**。
链路上每多一个调用点，就多一次"这个新接口忘了过某层"的机会——而漏掉的那层
不会报错，它只是**不存在**。这正是 ADR-008 的结论：横切关注点必须有一个
所有路径都必经的归属点，否则它会在半年后变成"大部分路径都过"。

所以这里不是"把四个类包一层"（那样的包装不增加任何约束），而是：
**调用方只能通过 `run()` 拿结果**，`run()` 内部固定顺序，顺序不可配。

## 顺序为什么是输入 → 审批 → 执行 → 输出 → 行为

- **审批在执行之前**，而不是"执行前检查一下能不能执行"。两者看着一样，
  差别在于未获批时**没有副作用发生过**——这不是代码风格，是红线。
- **输出在行为之前**：输出层决定"这次结果能不能交付"，行为层只记频率。
  把行为层放前面会出现"频率超限但内容合法"被拦断的情况，
  而 `behavior.py` 明确写了频率异常不该拦断当前调用。
- **行为层永不拦断**，只回传告警。它需要"执行之后"才有事件可记，
  所以是最后一步，且 `alerts` 是返回值的一部分而不是异常。

## 失败时给什么

不新造异常类型，沿用三处既有异常（各自带 `code`，与统一错误表同一张）：

| 阶段 | 异常 | code |
|---|---|---|
| 输入 | `BlockedInput` | `CONTENT_BLOCKED` |
| 审批 | `ApprovalRequired` | `APPROVAL_REQUIRED` |
| 输出 | `OutputRejected` | `CONTENT_BLOCKED` |

**每种异常都带上已经走过的 steps**（`err.guard_steps`），否则日志里只看到
"输出了违规内容"，看不到"它在输入层已经命中过一次可疑"——后者才是排查入口。

## `render` 为什么是必填的（当你要输出检查时）

`run()` 需要把执行结果变成文本才能做输出检查。如果结果是 `ArticleDraft`、
`ToolResult` 这类结构体，就得有人告诉它"哪一部分是要发布出去的正文"。

让它**默认取 `str(result)`** 是错的：`str(ArticleDraft(...))` 会把
`citations=`、`trace=` 一起拼进去，禁用词检查变成在检查数据类的 repr。
所以没传 `render` 又传了 `policy` → 直接抛 `ValueError`，
让"忘了说清楚检查什么"在第一次调用就暴露，而不是安静地检查了一堆无关文本。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from .behavior import (
    EVENT_APPROVAL_PENDING,
    EVENT_BUDGET_SPENT,
    EVENT_DANGEROUS_CALL,
    EVENT_STEP,
    BehaviorAlert,
    BehaviorMonitor,
)
from .hitl import ApprovalGate, ApprovalRequest, ApprovalRequired
from .input import BlockedInput, InputCheck, InputGuard
from .output import OutputCheck, OutputGuard, OutputPolicy, OutputRejected

__all__ = [
    "GuardedRun",
    "GuardrailPipeline",
    "GuardStep",
    "STAGE_APPROVAL",
    "STAGE_BEHAVIOR",
    "STAGE_EXECUTE",
    "STAGE_INPUT",
    "STAGE_OUTPUT",
]

STAGE_INPUT = "input"
STAGE_APPROVAL = "approval"
STAGE_EXECUTE = "execute"
STAGE_OUTPUT = "output"
STAGE_BEHAVIOR = "behavior"

SKIPPED = "skipped: "


@dataclass(frozen=True)
class GuardStep:
    """一层护栏的结论。`ok=False` 表示**这一层拦断了**，不是"这层有问题"。"""

    stage: str
    ok: bool
    code: str = ""
    note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def skipped(self) -> bool:
        return self.note.startswith(SKIPPED)


@dataclass(frozen=True)
class GuardedRun[T]:
    """一次被四层护栏夹住执行的完整记录。"""

    value: T
    text: str
    steps: tuple[GuardStep, ...]
    output: OutputCheck
    alerts: tuple[BehaviorAlert, ...] = ()
    approval: ApprovalRequest | None = None
    input: InputCheck | None = None

    @property
    def skipped(self) -> tuple[str, ...]:
        """这次**没有真正评估过**的层。

        报告里要能说出"哪几层这次没生效"，否则"四层护栏"听起来比实际更厚。
        注意区分两种"没拦断"：

        - `skipped` = 这层没做判断（没传 `policy` / 没提供动作）；
        - `note="not required"` = 这层判断了，结论是"不需要拦"。

        两者都 `ok=True`，但只有前者意味着"这块没设防"。
        """

        return tuple(step.stage for step in self.steps if step.skipped)


class GuardrailPipeline:
    """唯一入口。四层护栏由构造注入，顺序硬编码。"""

    def __init__(
        self,
        *,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
        behavior: BehaviorMonitor | None = None,
        approval: ApprovalGate | None = None,
        audit: Callable[[GuardStep], None] | None = None,
    ) -> None:
        self._input = input_guard or InputGuard()
        self._output = output_guard or OutputGuard()
        self._behavior = behavior or BehaviorMonitor()
        self._approval = approval
        # `audit` 是给 W2 `RequestLog` 的挂点：每层结论都要落进 trace，
        # 包括拦断的那一层。只在成功时记 audit 的写法，正好漏掉最该看的那种。
        self._audit = audit

    @property
    def behavior(self) -> BehaviorMonitor:
        return self._behavior

    async def run[T](
        self,
        text: str,
        *,
        execute: Callable[[str], Awaitable[T]],
        source: str = "user",
        policy: OutputPolicy | None = None,
        render: Callable[[T], str] | None = None,
        known_citations: set[int] | None = None,
        approval_for: tuple[Any, Mapping[str, object]] | None = None,
        scope: str = "",
        cost: float = 0.0,
    ) -> GuardedRun[T]:
        """跑一遍。`execute` 拿到的是**已隔离**的输入，不是原文本。

        让"用隔离后的形态"成为唯一顺手的选择：想用原文本得自己闭包捕获，
        那个动作足够显眼。
        """

        if policy is not None and render is None:
            raise ValueError(
                "传了 policy 就必须传 render：需要知道结果的哪一部分是要检查的正文"
            )

        steps: list[GuardStep] = []
        alerts: list[BehaviorAlert] = []

        # ---- 1. 输入 ------------------------------------------------------ #
        check = self._input.check(text, source=source)
        if not check.ok:
            self._note(
                steps,
                GuardStep(
                    stage=STAGE_INPUT,
                    ok=False,
                    code=BlockedInput.code,
                    note="；".join(check.violations),
                    detail=check.detail,
                ),
            )
            raise self._fail(
                BlockedInput("；".join(check.violations), violations=check.violations), steps
            )
        self._note(
            steps,
            GuardStep(
                stage=STAGE_INPUT,
                ok=True,
                detail={**check.detail, "hits": [hit.signature for hit in check.hits]},
            ),
        )

        # ---- 2. 审批（只有 dangerous 动作才会产生记录） -------------------- #
        request = await self._approve(approval_for, step_into=steps, alerts=alerts, scope=scope)

        # ---- 3. 执行 ------------------------------------------------------ #
        try:
            value = await execute(check.sanitized)
        except Exception as exc:
            self._note(
                steps, GuardStep(stage=STAGE_EXECUTE, ok=False, note=type(exc).__name__)
            )
            raise self._fail(exc, steps) from None
        self._note(steps, GuardStep(stage=STAGE_EXECUTE, ok=True))

        # ---- 4. 输出 ------------------------------------------------------ #
        output = self._check_output(
            value, policy=policy, render=render, known=known_citations, step_into=steps
        )

        # ---- 5. 行为（只记不拦） ------------------------------------------ #
        alerts.extend(self._observe(EVENT_STEP, scope=scope))
        if cost > 0:
            alerts.extend(self._observe(EVENT_BUDGET_SPENT, scope=scope, value=cost))
        self._note(
            steps,
            GuardStep(
                stage=STAGE_BEHAVIOR,
                ok=True,
                note="alerts" if alerts else "",
                detail={"alerts": [alert.rule for alert in alerts]},
            ),
        )

        return GuardedRun(
            value=value,
            text=check.sanitized,
            steps=tuple(steps),
            output=output,
            alerts=tuple(alerts),
            approval=request,
            input=check,
        )

    # -- 各阶段（拆出来是为了让 `run` 只有一条主流程） ---------------------- #

    async def _approve(
        self,
        approval_for: tuple[Any, Mapping[str, object]] | None,
        *,
        step_into: list[GuardStep],
        alerts: list[BehaviorAlert],
        scope: str,
    ) -> ApprovalRequest | None:
        if approval_for is None:
            self._note(
                step_into,
                GuardStep(stage=STAGE_APPROVAL, ok=True, note=SKIPPED + "调用方未提供动作"),
            )
            return None

        skill, arguments = approval_for
        try:
            assert self._approval is not None, "传了 approval_for 但没注入 ApprovalGate"
            request = await self._approval.guard_execute(skill, arguments)
        except ApprovalRequired as exc:
            # 频率事件要在**拦断时**记，否则 burst 规则永远看不到
            # "有人反复尝试未审批执行"这种最该告警的形态。
            alerts.extend(self._observe(EVENT_DANGEROUS_CALL, scope=scope))
            alerts.extend(self._observe(EVENT_APPROVAL_PENDING, scope=scope))
            self._note(
                step_into,
                GuardStep(
                    stage=STAGE_APPROVAL,
                    ok=False,
                    code=getattr(exc, "code", ""),
                    note=str(exc),
                    detail={"request_id": exc.request.id, "summary": exc.request.summary},
                ),
            )
            raise self._fail(exc, step_into) from None

        if request is None:
            # 注意这里**不**写 `skipped`：审批层确实做了判断（结论是"不需要批"）。
            # 把它记成 skipped 会让报告说"这次没走审批"，而实际上走了、也下了结论。
            self._note(
                step_into,
                GuardStep(stage=STAGE_APPROVAL, ok=True, note="not required: 非 dangerous 动作"),
            )
            return None

        alerts.extend(self._observe(EVENT_DANGEROUS_CALL, scope=scope))
        self._note(
            step_into,
            GuardStep(
                stage=STAGE_APPROVAL,
                ok=True,
                note="approved",
                detail={"request_id": request.id, "approved_by": request.decided_by},
            ),
        )
        return request

    def _check_output[T](
        self,
        value: T,
        *,
        policy: OutputPolicy | None,
        render: Callable[[T], str] | None,
        known: set[int] | None,
        step_into: list[GuardStep],
    ) -> OutputCheck:
        if policy is None:
            self._note(
                step_into,
                GuardStep(stage=STAGE_OUTPUT, ok=True, note=SKIPPED + "未指定 policy"),
            )
            return OutputCheck()

        assert render is not None  # `run` 入口已校验
        output = self._output.check(render(value), policy=policy, known_citations=known)
        if not output.ok:
            self._note(
                step_into,
                GuardStep(
                    stage=STAGE_OUTPUT,
                    ok=False,
                    code=OutputRejected.code,
                    note="；".join(output.violations),
                    detail=output.detail,
                ),
            )
            raise self._fail(OutputRejected(output.violations), step_into)
        self._note(step_into, GuardStep(stage=STAGE_OUTPUT, ok=True, detail=output.detail))
        return output

    # -- 小工具 ------------------------------------------------------------- #

    def _observe(
        self, event: str, *, scope: str, value: float = 1.0
    ) -> tuple[BehaviorAlert, ...]:
        alert = self._behavior.record(event, value=value, scope=scope)
        return (alert,) if alert is not None else ()

    def _note(self, steps: list[GuardStep], step: GuardStep) -> None:
        steps.append(step)
        if self._audit is not None:
            self._audit(step)

    def _fail(self, exc: Exception, steps: list[GuardStep]) -> Exception:
        """把已走过的 steps 挂到异常上。调用方 `except` 之后能直接读。"""

        cast(Any, exc).guard_steps = tuple(steps)
        return exc

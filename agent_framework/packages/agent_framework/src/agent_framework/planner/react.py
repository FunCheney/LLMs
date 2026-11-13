"""朴素 ReAct：观察 → 思考 → 行动，循环到模型自己说停（W9 D1 基线）。

**这个文件存在的唯一理由是当对照组。** 没有它，D5 的"Plan-and-Execute 显著
高于朴素 ReAct"就是一句无法证伪的话——而"我们做了个更复杂的东西所以更好"
恰好是本周最想避免的那类结论。

刻意保持"朴素"：不隐藏任何缺陷（没有计划、没有全局视野、只看最近 N 条观察）。
如果把它写得很聪明（比如偷偷加一个中间 summarizer），对照组就不再是对照组，
D5 比出来的差异会被这块补丁吃掉。
"""

from __future__ import annotations

from typing import Any

from .base import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    PlannerBase,
    Reasoner,
    StepRunner,
    TaskResult,
)
from .controls import AutonomyBudget, StopReason

__all__ = ["ReActPlanner"]


class ReActPlanner(PlannerBase):
    """ReAct 循环。`observation_window` 就是它的"记忆"——只能看最近 N 条。

    窗口是这类实现最容易被忽略的变量：给 2 和给 8 是两个不同的系统，
    而大多数对比实验不写这个数。所以它被做成显式参数，并进 `as_dict()`。
    """

    name = "react"

    def __init__(
        self,
        *,
        reasoner: Reasoner,
        runner: StepRunner,
        budget: AutonomyBudget | None = None,
        clock: Any = None,
        observation_window: int = 3,
    ) -> None:
        kwargs: dict[str, Any] = {
            "reasoner": reasoner,
            "runner": runner,
            "budget": budget,
            "observation_window": observation_window,
        }
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(**kwargs)

    async def run(self, task: str) -> TaskResult:
        answers: list[str] = []

        while True:
            stop = self._check_continue()
            if stop is not None:
                return self._partial(task, stop, answers)

            self._trace.turns += 1
            decision = await self._ask_next(self._state(task=task))
            self._meter.charge_tokens(decision.tokens)
            self._trace.tokens += decision.tokens

            if decision.is_finish:
                if decision.text.strip():
                    answers.append(decision.text.strip())
                return self._result(
                    task=task,
                    status=STATUS_COMPLETED,
                    reason=StopReason.COMPLETED,
                    answer="\n".join(answers),
                )

            record = await self.runner.run(decision)
            self._record(record)

            if record.status == "ok":
                continue

            # ReAct 没有计划可重规划，失败只能"再试一次"。**这正是它的缺陷**：
            # 试到预算耗尽为止，而耗尽的原因是同一个错误被重复了 8 次。
            # 这里如实保留这个行为，不做额外补救——补救会让对照组失真。
            if self._check_continue() is not None:
                return self._partial(
                    task,
                    self._check_continue() or StopReason.MAX_STEPS,
                    answers,
                    detail=f"最后一步失败（{record.code or record.status}）且预算耗尽",
                )

    def _partial(
        self,
        task: str,
        reason: StopReason,
        answers: list[str],
        *,
        detail: str = "",
    ) -> TaskResult:
        return self._result(
            task=task,
            status=STATUS_PARTIAL,
            reason=reason,
            answer="\n".join(answers),
            detail=detail or f"预算中止：{reason.value}",
        )

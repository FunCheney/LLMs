"""Plan-and-Execute：先出计划，再逐步执行，失败才重规划（W9 D1 主范式）。

为什么是主范式：capstone 只有"深度调研"这一格真的需要自主性（搜什么取决于
搜到啥），而它的失败模式是"漏了某个子问题"或"同一个子问题搜了三遍"。
先出计划恰好能把"要做哪几件事"变成可检查、可去重、可断言的清单；
纯 ReAct 的下一步完全由上一句观察决定，没人能回答"这个任务还差什么"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .base import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    Plan,
    PlannerBase,
    PlanStep,
    Reasoner,
    StepRunner,
    StepStatus,
    TaskResult,
)
from .controls import AutonomyBudget, StopReason

__all__ = ["PlanExecutePlanner"]


class PlanExecutePlanner(PlannerBase):
    """计划 → 逐步执行 → 失败重规划 → 成文。"""

    name = "plan_execute"

    def __init__(
        self,
        *,
        reasoner: Reasoner,
        runner: StepRunner,
        budget: AutonomyBudget | None = None,
        clock: Any = None,
        observation_window: int = 8,
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
        plan = await self._reasoner.propose_plan(task, max_steps=self._budget.max_steps)
        self._trace.plan_fingerprints.append(plan.fingerprint)
        # 见过的计划指纹。同一份计划被重提 ≥2 次 = 在原地打转，
        # 这时继续跑只会重复同样的失败，区别只是多花几倍钱。
        seen_plans = {plan.fingerprint: 1}

        answers: list[str] = []

        while True:
            stop = self._check_continue()
            if stop is not None:
                return self._partial(task, stop, plan, answers, steps_total=len(plan.steps))

            if plan.is_finished():
                break

            ready = plan.ready_steps()
            if not ready:
                # 还有 pending 但依赖永远不满足（依赖成环、或指向不存在的 id）。
                # 这不是"再等等就好"，是计划本身坏了——必须说出来，不能当成功。
                return self._result(
                    task=task,
                    status=STATUS_ABORTED,
                    reason=StopReason.ERROR,
                    answer="\n".join(answers),
                    detail=(
                        "计划卡死：没有可执行步骤，但仍有 "
                        f"{len([s for s in plan.steps if s.status is StepStatus.PENDING])} 步未完成"
                        "（检查依赖是否成环或指向不存在的步骤）"
                    ),
                    steps_total=len(plan.steps),
                )

            step = ready[0]
            plan = plan.with_status(step.id, StepStatus.RUNNING)

            self._trace.turns += 1
            decision = await self._ask_next(
                self._state(task=task, step=step, done_goals=_done_goals(plan))
            )
            self._meter.charge_tokens(decision.tokens)
            self._trace.tokens += decision.tokens

            if decision.is_finish:
                if decision.text.strip():
                    answers.append(decision.text.strip())
                plan = plan.with_status(step.id, StepStatus.DONE)
                continue

            record = await self.runner.run(decision, step_id=step.id)
            self._record(record)

            if record.status == "ok":
                plan = plan.with_status(step.id, StepStatus.DONE)
                continue

            # -- 失败：重规划 -------------------------------------------------- #
            plan = plan.with_status(step.id, StepStatus.FAILED)
            if self._meter.replans_exhausted():
                return self._partial(
                    task,
                    StopReason.MAX_REPLANS,
                    plan,
                    answers,
                    detail=(
                        f"步骤 {step.id} 失败（{record.code or record.status}），"
                        f"已重规划 {self._meter.replans} 次达上限"
                    ),
                )

            self._meter.count_replan()
            self._trace.replans += 1
            fresh = await self._reasoner.replan(plan, step)
            merged = _merge_replan(plan, fresh)
            self._trace.plan_fingerprints.append(merged.fingerprint)

            seen_plans[merged.fingerprint] = seen_plans.get(merged.fingerprint, 0) + 1
            # 阈值是 2：原计划算第一次提交，重规划又给出同一份就是第二次。
            # 阈值 3 看起来"更宽容"，实际会让第二次重复的那一轮**真的执行**，
            # 于是同一个失败被花两遍钱——而它的指纹早就告诉我们不会有新结果。
            if seen_plans[merged.fingerprint] >= 2:
                return self._partial(
                    task,
                    StopReason.PLAN_LOOP,
                    merged,
                    answers,
                    detail=(
                        f"重规划产出的计划与前次相同（指纹 {merged.fingerprint}），"
                        "判定为原地打转，中止"
                    ),
                )
            plan = merged

        # -- 成文 -------------------------------------------------------------- #
        composed = await self._compose(task=task, done_goals=_done_goals(plan))
        if composed.text.strip():
            answers.append(composed.text.strip())

        return self._result(
            task=task,
            status=STATUS_COMPLETED,
            reason=StopReason.COMPLETED,
            answer="\n".join(answers),
            steps_total=len(plan.steps),
        )

    def _partial(
        self,
        task: str,
        reason: StopReason,
        plan: Plan,
        answers: Sequence[str],
        *,
        detail: str = "",
        steps_total: int = 0,
    ) -> TaskResult:
        """超限中止。

        **必须交出已经拿到的部分结果**：调研的价值在于"拿到 4/5 的答案"，
        全有全无是最差的产品决策（W2 的 workflow 缝里已经写过同一条）。
        但同时 status 不能是 completed——否则"没做完"会被下游当成"做完了"。
        """

        return self._result(
            task=task,
            status=STATUS_PARTIAL,
            reason=reason,
            answer="\n".join(answers),
            detail=detail or f"预算中止：{reason.value}",
            steps_total=steps_total or len(plan.steps),
        )


def _done_goals(plan: Plan) -> tuple[str, ...]:
    return tuple(step.goal for step in plan.steps if step.status is StepStatus.DONE)


def _merge_replan(old: Plan, fresh: Plan) -> Plan:
    """把重规划结果里"已经做完的步骤"标回 done。

    这一步不做的话，重规划会把**已产生副作用的步骤重跑一遍**——W8 的幂等键
    能挡住完全相同的调用，但重规划往往会微调参数（换个 query），
    幂等键就变了，于是真的执行第二次。这是"重规划"最隐蔽的成本。

    匹配用 `id` 而不是 `goal`：id 是计划的身份，goal 是自然语言，模型重写一遍
    就会不一样。
    """

    done_ids = {
        step.id for step in old.steps if step.status in {StepStatus.DONE, StepStatus.SKIPPED}
    }
    steps: list[PlanStep] = []
    for step in fresh.steps:
        if step.id in done_ids:
            steps.append(replace(step, status=StepStatus.DONE))
        else:
            steps.append(step)
    return Plan(task=fresh.task, steps=steps, budget_tokens=fresh.budget_tokens)

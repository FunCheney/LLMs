"""supervisor：调度 + 三道守卫（W9 D3 的技术核心）。

三道守卫各自防一种**不同的**失败，缺一个都不算"有预算控制"：

| 守卫 | 防的是 | 失效的样子 |
|---|---|---|
| 步数上限 | 角色数量膨胀 / 无限细化 | 跑到天荒地老，账单继续涨 |
| 预算累计 | 每跳重置预算 | 每跳都显示"余额充足"，总额却是配置值的 N 倍 |
| 环路指纹 | 两个角色互踢皮球 | Outcome 只是"略降"，`path_repeat_rate` 从 0.05 跳到 0.6 |

它们**必须真的中断**。只记一条 warning 然后继续跑，等于把"有预算控制"写成
文档里的一句话——所以每个守卫都有一个反向用例：构造触发条件，断言返回
`partial` 而不是完整结果。

**步数记的是"角色行动轮数"，不是工具调用次数。** 这一点很容易写错：如果按
工具调用计数，一个只会说"我没有新问题可查"、不产出任何动作的角色（researcher
子问题耗尽、validator 反复判"还差引用"）会让步数永远停在原地，守卫永不触发，
循环本身倒是转得很欢。工具调用次数单独记在 `records` 里，给 D5 的
`steps_used` 用，两者不是同一个数。
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..planner.base import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_PENDING,
    Decision,
    RunTrace,
    StepRecord,
    StepRunner,
    TaskResult,
)
from ..planner.controls import StopReason, matrix
from ..skills.executor import SkillExecutor
from .handoff import Handoff, HandoffLedger, LoopDetected
from .roles import (
    ROLE_RESEARCHER,
    ROLE_SUPERVISOR,
    ROLE_VALIDATOR,
    AgentRole,
    ResearcherRole,
    RoleResult,
    SupervisorRole,
    ValidatorRole,
)
from .shared_state import Finding, SharedState, StaleStateError

__all__ = ["Supervisor", "SupervisorConfig"]


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    """守卫参数。

    `max_steps` 单独一份而不是复用 `AutonomyBudget.max_steps`：多 Agent 的
    一步是"一个角色行动一轮"，比单 Agent 的一步重得多（可能包含 3 次工具调用）。
    用同一个数会让两边的"步"含义不一样，成本对不上。
    """

    max_steps: int = 12
    max_budget_tokens: int = 20_000
    max_seconds: float = 120.0
    # 环路判定阈值：同一条 handoff 出现几次算环。
    repeat_threshold: int = 2

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        if self.max_budget_tokens < 1:
            raise ValueError("max_budget_tokens 必须 >= 1")
        if self.repeat_threshold < 2:
            raise ValueError("repeat_threshold 至少是 2（1 意味着第一步就判环）")


@dataclass(slots=True)
class _Trace:
    """内部账本。`rounds` 与 `records` 分开，见模块 docstring。"""

    records: list[StepRecord] = field(default_factory=list)
    tokens: int = 0
    rounds: int = 0

    @property
    def tool_calls(self) -> int:
        return len(self.records)


class Supervisor:
    """多角色编排。**自己不做任何检索与写作**，只调度 + 记账 + 守卫。"""

    def __init__(
        self,
        *,
        executor: SkillExecutor,
        roles: Mapping[str, AgentRole] | None = None,
        config: SupervisorConfig | None = None,
        scheduler: SupervisorRole | None = None,
        principal: Any = None,
        stage: str = "research",
        clock: Any = time.monotonic,
        questions: Any = (),
    ) -> None:
        self._config = config or SupervisorConfig()
        self._config.validate()
        self._runner = StepRunner(executor, principal=principal, stage=stage)
        self._scheduler = scheduler or SupervisorRole()
        self._roles: dict[str, AgentRole] = dict(
            roles
            or {
                ROLE_RESEARCHER: ResearcherRole(questions=questions),
                ROLE_VALIDATOR: ValidatorRole(),
            }
        )
        # supervisor 角色本身不执行；把它登记进来只是为了让"角色集合"完整，
        # 且能挡住"有人给 supervisor 也配一个会检索的实现"这种越界。
        self._roles.setdefault(ROLE_SUPERVISOR, self._scheduler)
        self._clock = clock
        self._trace = _Trace()

    # -- 只读 -------------------------------------------------------------- #

    @property
    def trace(self) -> _Trace:
        return self._trace

    def visible_tools(self) -> tuple[str, ...]:
        return self._runner.visible_tools()

    # -- 主循环 ------------------------------------------------------------ #

    async def run(self, task: str) -> TaskResult:
        started = self._clock()
        state = SharedState(task=task)
        ledger = HandoffLedger(repeat_threshold=self._config.repeat_threshold)
        last_role = "start"
        # 上一位角色是否要求"还有活、别换人"。只在它刚跑完的那一轮有效。
        continue_role = ""
        validated = False
        stop: StopReason | None = None
        detail = ""

        while True:
            # -- 守卫 1：步数（角色轮次） -------------------------------------- #
            if self._trace.rounds >= self._config.max_steps:
                stop, detail = (
                    StopReason.MAX_STEPS,
                    f"角色轮数达上限 {self._config.max_steps}",
                )
                break
            # -- 守卫 2：预算累计（不每跳重置） -------------------------------- #
            if self._trace.tokens >= self._config.max_budget_tokens:
                stop = StopReason.MAX_TOKENS
                detail = (
                    f"token 达上限 {self._config.max_budget_tokens}"
                    f"（已用 {self._trace.tokens}，累计而非每跳重置）"
                )
                break
            if float(self._clock() - started) >= self._config.max_seconds:
                stop = StopReason.MAX_SECONDS
                detail = f"墙钟达上限 {self._config.max_seconds}s"
                break

            role_name = self._scheduler.pick(
                state,
                last_role=last_role,
                validated=validated,
                continue_role=continue_role,
            )
            if role_name is None:
                break

            role = self._roles.get(role_name)
            if role is None:
                stop = StopReason.ERROR
                detail = f"角色 {role_name!r} 未注册"
                break
            if role_name == ROLE_SUPERVISOR:
                stop = StopReason.ERROR
                detail = "supervisor 被调度成执行角色：它只该调度"
                break

            handoff = Handoff(
                from_role=last_role,
                to_role=role_name,
                task=ask_for(role_name, state, task),
                # 传 id 不传正文：正文会让 handoff 变成第二条上下文通道。
                artifacts=tuple(
                    f"f{len(state.findings) - i}" for i in range(min(3, len(state.findings)))
                ),
                # **递减传递**：读当前余额，不是重新赋值。
                budget_left_tokens=max(0, self._config.max_budget_tokens - self._trace.tokens),
                step_no=self._trace.rounds,
            )
            # -- 守卫 3：环路指纹（记录之后立刻判，别等下一轮） ------------------ #
            try:
                ledger.record(handoff)
            except LoopDetected as exc:
                stop, detail = StopReason.PLAN_LOOP, str(exc)
                break
            self._trace.rounds += 1

            # 提交前先记版本：`act` 是 await，期间可能有别的角色写了状态。
            snapshot = state.version
            result: RoleResult = await role.act(
                task, state, budget_left=handoff.budget_left_tokens
            )
            # 角色申报的成本（推理 + 检索）进总账。`StepRecord.tokens` 是
            # 工具**输出**的估算，那不是我们付的钱，只留在轨迹里给评测用。
            self._trace.tokens += result.tokens

            pending = await self._execute(result, role_name=role_name)
            conflict = self._commit(state, role_name, result, expected=snapshot)
            if conflict is not None:
                stop = StopReason.ERROR
                detail = conflict
                break

            if pending is not None:
                return self._finish(
                    state=state,
                    ledger=ledger,
                    status=STATUS_PENDING,
                    reason=StopReason.COMPLETED,
                    detail=pending,
                    started=started,
                )

            if role_name == ROLE_VALIDATOR and not result.needs_fix:
                validated = True

            produced = bool(result.actions or result.findings or result.asked)
            if not produced and role_name == last_role:
                # **同一个角色被连着派两次，两次都空手**：第三次必然还是空手。
                # 判据必须是"连着两次"而不是"这次空手"——validator 判
                # `needs_fix` 时同样不产出任何东西，而它下一棒该交给 researcher。
                stop = StopReason.NO_PROGRESS
                detail = (
                    f"{role_name} 连续两轮没有产出（动作/发现/子问题都没有）：{result.note}"
                )
                break
            last_role = role_name
            continue_role = role_name if result.can_continue else ""

        status = STATUS_COMPLETED if stop is None else STATUS_ABORTED
        if stop is not None and state.findings:
            # 有半成品就交出去：报告的 4/5 也是价值。
            status = STATUS_PARTIAL
        return self._finish(
            state=state,
            ledger=ledger,
            status=status,
            reason=stop or StopReason.COMPLETED,
            detail=detail,
            started=started,
        )

    # -- 内部 -------------------------------------------------------------- #

    async def _execute(self, result: RoleResult, *, role_name: str) -> str | None:
        """执行角色给出的动作。返回非 None = 需要人工，调用方提交后收尾。"""

        records, pending = await run_role_actions(
            self._runner, result, role_name=role_name
        )
        self._trace.records.extend(records)
        return pending

    @staticmethod
    def _commit(
        state: SharedState, role_name: str, result: RoleResult, *, expected: int
    ) -> str | None:
        """提交一批增量。版本冲突返回原因（不静默重试），成功返回 None。

        冲突**必须当成一次事件**：静默丢弃一方的产出比中止一次更贵，
        而自动重试会把"谁的工作被丢了"这个问题藏起来——丢工作的人不会收到通知。
        """

        try:
            state.commit(
                role_name,
                findings=result.findings,
                asked=result.asked,
                tokens=result.tokens,
                expected_version=expected,
            )
        except StaleStateError as exc:
            return str(exc)
        return None

    def _finish(
        self,
        *,
        state: SharedState,
        ledger: HandoffLedger,
        status: str,
        reason: StopReason,
        detail: str,
        started: float,
    ) -> TaskResult:
        return build_result(
            state=state,
            ledger=ledger,
            records=self._trace.records,
            tokens=self._trace.tokens,
            status=status,
            reason=reason.value,
            detail=detail,
            seconds=float(self._clock() - started),
        )


def ask_for(role_name: str, state: SharedState, task: str) -> str:
    """这一跳到底在请对方做什么。**两个 driver 共用**，否则指纹口径会分叉。

    **带上轮次**：researcher 每轮查不同子问题，是不同的一次请求；否则
    "第 2 次派研究员"会被当成"同一个请求重复"，正常流程被误判成环。
    validator 的差事则是恒定的（校验同一份报告），重复出现就是环。
    """

    if role_name == ROLE_RESEARCHER:
        return f"检索：{task}（第 {len(state.asked) + 1} 轮）"
    if role_name == ROLE_VALIDATOR:
        return f"校验引用：{task}"
    return task


async def run_role_actions(
    runner: StepRunner, result: RoleResult, *, role_name: str
) -> tuple[list[StepRecord], str | None]:
    """执行角色给出的动作。返回 `(新记录, 需人工时的说明)`。

    **抽成模块级函数是为了 D4 的对照成立**：LangGraph 版与自研版的差异只在
    "谁决定下一个跑谁、守卫放在哪"，执行与记账必须走同一份代码。否则对比
    会退化成"比较两份重复实现的细节差异"，而那正是最容易得出假结论的地方。
    """

    records: list[StepRecord] = []
    for action in result.actions:
        # 控制权矩阵里"是否发布"归人。到了这里就不能再往下走——
        # 让代码替人做发布决定，是这类系统最贵的一次越权。
        if runner.requires_human(action.tool):
            if not matrix().human_required("是否发布"):
                raise RuntimeError("控制权矩阵说'是否发布'不归人，但 runner 判了需要人批")
            return records, (
                f"动作 {action.tool} 需要人工审批，已挂起（矩阵：是否发布 → human）"
            )

        record, payload = await runner.run_structured(
            Decision(kind="tool", tool=action.tool, arguments=dict(action.arguments)),
            step_id=action.question,
        )
        records.append(record)

        if record.status == "ok":
            claim, citation = evidence_of(record.observation, payload)
            if claim:
                result.findings.append(
                    Finding(claim=claim, citation=citation, role=role_name)
                )
    return records, None


def build_result(
    *,
    state: SharedState,
    ledger: HandoffLedger,
    records: Sequence[StepRecord],
    tokens: int,
    status: str,
    reason: str,
    detail: str,
    seconds: float,
) -> TaskResult:
    """把一次多 Agent 运行的账目组装成 `TaskResult`。两个 driver 共用。"""

    trace = RunTrace(
        records=list(records),
        # 多 Agent 没有"计划"，这里放 handoff 指纹：语义对齐成
        # "同一条交接走过几次"，`path_repeat_rate` 据此算。
        plan_fingerprints=[item.key for item in ledger.handoffs],
        replans=0,
        tokens=tokens,
        seconds=seconds,
        turns=len(ledger.handoffs),
    )
    return TaskResult(
        task=state.task,
        status=status,
        stop_reason=reason,
        answer=render_report(state),
        detail=detail,
        trace=trace,
        steps_done=len([r for r in records if r.status == "ok"]),
        steps_total=len(ledger.handoffs),
    )


_URL_RE = re.compile(r"https?://[^\s\"'<>()（）\[\]【】,，。；;]+")


def evidence_of(observation: str, payload: object) -> tuple[str, str]:
    """从一次工具执行里取出"结论"与"出处"。

    **优先用结构化载荷**：检索结果的结构化形态是
    `{"query":…, "results":[{title,url,snippet}]}`，直接拿它当结论的话，
    报告里就是一段 JSON 包络——读者看到的是信封，不是信。`title: snippet`
    才是那条证据本身，`url` 是它的出处。

    载荷不合用时退回原文，并**只从原文里抠 URL**。两处都取不到就返回空
    引用：空引用会让这条发现被 validator 拦下来，比"随便填个工具名当出处"诚实。
    """

    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                title = str(first.get("title") or "").strip()
                snippet = str(first.get("snippet") or "").strip()
                url = str(first.get("url") or "").strip()
                claim = f"{title}：{snippet}" if title and snippet else (title or snippet)
                if claim:
                    return claim, url

    match = _URL_RE.search(observation)
    return observation, match.group(0).strip() if match else ""


def render_report(state: SharedState) -> str:
    cited = [item for item in state.findings if item.cited]
    uncited = state.uncited
    lines = [f"# {state.task}", "", "## 结论（带出处）", ""]
    lines += [f"- {item.claim} [{item.citation}]" for item in cited] or [
        "（没有拿到带出处的结论）"
    ]
    if uncited:
        lines += ["", "## 未采信（缺出处）", ""]
        lines += [f"- {item.claim}（来自 {item.role}，没有引用）" for item in uncited]
    return "\n".join(lines)

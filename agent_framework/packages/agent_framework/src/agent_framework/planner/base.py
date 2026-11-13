"""规划器的数据结构与协议（W9 D1）。

**这一层只编排，不执行。** W8 的 `SkillExecutor` 已经有五道门，规划器若自己
再写一遍"检查权限 → 调工具"，就会出现两套判定规则，而它们迟早在某次改动后
给出不一致的答案。所以这里的 `StepRunner` 是**唯一**的调用出口，它把
`ToolCall` 交给 executor，自己只负责把结果记成 `StepRecord`。

三条设计选择值得说明：

1. **`PlanStep` 是 frozen 的，状态变更走 `Plan.with_status()`。**
   状态机如果允许原地改 `step.status`，那么"某一步被改成 done 了两次"
   这种错误不会有任何拦截点；换成返回新对象，非法转移必须在
   `_ALLOWED` 里查一遍才能通过。

2. **`Reasoner` 是协议，不是"必须调 LLM"。**
   D5 要做 ReAct / Plan-and-Execute / 多 Agent 的对比，如果三者各自换一个模型，
   结论就变成"谁用了更好的模型"。所以三档对比必须共用同一个 `Reasoner`——
   离线档用脚本化的确定性实现，在线档用同一个 LLM 客户端。**变量只能有一个。**

3. **`RunTrace` 是评测的一等输入，不是调试日志。**
   Trajectory 四个指标（工具选择准确率、无效调用率、步数、路径重复率）全部
   从它算出来。把它做成随手写的 list，W9 的过程评测就只能去解析文本日志。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from ..llm.types import ToolCall
from ..retrieval.acl import Principal
from ..retrieval.context_assemble import estimate_tokens
from ..skills.executor import SkillExecutor, ToolExecution
from .controls import (
    OWNER_CODE,
    AutonomyBudget,
    BudgetMeter,
    ControlMatrix,
    ControlViolation,
    StopReason,
    matrix,
)

__all__ = [
    "DECISION_FINISH",
    "DECISION_TOOL",
    "OBSERVATION_CHARS",
    "STATUS_ABORTED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_PARTIAL",
    "STATUS_PENDING",
    "AgentState",
    "Decision",
    "Plan",
    "PlanStep",
    "Planner",
    "PlannerBase",
    "Reasoner",
    "RunTrace",
    "StepRecord",
    "StepRunner",
    "StepStateError",
    "StepStatus",
    "TaskResult",
    "args_digest",
    "plan_fingerprint",
]


# --------------------------------------------------------------------------- #
# 步骤状态机
# --------------------------------------------------------------------------- #


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


# 允许的转移。**穷举**而不是"除了 done 都能改"：白名单写出来之后，
# 新增状态必须在这里显式加一条，漏了会被 `with_status` 拒掉。
_ALLOWED: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset({StepStatus.RUNNING, StepStatus.SKIPPED}),
    StepStatus.RUNNING: frozenset({StepStatus.DONE, StepStatus.FAILED}),
    # failed 可以回到 running：重规划后同一步允许重试。但 done 是终态——
    # "完成过又重跑"会让幂等与成本统计同时失真。
    StepStatus.FAILED: frozenset({StepStatus.RUNNING, StepStatus.SKIPPED}),
    StepStatus.DONE: frozenset(),
    StepStatus.SKIPPED: frozenset(),
}


class StepStateError(ValueError):
    """非法状态转移。空异常在这里是有用的：它能指出"第几步从什么改到什么"。"""


@dataclass(frozen=True, slots=True)
class PlanStep:
    """一个计划步骤。**它描述意图，不描述执行结果**——结果在 `StepRecord` 里。"""

    id: str
    goal: str
    # 期望的工具名（internal name，如 `search__search_web`）。None = 纯推理步骤。
    tool: str | None = None
    # 建议参数。模型可以改，改了之后仍要过 A14 校验（W8）。
    arguments: Mapping[str, Any] = field(default_factory=dict)
    # 依赖的步骤 id：只有全部 done 才能进 running。
    depends_on: tuple[str, ...] = ()
    status: StepStatus = StepStatus.PENDING

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "tool": self.tool,
            "depends_on": list(self.depends_on),
            "status": self.status.value,
        }


@dataclass(slots=True)
class Plan:
    """一份计划。`fingerprint` 用于检测"同一份计划被反复重提"。"""

    task: str
    steps: list[PlanStep] = field(default_factory=list)
    budget_tokens: int = 0

    # -- 查询 -------------------------------------------------------------- #

    def get(self, step_id: str) -> PlanStep | None:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def ready_steps(self) -> list[PlanStep]:
        """依赖已满足且尚未执行的步骤。

        依赖指向**不存在**的 id 时视为不满足（而不是"忽略这条依赖"）：
        否则一个拼错的依赖会让步骤提前跑，而错误不会暴露。
        """

        by_id = {step.id: step for step in self.steps}
        ready: list[PlanStep] = []
        for step in self.steps:
            if step.status is not StepStatus.PENDING:
                continue
            satisfied = True
            for dep in step.depends_on:
                parent = by_id.get(dep)
                if parent is None or parent.status is not StepStatus.DONE:
                    satisfied = False
                    break
            if satisfied:
                ready.append(step)
        return ready

    def is_finished(self) -> bool:
        return all(
            step.status in {StepStatus.DONE, StepStatus.SKIPPED} for step in self.steps
        )

    def failed_steps(self) -> list[PlanStep]:
        return [step for step in self.steps if step.status is StepStatus.FAILED]

    @property
    def fingerprint(self) -> str:
        return plan_fingerprint(self.steps)

    # -- 变更 -------------------------------------------------------------- #

    def with_status(self, step_id: str, status: StepStatus) -> Plan:
        """返回**新的** Plan。非法转移抛 `StepStateError`。"""

        current = self.get(step_id)
        if current is None:
            raise StepStateError(f"计划里没有步骤 {step_id!r}")
        if status not in _ALLOWED[current.status]:
            raise StepStateError(
                f"步骤 {step_id} 不能从 {current.status.value} 变成 {status.value}"
                f"（允许：{sorted(item.value for item in _ALLOWED[current.status])}）"
            )
        steps = [
            replace(step, status=status) if step.id == step_id else step
            for step in self.steps
        ]
        return Plan(task=self.task, steps=steps, budget_tokens=self.budget_tokens)


def plan_fingerprint(steps: Sequence[PlanStep]) -> str:
    """计划指纹：只看 `(id, tool, goal)`，不看状态。

    状态会随着执行变化，如果算进去，同一份计划每跑一步就换一个指纹，
    "重复计划检测"就永远不命中——这恰好是它要防的那种失效。
    """

    raw = "\n".join(f"{step.id}\x00{step.tool or ''}\x00{step.goal}" for step in steps)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 决策与结果
# --------------------------------------------------------------------------- #


DECISION_TOOL = "tool"
DECISION_FINISH = "finish"


@dataclass(frozen=True, slots=True)
class Decision:
    """模型的一步决策。`tokens` 让规划器把推理成本也记进预算。

    不记推理成本是最常见的"预算"假象：只算工具调用、不算模型调用，
    于是"预算没超"而账单翻倍。W7 踩过同一个坑（摘要调用没记账）。
    """

    kind: str
    tool: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""
    tokens: int = 0

    @property
    def is_finish(self) -> bool:
        return self.kind == DECISION_FINISH


@dataclass(frozen=True, slots=True)
class StepRecord:
    """一次工具执行的过程记录。**Trajectory 指标的唯一数据源。**"""

    index: int
    step_id: str
    tool: str
    args_digest: str
    status: str  # ok / error / denied
    code: str = ""
    tokens: int = 0
    replayed: bool = False
    # 结果摘要（进上下文用）。不存全文：轨迹记录会被逐条喂给下一个 reasoner，
    # 全文会让"窗口"这个变量失去意义。
    observation: str = ""

    @property
    def is_invalid(self) -> bool:
        """无效调用：工具不存在 / 不可见 / 参数非法 / 没权限。

        与"工具执行报错"分开：前者是**选择或契约**的问题（归因到工具层/prompt），
        后者是**对方系统**的问题（归因到外部依赖）。混在一起就分不清该改哪边。
        """

        return self.status == "denied" or self.code in {
            "tool_not_visible",
            "tool_not_found",
            "invalid_arguments",
            "permission_denied",
            "sandbox_denied",
        }


@dataclass(slots=True)
class RunTrace:
    """一次任务的完整轨迹。评测与归因都读它。"""

    records: list[StepRecord] = field(default_factory=list)
    # 每一版计划的指纹（含重规划产出的新版本）。重复值 = 计划在打转。
    plan_fingerprints: list[str] = field(default_factory=list)
    replans: int = 0
    tokens: int = 0
    cost: float = 0.0
    seconds: float = 0.0
    # 无工具调用的纯推理轮次（reasoner 被调了几次）。
    turns: int = 0

    @property
    def step_count(self) -> int:
        return len(self.records)

    @property
    def distinct_paths(self) -> int:
        return len({(item.tool, item.args_digest) for item in self.records})

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.step_count,
            "turns": self.turns,
            "replans": self.replans,
            "tokens": self.tokens,
            "cost": round(self.cost, 6),
            "seconds": round(self.seconds, 3),
            "distinct_paths": self.distinct_paths,
            "plan_fingerprints": list(self.plan_fingerprints),
        }


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #


STATUS_COMPLETED = "completed"
# 预算/步数/环路中止，但已经拿到的部分结果是**真的**，所以要交出去。
STATUS_PARTIAL = "partial"
STATUS_ABORTED = "aborted"
STATUS_FAILED = "failed"
# 需要人工（发布类动作）。与 aborted 分开：前者在等人，后者已经结束——
# 值班队列里这两种的处置完全不同。
STATUS_PENDING = "pending"


@dataclass(slots=True)
class TaskResult:
    """规划器统一返回值。

    `partial` 与 `aborted` 的区别是对**产品**有意义的：前者"拿到了大半，可以给用户"，
    后者"什么也没拿到"。用一个 `success: bool` 表达不了这个差别，而产品文案
    完全依赖它。
    """

    task: str
    status: str
    stop_reason: str = StopReason.COMPLETED.value
    answer: str = ""
    detail: str = ""
    trace: RunTrace = field(default_factory=RunTrace)
    steps_done: int = 0
    steps_total: int = 0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETED

    @property
    def usable(self) -> bool:
        """有没有可以交给用户的答案（completed 与 partial 都算）。"""

        return self.status in {STATUS_COMPLETED, STATUS_PARTIAL} and bool(self.answer.strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps_done": self.steps_done,
            "steps_total": self.steps_total,
            "detail": self.detail,
            "trace": self.trace.as_dict(),
        }


# --------------------------------------------------------------------------- #
# Reasoner：唯一可替换的"大脑"
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AgentState:
    """喂给 reasoner 的可见状态。

    `observations` 就是 ReAct 的窗口、P&E 的计划状态——**两者共用这个结构，
    差异只在窗口给多大**。这样 D5 的对比才不是"两个不同的东西在比"。

    `visible_tools` 由代码填（控制权矩阵里"工具是否可见"归 code）。模型只能
    在**看得见的集合里**选，这既省 token，也把"它怎么知道有这个工具"这件事
    变成了代码的责任。
    """

    task: str
    # 最近的观察（工具结果摘要），最新在末尾。长度由规划器控制。
    observations: tuple[str, ...] = ()
    # 已完成的步骤目标（P&E 用它保持全局视野，ReAct 为空）。
    done_goals: tuple[str, ...] = ()
    # 当前步骤（P&E 有，ReAct 为 None）。
    step: PlanStep | None = None
    steps_used: int = 0
    tokens_left: int = 0
    visible_tools: tuple[str, ...] = ()


class Reasoner(Protocol):
    """规划/决策/重规划/成文四段。**三种编排范式共用同一个实现**，否则对比无效。"""

    async def propose_plan(self, task: str, *, max_steps: int) -> Plan: ...

    async def replan(self, plan: Plan, failed: PlanStep) -> Plan: ...

    async def next_action(self, state: AgentState) -> Decision: ...

    async def compose(self, state: AgentState) -> Decision: ...


# --------------------------------------------------------------------------- #
# StepRunner：唯一执行出口
# --------------------------------------------------------------------------- #


class StepRunner:
    """把 `Decision` 变成一次真实的工具执行。**规划器不许绕过它。**

    它做的事只有三件：拼 `ToolCall`、调 `SkillExecutor`、把结果投影成
    `StepRecord` + 一段可进上下文的 observation。权限、可见面、A14、沙箱
    全在 executor 里（W8），这里**一条都不重复实现**。

    构造时就向控制权矩阵"报到"：越权判定与可见面必须归代码。这一步不是
    形式主义——如果哪天有人把矩阵里这两条改成 `model`，服务会在装配期
    直接抛 `ControlViolation`，而不是安静地把权限交给模型。
    """

    def __init__(
        self,
        executor: SkillExecutor,
        *,
        principal: Principal | None = None,
        stage: str = "research",
        controls: ControlMatrix | None = None,
    ) -> None:
        self._executor = executor
        self._principal = principal
        self._stage = stage
        self._controls = controls or matrix()
        self._controls.assert_code_owned("是否越过权限")
        self._controls.assert_code_owned("工具是否可见")
        self._seq = 0

    @property
    def executor(self) -> SkillExecutor:
        return self._executor

    def visible_tools(self) -> tuple[str, ...]:
        """当前可见面。规划器在**提议之前**就据此裁剪，而不是等执行被拒。

        "工具是否可见"归代码，所以这个视图是代码算出来的；模型只能在这个
        集合里选。它既省 token，也把"模型怎么会知道有这个工具"变成代码的责任。
        若矩阵把这条判给模型，这里直接拒绝——模型能"自己申请"看见某个工具，
        是越权的前置条件。
        """

        owner = self._controls.owner_of("工具是否可见")
        if owner != OWNER_CODE:
            raise ControlViolation(f"工具是否可见应由代码决定，当前矩阵是 {owner}")
        decision = self._executor.surface(principal=self._principal, stage=self._stage)
        return decision.names()

    def requires_human(self, tool: str) -> bool:
        """这一步要不要人批。

        矩阵里"是否发布"归 human，而"要发布"在数据上表现为该 Skill 声明了
        `publish` 权限。两者对上才算数：只按工具名硬编码 `publish_post`
        会在新增一个发布类 Skill 时静默漏掉审批。
        """

        if not self._controls.human_required("是否发布"):
            return False
        skill = self._executor.registry.get(tool)
        return skill is not None and "publish" in skill.permissions

    async def run(self, decision: Decision, *, step_id: str = "") -> StepRecord:
        record, _ = await self.run_structured(decision, step_id=step_id)
        return record

    async def run_structured(
        self, decision: Decision, *, step_id: str = ""
    ) -> tuple[StepRecord, dict[str, Any] | None]:
        """同 `run`，但把结构化载荷一并交出来。

        证据抽取（D2）需要搜索结果里的 `url/snippet`，而 `StepRecord` 只留
        一段给人看的 observation。把 payload 塞进 `StepRecord` 会让轨迹记录
        变成"顺带存了整份检索结果"，既占内存又让 Trace 的体积不可控；
        所以结构化出口单独给一个方法，需要的人自己接。
        """

        # 查一次矩阵，是为了让"模型拥有这两项"在代码里有落脚点，
        # 而不是只写在文档的表格里。
        tool_owned_by_model = self._controls.model_may_decide("调用哪个工具")
        args_owned_by_model = self._controls.model_may_decide("工具参数")
        if not tool_owned_by_model or not args_owned_by_model:
            raise ControlViolation(
                "工具选择/参数应由模型决定；若归代码，请走确定性 Workflow 而不是规划器"
            )

        self._seq += 1
        call = ToolCall(
            id=f"plan-{self._seq}",
            name=decision.tool,
            arguments=dict(decision.arguments),
        )
        execution: ToolExecution = await self._executor.execute(
            call, principal=self._principal, stage=self._stage
        )
        record = StepRecord(
            index=self._seq,
            step_id=step_id,
            tool=decision.tool,
            args_digest=args_digest(call.arguments),
            status="ok" if execution.ok else ("denied" if execution.code in _DENIED else "error"),
            code="" if execution.ok else execution.code,
            tokens=estimate_tokens(_text_of(execution)) + decision.tokens,
            replayed=bool(execution.outcome and execution.outcome.replayed),
            observation=_observation_of(execution),
        )
        payload = execution.outcome.payload if execution.outcome else None
        return record, payload


# 这几类码说明"我们自己的门挡住了"，不是对方系统坏了。
_DENIED = frozenset(
    {
        "tool_not_visible",
        "tool_not_found",
        "invalid_arguments",
        "permission_denied",
        "sandbox_denied",
    }
)

# 观察截断：喂回给 reasoner 的工具结果必须短。理由与 W7 的 tool_result 截断相同，
# 但这里更狠——轨迹记录会**逐条**进入下一轮上下文，一条 2k 字的结果乘以 6 步
# 就把窗口吃满，"窗口大小"这个实验变量也随之失效。
OBSERVATION_CHARS = 400


def _text_of(execution: ToolExecution) -> str:
    if execution.message is not None and isinstance(execution.message.content, str):
        return execution.message.content
    return execution.detail or ""


def _observation_of(execution: ToolExecution) -> str:
    if not execution.ok:
        return f"[{execution.code}] {execution.detail}"[:OBSERVATION_CHARS]
    text = _text_of(execution)
    if len(text) <= OBSERVATION_CHARS:
        return text
    # 截断要留标记：让 reasoner 知道"这里被截了"，否则它会以为结果就这么短，
    # 然后基于残缺信息下结论。
    return text[:OBSERVATION_CHARS] + f"…[截断 {len(text) - OBSERVATION_CHARS} 字]"


def args_digest(arguments: Mapping[str, Any]) -> str:
    """参数摘要。与 W8 审计同一口径：排序后哈希，只留 12 位。

    用它算 `path_repeat_rate`：同工具同参数重复出现 = 规划器在打转。
    存原文会让轨迹记录体积失控，且参数里可能有用户数据。
    """

    import json

    raw = json.dumps(dict(arguments), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #


class Stopwatch:
    """墙钟计时。注入 `clock` 是为了让 `max_seconds` 能在测试里被确定性触发。"""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self._start = clock()

    def elapsed(self) -> float:
        return float(self._clock() - self._start)


def meter_for(budget: AutonomyBudget, *, clock: Any = time.monotonic) -> BudgetMeter:
    """便捷构造。放在 base 里，避免各规划器各写一份时钟注入逻辑。"""

    return BudgetMeter(budget, clock=clock)


# --------------------------------------------------------------------------- #
# Planner 协议与共享循环
# --------------------------------------------------------------------------- #


class Planner(Protocol):
    """规划器协议。

    刻意**没有**照抄"`plan` / `replan` / `execute` 三个方法各自暴露"的形态：
    那样调用方就得自己写循环，而循环里恰好是预算、步数、环路检测、部分结果
    这四件最容易写错的事——每多一个调用方，就多一份走样的守卫。
    所以这里只暴露 `run(task)`，循环归规划器所有。
    """

    @property
    def name(self) -> str: ...

    async def run(self, task: str) -> TaskResult: ...


class PlannerBase:
    """ReAct / Plan-and-Execute / 多 Agent 共用的循环零件。

    **共用**是刻意的：D5 要比较"编排范式"，如果三种范式各自实现一套预算判定，
    比出来的差异里就混进了"谁的预算代码写得松"。共用一个 `PlannerBase`，
    变量才只剩编排本身。
    """

    name = "base"

    def __init__(
        self,
        *,
        reasoner: Reasoner,
        runner: StepRunner | None = None,
        budget: AutonomyBudget | None = None,
        clock: Any = time.monotonic,
        observation_window: int = 8,
    ) -> None:
        self._reasoner = reasoner
        # 可为 None：D2 的 `ResearchPlanner` 把执行交给检索后端（后端自己持有
        # runner），规划器本身不直接调工具。硬塞一个用不上的 runner 会让
        # "它到底会不会执行"这件事变得含糊。
        self._runner = runner
        self._budget = budget or AutonomyBudget()
        self._meter = BudgetMeter(self._budget, clock=clock)
        self._trace = RunTrace()
        self._clock = clock
        self._started = clock()
        # 观察窗口：ReAct 靠它"记住"前面发生了什么。它是最容易被忽略的变量——
        # 窗口给 2 和给 8，ReAct 的表现是两个东西，而论文里通常不写这个数。
        self._window = max(1, observation_window)
        self._controls = matrix()

    # -- 只读 -------------------------------------------------------------- #

    @property
    def trace(self) -> RunTrace:
        return self._trace

    @property
    def meter(self) -> BudgetMeter:
        return self._meter

    @property
    def reasoner(self) -> Reasoner:
        return self._reasoner

    @property
    def runner(self) -> StepRunner:
        if self._runner is None:
            raise RuntimeError(f"{self.name} 未配置 StepRunner，无法执行工具")
        return self._runner

    @property
    def observation_window(self) -> int:
        return self._window

    # -- 循环零件 ---------------------------------------------------------- #

    def _observe(self) -> tuple[str, ...]:
        """最近 N 条观察。**超窗丢弃而不是摘要**——摘要会引入一次额外调用，
        把"窗口大小"这个自变量搅浑。"""

        records = self._trace.records[-self._window :]
        return tuple(f"[{item.tool}] {item.observation}" for item in records)

    def _state(self, *, task: str, step: PlanStep | None = None,
               done_goals: Sequence[str] = ()) -> AgentState:
        return AgentState(
            task=task,
            observations=self._observe(),
            done_goals=tuple(done_goals),
            step=step,
            steps_used=self._meter.steps,
            tokens_left=self._meter.tokens_left,
            visible_tools=tuple(sorted(self._visible())),
        )

    def _record(self, record: StepRecord) -> None:
        self._trace.records.append(record)
        self._trace.tokens += record.tokens
        self._meter.count_step()
        self._meter.charge_tokens(record.tokens)

    def _result(
        self,
        *,
        task: str,
        status: str,
        reason: StopReason,
        answer: str = "",
        detail: str = "",
        steps_total: int = 0,
    ) -> TaskResult:
        self._trace.seconds = float(self._clock() - self._started)
        return TaskResult(
            task=task,
            status=status,
            stop_reason=reason.value,
            answer=answer,
            detail=detail,
            trace=self._trace,
            steps_done=len([r for r in self._trace.records if r.status == "ok"]),
            steps_total=steps_total or len(self._trace.records),
        )

    async def _ask_next(self, state: AgentState) -> Decision:
        """问模型下一步。**先确认"下一步"这条真的归模型。**

        如果哪天有人把矩阵里这条改给代码，那就说明这条链路该写成固定 Workflow，
        而不是继续用规划器——在调用点炸掉比让规划器假装自主更诚实。
        """

        if not self._controls.model_may_decide("下一步做什么"):
            raise ControlViolation(
                "下一步做什么应由模型决定；若归代码，这就是 Workflow 不是 Planner"
            )
        return await self._reasoner.next_action(state)

    async def _compose(self, *, task: str, done_goals: Sequence[str]) -> Decision:
        self._trace.turns += 1
        decision = await self._reasoner.compose(
            self._state(task=task, done_goals=done_goals)
        )
        self._meter.charge_tokens(decision.tokens)
        self._trace.tokens += decision.tokens
        return decision

    def _visible(self) -> frozenset[str]:
        return frozenset(self.runner.visible_tools())

    def _check_continue(self) -> StopReason | None:
        """循环的"能不能继续"只有一个入口。

        它先向控制权矩阵确认这条归代码，再问计量器。多这一步的意义：
        模型在本次决策里说"我还能继续"不作数——`Decision` 里没有、也不该有
        这个字段。守卫若能被模型的措辞绕过，它就不是守卫。
        """

        self._controls.assert_code_owned("是否能继续")
        return self._meter.exhausted()


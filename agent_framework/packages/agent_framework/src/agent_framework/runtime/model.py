"""统一执行模型（W11 D3）：`Run` / `Step` / `Attempt` / `Event` + 一张 9 态状态机。

## 为什么要有这个文件

W11 之前系统里有**四套互相对不上的状态**：

| 来源 | 状态 |
|---|---|
| W2 `/v1/tasks` | `queued / running / succeeded / failed` |
| W9 `PlanStep` | `pending / running / done / failed / skipped` |
| W10 审批 | `pending / approved / rejected / expired` |
| 时间线 | 只有日志行，没有对象 |

四套状态机都不可能覆盖对方，于是"任务显示 running，但某一步其实卡在等审批"
这类问题只能靠人肉翻日志。W11 做的不是"再加一套"，而是**留一套 + 四张映射表**
（`LEGACY_MAPPINGS`，见文件末尾），让旧枚举都能翻译进来说同一句话。

## 一个枚举、两张转移表

`Run` 与 `Step` 共用 `RunState` 九个值，但**允许的边不同**：
`Step` 永远不会进入 `PLANNING`（计划是 run 级的产物），`Run` 也不会两次 `COMPLETED`。

共用枚举的理由是"同一件事用同一个词"——如果 step 用 `done`、run 用 `completed`，
报告里就会出现两种写法指同一个终态，而查报告的人会以为是两件事。

## 三件必须成立的事

1. **非法迁移抛错，不是静默忽略**（`IllegalTransition`）。静默忽略会让"状态对不上"
   变成最难查的一类 bug：代码在跑，状态在漂，日志里全是成功。
2. **`Event` 只追加**。`Run.replay()` 只读事件就能重建内存态，
   不依赖"某张表的当前值"——那张表才是最容易被人手改的东西。
3. **`run.state` 可以**从 `steps` 推导出来（`derive_state()`）。
   两边一旦分叉，`reconcile()` 会纠正并留一条 `reconciled` 事件。
   这条是"任务显示 running 但卡在等审批"的结构性解法：状态不是人写的，是算出来的。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..error_codes import is_caller_code, is_reliability_code

__all__ = [
    "IllegalTransition",
    "LEGACY_MAPPINGS",
    "stuck_runs",
    "state_breakdown",
    "Attempt",
    "Event",
    "Run",
    "RunState",
    "Step",
    "allowed_from_run",
    "allowed_from_step",
    "clone",
    "is_terminal",
    "state_from_legacy",
]


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #


class RunState(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    # 等人（W10 HITL）。**独立状态，不是 running 的子状态**：
    # 混进 running 会让"等人"被算进"卡住"，D1 的可用性/延迟图直接失真。
    WAITING_APPROVAL = "waiting_approval"
    TOOL_FAILED = "tool_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL = frozenset({RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED})

# 白名单，穷举写出来。新增状态必须在这里显式加边，漏了会被 `transition()` 拒掉。
_RUN_EDGES: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.PLANNING, RunState.CANCELLED}),
    RunState.PLANNING: frozenset(
        {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.RUNNING: frozenset(
        {
            RunState.WAITING_TOOL,
            RunState.WAITING_APPROVAL,
            RunState.TOOL_FAILED,
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.WAITING_TOOL: frozenset(
        {RunState.RUNNING, RunState.TOOL_FAILED, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.WAITING_APPROVAL: frozenset(
        {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
    ),
    # 工具失败可以回 running：换参数/换工具重试是同一步的第二次尝试。
    RunState.TOOL_FAILED: frozenset(
        {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.COMPLETED: frozenset(),
    # FAILED 可以回 RUNNING：**续跑**（D3 的断点续跑就是"失败的 run 接着跑"）。
    # 同一个 run_id、同一条事件流，历史不被改写。
    # 但 COMPLETED 不能复活：完成过又重跑会让幂等与成本统计同时失真。
    RunState.FAILED: frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.CANCELLED: frozenset(),
}

# Step 的表：没有 PLANNING（步骤不会被"规划"），FAILED 可以回 RUNNING（续跑/重试）。
_STEP_EDGES: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset(
        {RunState.RUNNING, RunState.WAITING_APPROVAL, RunState.CANCELLED}
    ),
    RunState.RUNNING: frozenset(
        {
            RunState.WAITING_TOOL,
            RunState.WAITING_APPROVAL,
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.WAITING_TOOL: frozenset(
        {RunState.RUNNING, RunState.TOOL_FAILED, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.WAITING_APPROVAL: frozenset(
        {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.TOOL_FAILED: frozenset(
        {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.COMPLETED: frozenset(),
    # 与 run 不同：失败的步骤在续跑时要能被重新执行，否则断点续跑无从谈起。
    RunState.FAILED: frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.CANCELLED: frozenset(),
    RunState.PLANNING: frozenset(),
}


class IllegalTransition(ValueError):
    """非法状态迁移。带上 `(from, to, 允许集合)`——空异常在这里没有用。"""


def allowed_from_run(state: RunState) -> frozenset[RunState]:
    return _RUN_EDGES[state]


def allowed_from_step(state: RunState) -> frozenset[RunState]:
    return _STEP_EDGES[state]


def _check(
    edges: Mapping[RunState, frozenset[RunState]],
    current: RunState,
    target: RunState,
    *,
    what: str,
) -> None:
    if target not in edges[current]:
        allowed = ", ".join(sorted(item.value for item in edges[current])) or "（终态）"
        raise IllegalTransition(
            f"{what} 不能从 {current.value} 变成 {target.value}（允许：{allowed}）"
        )


# --------------------------------------------------------------------------- #
# 时间
# --------------------------------------------------------------------------- #


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Attempt：一次真实的调用
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Attempt:
    """一步里的**一次**工具/模型调用。

    `error_code` 用的是 `agent_framework.error_codes` 里那份码表（W4 收敛的唯一一份）。
    这里**不新造码**：W4 刚把两份漂移的码表合成一份（10 个码里 6 个判定相反），
    再抄一份必然重演。
    """

    n: int
    started_at: datetime
    tool: str | None = None
    # 参数指纹。Timeline 靠它回答"这次重试是不是改了参数"——
    # 没有它，"为什么重试"只能靠人翻日志里的两份 JSON。
    args_digest: str | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    latency_ms: float | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.error_code is None

    @property
    def is_upstream_fault(self) -> bool:
        """厂商侧的锅（5xx/超时/限流/熔断）。**不算模型答错**。"""

        return is_reliability_code(self.error_code)

    @property
    def is_caller_fault(self) -> bool:
        """我方把请求写错了。既不算厂商的，也不算模型的。"""

        return is_caller_code(self.error_code)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "tool": self.tool,
            "args_digest": self.args_digest,
            "error_code": self.error_code,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Attempt:
        finished = payload.get("finished_at")
        return cls(
            n=int(payload["n"]),
            started_at=datetime.fromisoformat(str(payload["started_at"])),
            tool=payload.get("tool"),
            args_digest=payload.get("args_digest"),
            finished_at=datetime.fromisoformat(str(finished)) if finished else None,
            error_code=payload.get("error_code"),
            latency_ms=payload.get("latency_ms"),
            detail=str(payload.get("detail") or ""),
        )


# --------------------------------------------------------------------------- #
# Event：只追加
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Event:
    """过程记录。**只追加，从不改写。**

    `kind` 是自由字符串而不是枚举：把它做成枚举，每次加一种事件就是一次破坏性改动，
    而事件类型是会长出来的东西。可以自由生长的部分就让它自由生长，
    真正需要收敛的是**状态**（`RunState` 九态）。
    """

    seq: int
    at: datetime
    kind: str
    state: RunState | None = None
    step_id: str | None = None
    attempt: int | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at": self.at.isoformat(),
            "kind": self.kind,
            "state": self.state.value if self.state else None,
            "step_id": self.step_id,
            "attempt": self.attempt,
            "data": dict(self.data),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Event:
        raw_state = payload.get("state")
        return cls(
            seq=int(payload["seq"]),
            at=datetime.fromisoformat(str(payload["at"])),
            kind=str(payload["kind"]),
            state=RunState(raw_state) if raw_state else None,
            step_id=payload.get("step_id"),
            attempt=payload.get("attempt"),
            data=dict(payload.get("data") or {}),
        )

    @classmethod
    def from_json(cls, line: str) -> Event:
        return cls.from_dict(json.loads(line))


# --------------------------------------------------------------------------- #
# Step
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Step:
    """一个子目标。`attempts` 是这一步试过的每一次。"""

    step_id: str
    goal: str
    state: RunState = RunState.CREATED
    tool: str | None = None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.state is RunState.COMPLETED

    @property
    def finished(self) -> bool:
        return self.state in _TERMINAL

    def last_attempt(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "goal": self.goal,
            "state": self.state.value,
            "tool": self.tool,
            "attempts": [item.as_dict() for item in self.attempts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Step:
        return cls(
            step_id=str(payload["step_id"]),
            goal=str(payload["goal"]),
            state=RunState(payload["state"]),
            tool=payload.get("tool"),
            attempts=[Attempt.from_dict(item) for item in payload.get("attempts") or ()],
        )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

# 推导 run 状态时的优先级。**顺序是设计的一部分**：
# "有人在等审批"必须压过"有人在跑"——否则一个 8 步的任务永远显示 running，
# 那一步的审批被淹没在正常进度里，而它恰恰是唯一需要人动手的地方。
_DERIVE_ORDER: tuple[RunState, ...] = (
    RunState.WAITING_APPROVAL,
    RunState.WAITING_TOOL,
    RunState.TOOL_FAILED,
    RunState.RUNNING,
)


@dataclass(slots=True)
class Run:
    """一次任务执行。`events` 是唯一可信的历史，`state` 是它的投影。"""

    run_id: str
    state: RunState = RunState.CREATED
    steps: list[Step] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    goal: str = ""

    # -- 查询 -------------------------------------------------------------- #

    def step(self, step_id: str) -> Step:
        for item in self.steps:
            if item.step_id == step_id:
                return item
        raise KeyError(f"run {self.run_id} 里没有步骤 {step_id!r}")

    def find_step(self, step_id: str) -> Step | None:
        for item in self.steps:
            if item.step_id == step_id:
                return item
        return None

    @property
    def pending_steps(self) -> list[Step]:
        return [item for item in self.steps if not item.finished]

    def attempts(self) -> Iterator[Attempt]:
        for item in self.steps:
            yield from item.attempts

    def derive_state(self) -> RunState:
        """从 steps 推出 run 该处于什么状态。

        没有 steps 时保持 `CREATED`/`PLANNING`（计划还没落地，没什么可推的）。
        """

        if not self.steps:
            return self.state

        states = {item.state for item in self.steps}

        for candidate in _DERIVE_ORDER:
            if candidate in states:
                return candidate

        if states <= {RunState.COMPLETED}:
            return RunState.COMPLETED
        if states <= {RunState.COMPLETED, RunState.CANCELLED}:
            return RunState.CANCELLED if RunState.CANCELLED in states else RunState.COMPLETED
        if RunState.FAILED in states or RunState.TOOL_FAILED in states:
            return RunState.FAILED
        return self.state

    # -- 变更（唯一入口） --------------------------------------------------- #

    def advance(self, target: RunState, *, at: datetime | None = None, reason: str = "") -> Event:
        """run 级状态迁移。非法则抛 `IllegalTransition`。"""

        _check(_RUN_EDGES, self.state, target, what=f"run {self.run_id}")
        self.state = target
        return self._append("run_state", at=at, state=target, data={"reason": reason})

    def add_step(
        self, step_id: str, goal: str, *, tool: str | None = None, at: datetime | None = None
    ) -> Step:
        if self.find_step(step_id) is not None:
            raise ValueError(f"run {self.run_id} 已有步骤 {step_id!r}")
        step = Step(step_id=step_id, goal=goal, tool=tool)
        self.steps.append(step)
        self._append("step_added", at=at, step_id=step_id, data={"goal": goal, "tool": tool})
        return step

    def mark_step(
        self,
        step_id: str,
        target: RunState,
        *,
        at: datetime | None = None,
        error_code: str | None = None,
        reason: str = "",
    ) -> Event:
        """step 级状态迁移，并顺带把 run 状态对齐。

        **原子性**：先校验 step 的边，再校验"对齐后的 run 状态是否可达"，
        两个都通过才落盘。否则会出现"step 变了、run 没变"的半截状态——
        而半截状态正是这个模型想消灭的东西。
        """

        step = self.step(step_id)
        _check(_STEP_EDGES, step.state, target, what=f"step {step_id}")

        previous = step.state
        step.state = target

        expected = self.derive_state()
        if expected is not self.state and expected not in _RUN_EDGES[self.state]:
            step.state = previous
            raise IllegalTransition(
                f"step {step_id} 想进 {target.value}，但 run {self.run_id} 还在 "
                f"{self.state.value}，推导出的 {expected.value} 不是合法后继；"
                f"先把 run advance 到能容纳它的状态"
            )

        event = self._append(
            "step_state",
            at=at,
            state=target,
            step_id=step_id,
            data={"reason": reason, "error_code": error_code},
        )
        self.reconcile(at=at)
        return event

    def record_attempt(
        self,
        step_id: str,
        *,
        started_at: datetime,
        tool: str | None = None,
        args_digest: str | None = None,
        finished_at: datetime | None = None,
        error_code: str | None = None,
        latency_ms: float | None = None,
        detail: str = "",
    ) -> Attempt:
        """记一次尝试。**幂等**：同一个 `n` 不会被记两次（重放时会撞上）。"""

        step = self.step(step_id)
        n = len(step.attempts) + 1
        attempt = Attempt(
            n=n,
            started_at=started_at,
            tool=tool if tool is not None else step.tool,
            args_digest=args_digest,
            finished_at=finished_at,
            error_code=error_code,
            latency_ms=latency_ms,
            detail=detail,
        )
        step.attempts.append(attempt)
        self._append(
            "attempt",
            at=finished_at or started_at,
            step_id=step_id,
            attempt=n,
            data={"error_code": error_code, "latency_ms": latency_ms},
        )
        return attempt

    def reconcile(self, *, at: datetime | None = None) -> RunState | None:
        """把 `state` 对齐到 `derive_state()`。分叉时留一条 `reconciled` 事件。

        这是"任务显示 running，但某一步卡在等审批"的结构性解法：
        状态不是人手写的，是算出来的；人写错了会被这里纠正并留下痕迹。

        推导出的状态**不可达**时抛 `IllegalTransition`（例如 run 还在 `CREATED`
        就有 step 跑起来了——那说明有人跳过了 `PLANNING/RUNNING`）。
        """

        expected = self.derive_state()
        if expected is self.state:
            return None

        if expected not in _RUN_EDGES[self.state]:
            raise IllegalTransition(
                f"run {self.run_id} 当前 {self.state.value}，推导出 {expected.value}，"
                f"但这不是合法后继；请先 advance 到能容纳它的状态"
            )

        previous = self.state
        self.state = expected
        self._append(
            "reconciled",
            at=at,
            state=expected,
            data={"from": previous.value, "derived": True},
        )
        return expected

    def start(self, *, at: datetime | None = None) -> None:
        """`CREATED → PLANNING → RUNNING`。

        写成一步是刻意的：几乎所有调用方都是"接了任务就规划、规划完就跑"，
        让他们自己连两次 `advance` 只会多两处可以忘。
        """

        if self.state is RunState.CREATED:
            self.advance(RunState.PLANNING, at=at, reason="start")
        if self.state is RunState.PLANNING:
            self.advance(RunState.RUNNING, at=at, reason="start")

    def cancel(self, *, at: datetime | None = None, reason: str = "") -> list[Event]:
        """取消：先取消未完成的步骤，再取消 run。终态步骤不动。"""

        events: list[Event] = []
        for step in self.steps:
            if not step.finished:
                events.append(
                    self.mark_step(step.step_id, RunState.CANCELLED, at=at, reason=reason)
                )
        if self.state not in _TERMINAL:
            events.append(self.advance(RunState.CANCELLED, at=at, reason=reason))
        return events

    def note(
        self,
        kind: str,
        *,
        at: datetime | None = None,
        step_id: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> Event:
        """追加一条**非状态**事件（`resumed_skip` 这类）。

        留一个公开入口，是为了让 `runtime/resume.py` 不必去碰 `_append`：
        续跑事件必须落在同一个事件流里，否则重放时会丢一段历史。
        """

        return self._append(kind, at=at, step_id=step_id, data=data)

    def _append(
        self,
        kind: str,
        *,
        at: datetime | None = None,
        state: RunState | None = None,
        step_id: str | None = None,
        attempt: int | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> Event:
        event = Event(
            seq=len(self.events) + 1,
            at=at or utcnow(),
            kind=kind,
            state=state,
            step_id=step_id,
            attempt=attempt,
            data=dict(data or {}),
        )
        self.events.append(event)
        return event

    # -- 序列化 / 重放 ------------------------------------------------------ #

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "steps": [item.as_dict() for item in self.steps],
            "events": [item.as_dict() for item in self.events],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Run:
        return cls(
            run_id=str(payload["run_id"]),
            state=RunState(payload["state"]),
            steps=[Step.from_dict(item) for item in payload.get("steps") or ()],
            events=[Event.from_dict(item) for item in payload.get("events") or ()],
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            goal=str(payload.get("goal") or ""),
        )

    def replay(self, events: Sequence[Event], *, from_state: RunState = RunState.CREATED) -> Run:
        """**只读事件**重建内存态。

        不读"某张表的当前值"：那张表才是最容易被人手改、被并发写坏的东西，
        而事件流是追加的，冲突只可能以"顺序不同"出现，不会以"值被覆盖"出现。
        """

        self.state = from_state
        # 事件流本身就是历史：重放后的对象必须"带着"这段历史，
        # 否则它无法再被渲染、再被序列化——那就不叫重建，叫"算了个状态"。
        self.events = list(events)
        for item in self.steps:
            item.state = RunState.CREATED
            item.attempts.clear()

        for event in events:
            if event.kind == "run_state" and event.state is not None:
                self.state = event.state
            elif event.kind == "step_state" and event.step_id and event.state is not None:
                step = self.find_step(event.step_id)
                if step is not None:
                    step.state = event.state
            elif event.kind == "reconciled" and event.state is not None:
                self.state = event.state
        return self

    @classmethod
    def from_events(
        cls,
        *,
        run_id: str,
        steps: Iterable[tuple[str, str]],
        events: Sequence[Event],
        created_at: datetime | None = None,
        goal: str = "",
    ) -> Run:
        """从"计划骨架 + 事件流"复原。骨架只需要 `(step_id, goal)`——

        目标文本是**计划**的产物，不是执行过程的产物；执行只改变状态，
        所以重放不需要（也不该）从事件里猜目标。
        """

        run = cls(
            run_id=run_id,
            created_at=created_at or utcnow(),
            goal=goal,
            steps=[Step(step_id=sid, goal=goal_text) for sid, goal_text in steps],
        )
        return run.replay(events)


# --------------------------------------------------------------------------- #
# 旧枚举 → 九态的映射表
# --------------------------------------------------------------------------- #

# 四张映射表放在一个地方，`test_runtime_model.py` 会穷举每个旧枚举值，
# 缺一个就红——这是"真的收敛了"与"又加了一套"的分界线。
LEGACY_MAPPINGS: dict[str, dict[str, RunState]] = {
    # W2 `TaskStatus`
    "task_status": {
        "queued": RunState.CREATED,
        "running": RunState.RUNNING,
        "succeeded": RunState.COMPLETED,
        "failed": RunState.FAILED,
    },
    # W9 `StepStatus`
    "step_status": {
        "pending": RunState.CREATED,
        "running": RunState.RUNNING,
        "done": RunState.COMPLETED,
        "failed": RunState.FAILED,
        "skipped": RunState.CANCELLED,
    },
    # W10 审批
    "approval": {
        "pending": RunState.WAITING_APPROVAL,
        "approved": RunState.RUNNING,
        "rejected": RunState.FAILED,
        "expired": RunState.FAILED,
    },
}


def state_from_legacy(kind: str, value: str) -> RunState:
    """把旧枚举值翻成九态之一。翻不出来就抛——**不猜**。

    静默兜底（比如未知值一律当 RUNNING）会让"某个状态没接上"
    看起来像"任务还在跑"，这是我们最不想要的那种安静。
    """

    table = LEGACY_MAPPINGS.get(kind)
    if table is None:
        raise KeyError(f"未知的旧状态体系 {kind!r}（已知：{sorted(LEGACY_MAPPINGS)}）")
    try:
        return table[value]
    except KeyError:
        raise KeyError(
            f"{kind} 的值 {value!r} 没有映射（已知：{sorted(table)}）"
        ) from None


# 供 `Run.replay` 之外的复查工具用：一条事件的状态是否与它的 kind 自洽。
def is_terminal(state: RunState) -> bool:
    return state in _TERMINAL


# 看板口径：**"等人"不是"卡住"**。
# 把 WAITING_APPROVAL 算进"卡住"，SLO 的"任务卡住率"会在每一次正常审批等待时上升，
# 于是真正卡住的任务被淹掉——D1 的告警随之失效。
_NOT_STUCK = frozenset(
    {
        RunState.WAITING_APPROVAL,
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.CANCELLED,
    }
)


def state_breakdown(runs: Iterable[Run]) -> dict[str, int]:
    """按状态计数。看板直接读它，不用在前端再写一遍九态枚举。"""

    counts: dict[str, int] = {}
    for run in runs:
        key = run.state.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def stuck_runs(runs: Iterable[Run]) -> list[Run]:
    """真正卡住的 run（等人审批不算）。

    判据只有一条：既没结束、也不在等人。别再加"超过 N 分钟"这种阈值——
    阈值属于告警规则（`ops/alerts.yml` 的 `for:`），不属于模型。
    """

    return [run for run in runs if run.state not in _NOT_STUCK]


def clone(run: Run) -> Run:
    """深拷贝（跨"重启"用）。`replace` 只做浅拷贝，steps 会共享。"""

    return Run.from_dict(run.as_dict())

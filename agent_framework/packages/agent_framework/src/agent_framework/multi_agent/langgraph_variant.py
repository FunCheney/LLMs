"""LangGraph 对照实现（W9 D4）。**只做对照，不替换主实现。**

同一张图：

```
START → dispatch ─┬→ researcher ─┐
                  ├→ validator ──┼→ dispatch
                  └→ END         │
```

三条角色规则、三道守卫、执行与记账**全部复用** `supervisor.py` 里同一份代码，
唯一的差别是「谁在驱动这个循环」：

| | 自研（`supervisor.py`） | 本文件 |
|---|---|---|
| 循环 | `while True` 在 `Supervisor.run()` 里 | 图的边，由框架调度 |
| 守卫位置 | 循环顶部，一个函数里连排三个 `if` | 全部塞进 `dispatch` 节点内部 |
| 中止方式 | `break` + 具名 `StopReason` | 写 `route = END`，或框架抛 `GraphRecursionError` |
| 业务状态 | 一个 `SharedState` 对象 | 仍然是一个 `SharedState` 对象——**没进 graph state** |

最后一行是关键发现，不是偷懒：LangGraph 的 state 通道靠 reducer 合并，
而 reducer 的默认语义是「合并/追加」，**它不会拒绝一次基于旧快照的写入**。
把 `SharedState` 交给它，乐观锁就没了。所以这里的 graph state 只剩一个
路由字段，业务状态留在外面——这不是"没用好框架"，是两套状态机制本来就
不在同一个抽象层。详见 `docs/w9/langgraph-vs-inhouse.md`。

依赖：`langgraph` 是**可选依赖**，没装时本模块 import 不报错，
`LANGRAPH_AVAILABLE` 为 False，测试自动 skip（离线档不受影响）。
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

from ..planner.base import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_PENDING,
    StepRecord,
    StepRunner,
    TaskResult,
)
from ..planner.controls import StopReason
from ..skills.executor import SkillExecutor
from .handoff import Handoff, HandoffLedger, LoopDetected
from .roles import (
    ROLE_RESEARCHER,
    ROLE_SUPERVISOR,
    ROLE_VALIDATOR,
    AgentRole,
    ResearcherRole,
    SupervisorRole,
    ValidatorRole,
)
from .shared_state import SharedState, StaleStateError
from .supervisor import (
    SupervisorConfig,
    ask_for,
    build_result,
    run_role_actions,
)

try:  # pragma: no cover - 取决于环境里装没装
    from langgraph.errors import GraphRecursionError
    from langgraph.graph import END, START, StateGraph

    LANGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover

    class GraphRecursionError(RuntimeError):  # type: ignore[no-redef]
        """占位：没装 langgraph 时给测试一个能 except 的类。"""

    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment,misc]
    LANGRAPH_AVAILABLE = False


__all__ = ["LANGRAPH_AVAILABLE", "LangGraphSupervisor"]


class _Route(TypedDict, total=False):
    """graph state。只剩一个路由字段，理由见模块 docstring。"""

    route: str


@dataclass(slots=True)
class _Run:
    """业务账本。放在闭包里而不是 graph state 里——乐观锁不能被 reducer 吃掉。"""

    state: SharedState
    ledger: HandoffLedger
    records: list[StepRecord] = field(default_factory=list)
    tokens: int = 0
    rounds: int = 0
    last_role: str = "start"
    continue_role: str = ""
    validated: bool = False
    stop: str = ""
    detail: str = ""


class LangGraphSupervisor:
    """用 LangGraph 表达同一张图。接口与 `Supervisor` 对齐，方便并排跑。"""

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
        recursion_limit: int = 200,
    ) -> None:
        if not LANGRAPH_AVAILABLE:
            raise RuntimeError(
                "未安装 langgraph：`uv pip install langgraph` 后再跑对照档"
            )

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
        self._roles.setdefault(ROLE_SUPERVISOR, self._scheduler)
        self._clock = clock
        self._recursion_limit = recursion_limit

    # -- 主入口 ------------------------------------------------------------ #

    async def run(self, task: str) -> TaskResult:
        run = _Run(state=SharedState(task=task), ledger=HandoffLedger(
            repeat_threshold=self._config.repeat_threshold
        ))
        started = self._clock()
        graph = self._build(run, task)

        try:
            await graph.ainvoke(
                {"route": ""}, config={"recursion_limit": self._recursion_limit}
            )
        except GraphRecursionError as exc:
            # 框架的守卫：它数的是 superstep，抛异常而不是给一个具名原因。
            # 这个原因码不在我们的 `StopReason` 里——这正是边界成本，见文档。
            run.stop = "recursion_limit"
            run.detail = f"langgraph 触发 recursion_limit={self._recursion_limit}：{exc}"

        return self._result(run, started)

    # -- 图的装配 ---------------------------------------------------------- #

    def _build(self, run: _Run, task: str) -> Any:
        roles = {name: role for name, role in self._roles.items() if name != ROLE_SUPERVISOR}

        async def dispatch(update: _Route) -> Any:
            # 上一节点叫停了就不再派活。
            if run.stop:
                return {"route": END}

            # -- 守卫 1/2/3 ---------------------------------------------- #
            # 全部在节点内部。框架不认识"预算"，它只认识 superstep 数。
            if run.rounds >= self._config.max_steps:
                run.stop, run.detail = (
                    StopReason.MAX_STEPS.value,
                    f"角色轮数达上限 {self._config.max_steps}",
                )
                return {"route": END}
            if run.tokens >= self._config.max_budget_tokens:
                run.stop = StopReason.MAX_TOKENS.value
                run.detail = (
                    f"token 达上限 {self._config.max_budget_tokens}"
                    f"（已用 {run.tokens}，累计而非每跳重置）"
                )
                return {"route": END}

            role_name = self._scheduler.pick(
                run.state,
                last_role=run.last_role,
                validated=run.validated,
                continue_role=run.continue_role,
            )
            if role_name is None:
                run.stop = StopReason.COMPLETED.value
                return {"route": END}
            if role_name not in roles:
                run.stop, run.detail = (
                    StopReason.ERROR.value,
                    f"角色 {role_name!r} 未注册（或试图让 supervisor 执行）",
                )
                return {"route": END}

            handoff = Handoff(
                from_role=run.last_role,
                to_role=role_name,
                task=ask_for(role_name, run.state, task),
                artifacts=tuple(
                    f"f{len(run.state.findings) - i}"
                    for i in range(min(3, len(run.state.findings)))
                ),
                budget_left_tokens=max(
                    0, self._config.max_budget_tokens - run.tokens
                ),
                step_no=run.rounds,
            )
            try:
                run.ledger.record(handoff)
            except LoopDetected as exc:
                run.stop, run.detail = StopReason.PLAN_LOOP.value, str(exc)
                return {"route": END}

            run.rounds += 1
            return {"route": role_name}

        def make_actor(role_name: str) -> Any:
            async def act(update: _Route) -> Any:
                outcome = await self._act_once(run, role_name, task)
                if outcome is not None:
                    run.stop, run.detail = outcome
                    return {"route": END}
                return {"route": ""}

            return act

        graph = StateGraph(_Route)
        # `add_node` 的 overload 围绕 `TypedDictLike` / `TypedDictLikeV2` 展开，
        # mypy 在把 async 闭包匹配到 `_Node` 协议时会失败（同样签名的角色节点
        # 因为走 `make_actor` 的 `Any` 返回而绕过了检查）。这是框架类型签名的
        # 摩擦点，记在这里，而不是为了迁就它把节点写死成四个。
        graph.add_node("dispatch", dispatch)  # type: ignore[call-overload]
        for name in roles:
            graph.add_node(name, make_actor(name))
        graph.add_edge(START, "dispatch")
        graph.add_conditional_edges(
            "dispatch",
            _router,
            {**{name: name for name in roles}, END: END},
        )
        for name in roles:
            # 角色节点跑完一律回 dispatch：**下一棒由 dispatch 决定**，
            # 角色自己不连下一个角色。这一点和自研版是同一个不变式。
            graph.add_edge(name, "dispatch")
        return graph.compile()

    async def _act_once(
        self, run: _Run, role_name: str, task: str
    ) -> tuple[str, str] | None:
        """跑一个角色的一轮。返回 `(stop_reason, detail)` = 要收尾了。"""

        role = self._roles[role_name]
        snapshot = run.state.version
        result = await role.act(
            task,
            run.state,
            budget_left=max(0, self._config.max_budget_tokens - run.tokens),
        )
        run.tokens += result.tokens

        records, pending = await run_role_actions(self._runner, result, role_name=role_name)
        run.records.extend(records)

        try:
            run.state.commit(
                role_name,
                findings=result.findings,
                asked=result.asked,
                tokens=result.tokens,
                expected_version=snapshot,
            )
        except StaleStateError as exc:
            return StopReason.ERROR.value, str(exc)

        if pending is not None:
            # 注意：这里返回的是"要收尾"，但语义是 pending 而不是 aborted。
            return "pending", pending

        if role_name == ROLE_VALIDATOR and not result.needs_fix:
            run.validated = True

        produced = bool(result.actions or result.findings or result.asked)
        if not produced and role_name == run.last_role:
            return (
                StopReason.NO_PROGRESS.value,
                f"{role_name} 连续两轮没有产出：{result.note}",
            )
        run.last_role = role_name
        run.continue_role = role_name if result.can_continue else ""
        return None

    # -- 收尾 -------------------------------------------------------------- #

    def _result(self, run: _Run, started: float) -> Any:
        if run.stop == "pending":
            status, reason = STATUS_PENDING, StopReason.COMPLETED.value
        elif run.stop in ("", StopReason.COMPLETED.value):
            status, reason = STATUS_COMPLETED, StopReason.COMPLETED.value
        else:
            status = STATUS_ABORTED
            if run.state.findings:
                status = STATUS_PARTIAL
            reason = run.stop

        return build_result(
            state=run.state,
            ledger=run.ledger,
            records=run.records,
            tokens=run.tokens,
            status=status,
            reason=reason,
            detail=run.detail,
            seconds=float(self._clock() - started),
        )


def _router(update: _Route) -> str:
    return update.get("route") or END

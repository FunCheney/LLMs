"""W9 D4：自研 supervisor 与 LangGraph 对照实现——同一张图，结果必须可比。

这些用例回答一个具体问题：**把 driver 换掉之后，行为会不会变。**
如果会变，那么"守卫写在编排层最外层"这件事就只是自研的巧合，而不是设计；
如果不变，说明这套守卫的语义是可迁移的，换框架也不会丢。

未装 `langgraph` 时整文件 skip（它是 extra，不是主依赖）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from agent_framework.multi_agent import (
    Finding,
    SharedState,
    StaleStateError,
    Supervisor,
    SupervisorConfig,
)
from agent_framework.multi_agent.langgraph_variant import (
    LANGRAPH_AVAILABLE,
    LangGraphSupervisor,
)
from agent_framework.multi_agent.roles import RoleResult
from agent_framework.planner import STATUS_COMPLETED, STATUS_PARTIAL, StopReason
from agent_framework.retrieval.acl import Principal
from agent_framework.skills import (
    PermissionGuard,
    SandboxPolicy,
    SkillExecutor,
    SkillRegistry,
)
from agent_framework.skills.servers import SearchServer, WorkspaceServer
from agent_framework.skills.servers.upstream import SearchHit

pytestmark = pytest.mark.skipif(
    not LANGRAPH_AVAILABLE, reason="未安装 langgraph（`uv sync --extra langgraph`）"
)

EDITOR = Principal(subject_id="editor", groups=frozenset({"editor"}))


class FakeSearch:
    def __init__(self, *, url: bool = True) -> None:
        self.calls: list[str] = []
        self._url = url

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.calls.append(query)
        link = f"https://example.com/{len(self.calls)}" if self._url else ""
        return [SearchHit(f"{query} 的结论", link, "snippet", 1)]


def _env(tmp_path: Path, *, url: bool = True) -> tuple[SkillExecutor, FakeSearch]:
    upstream = FakeSearch(url=url)
    sandbox = SandboxPolicy(write_roots=(tmp_path,), command_prefixes=("echo",))
    executor = SkillExecutor(
        registry=SkillRegistry(),
        clients={
            "search": SearchServer(upstream).client(),
            "workspace": WorkspaceServer(root=tmp_path, sandbox=sandbox).client(),
        },
        sandbox=sandbox,
        guard=PermissionGuard(approver=_approve),
    )
    return executor, upstream


async def _approve(skill: object, arguments: object, summary: str) -> str:
    return "approved"


def _both(
    tmp_path: Path,
    *,
    config: SupervisorConfig | None = None,
    roles: dict[str, object] | None = None,
    scheduler: object = None,
    questions: tuple[str, ...] = ("问题一", "问题二"),
    recursion_limit: int = 200,
) -> tuple[Supervisor, FakeSearch, LangGraphSupervisor, FakeSearch]:
    """同样的装配来两份：两个 driver 必须吃同样的输入。"""

    common = {
        "config": config,
        "roles": roles,  # type: ignore[dict-item]
        "scheduler": scheduler,  # type: ignore[dict-item]
        "principal": EDITOR,
        "stage": "research",
        "questions": questions,
    }
    ex_a, up_a = _env(tmp_path)
    ex_b, up_b = _env(tmp_path)
    return (
        Supervisor(executor=ex_a, **common),  # type: ignore[arg-type]
        up_a,
        LangGraphSupervisor(
            executor=ex_b, recursion_limit=recursion_limit, **common  # type: ignore[arg-type]
        ),
        up_b,
    )


# --------------------------------------------------------------------------- #
# 1. 结果可比
# --------------------------------------------------------------------------- #


class TestParity:
    async def test_happy_path_agrees(self, tmp_path: Path) -> None:
        """正常流程：状态、轮数、上游调用次数、报告正文全部一致。"""

        sup, up_a, graph, up_b = _both(tmp_path)
        a = await sup.run("调研一件事")
        b = await graph.run("调研一件事")

        assert a.status == b.status == STATUS_COMPLETED
        assert a.stop_reason == b.stop_reason
        assert a.trace.turns == b.trace.turns
        assert a.trace.step_count == b.trace.step_count
        assert up_a.calls == up_b.calls
        # 报告正文一致 → 说明两边拿到的是同一批证据，不是"各查各的"
        assert a.answer == b.answer

    async def test_missing_citation_agrees(self, tmp_path: Path) -> None:
        """拿不到出处时的行为也要一致（不能一边判失败一边放行）。"""

        ex_a, _ = _env(tmp_path, url=False)
        ex_b, _ = _env(tmp_path, url=False)
        questions = ("问题一",)
        a = await Supervisor(
            executor=ex_a, principal=EDITOR, stage="research", questions=questions
        ).run("调研一件事")
        b = await LangGraphSupervisor(
            executor=ex_b, principal=EDITOR, stage="research", questions=questions
        ).run("调研一件事")

        assert a.status == b.status
        assert a.stop_reason == b.stop_reason
        assert "未采信" in a.answer
        assert a.answer == b.answer

    async def test_loop_guard_agrees(self, tmp_path: Path) -> None:
        """互踢皮球：两个 driver 都在同一轮、以同一个原因停下。"""

        sup, _, graph, _ = _both(
            tmp_path,
            config=SupervisorConfig(max_steps=50),
            roles={"a": _PingPong("a"), "b": _PingPong("b")},
            scheduler=_Alternating(),
        )
        a = await sup.run("做点什么")
        b = await graph.run("做点什么")

        assert a.stop_reason == b.stop_reason == StopReason.PLAN_LOOP.value
        assert a.trace.turns == b.trace.turns
        assert a.status == b.status == STATUS_PARTIAL

    async def test_budget_guard_agrees(self, tmp_path: Path) -> None:
        """预算守卫：停在同一个 token 数上。

        这条最关键——如果 LangGraph 版把预算算成"每跳重置"，它会跑得更久，
        而人眼看报告只会觉得"框架版更勤奋"。
        """

        sup, _, graph, _ = _both(
            tmp_path,
            config=SupervisorConfig(max_steps=50, max_budget_tokens=250),
            roles={f"r{i}": _Costly(f"r{i}") for i in range(50)},
            scheduler=_Sequence(),
        )
        a = await sup.run("做点什么")
        b = await graph.run("做点什么")

        assert a.stop_reason == b.stop_reason == StopReason.MAX_TOKENS.value
        assert a.trace.tokens == b.trace.tokens == 300
        assert a.trace.turns == b.trace.turns


# --------------------------------------------------------------------------- #
# 2. 不可迁移的那部分：框架自己的中止原因
# --------------------------------------------------------------------------- #


class TestFrameworkOnlyStop:
    async def test_recursion_limit_is_not_one_of_our_reasons(self, tmp_path: Path) -> None:
        """`recursion_limit` 不在 `StopReason` 里。

        这是**边界成本**，不是 bug：框架数的是 superstep，抛异常而不是给一个
        具名原因。要用它就得扩枚举、或者接受两套口径。自研版没有这个问题——
        它永远能说出一个原因。
        """

        _, _, graph, _ = _both(
            tmp_path,
            config=SupervisorConfig(max_steps=50),
            roles={"a": _PingPong("a"), "b": _PingPong("b")},
            scheduler=_Alternating(),
            recursion_limit=3,
        )
        result = await graph.run("做点什么")

        assert result.stop_reason == "recursion_limit"
        assert result.stop_reason not in {item.value for item in StopReason}
        # 被框架掐断时没有产出报告的余地，但答案字段仍然是"诚实"的空结论
        assert result.status in {STATUS_PARTIAL, "aborted"}

    async def test_our_guards_fire_before_the_framework_limit(
        self, tmp_path: Path
    ) -> None:
        """把 recursion_limit 放大，停下的原因就回到我们自己的枚举里。

        换句话说：**框架的守卫不能替代业务守卫**，只能兜底。
        """

        _, _, graph, _ = _both(
            tmp_path,
            config=SupervisorConfig(max_steps=50),
            roles={"a": _PingPong("a"), "b": _PingPong("b")},
            scheduler=_Alternating(),
            recursion_limit=200,
        )
        result = await graph.run("做点什么")
        assert result.stop_reason == StopReason.PLAN_LOOP.value


# --------------------------------------------------------------------------- #
# 3. graph state 的边界
# --------------------------------------------------------------------------- #


class TestGraphStateBoundary:
    async def test_business_state_is_not_in_graph_state(self, tmp_path: Path) -> None:
        """`SharedState` 没进 graph state。

        不是偷懒：LangGraph 的 state 靠 reducer 合并，而 reducer 的默认语义
        （合并/追加）**不会拒绝一次基于旧快照的写入**。把 `SharedState` 交出去，
        乐观锁就没了。所以 graph state 只剩一个路由字段。
        """

        from agent_framework.multi_agent.langgraph_variant import _Route

        assert list(_Route.__annotations__) == ["route"]
        assert "shared" not in _Route.__annotations__

    async def test_version_still_guards_commits(self, tmp_path: Path) -> None:
        """少了 graph state，版本号照旧生效——写的是同一个 `SharedState`。"""

        state = SharedState(task="t")
        state.commit("a", findings=[Finding("x")])
        with pytest.raises(StaleStateError):
            state.commit("b", expected_version=0)


# --------------------------------------------------------------------------- #
# 与 supervisor 测试共用的假角色
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _PingPong:
    name: str = "a"

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        return RoleResult(
            findings=[Finding(claim=f"{self.name} 说还得再看看", role=self.name)],
            note="再看看",
        )


@dataclass(slots=True)
class _Costly:
    name: str = "r0"
    tokens_per_round: int = 100

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        return RoleResult(
            findings=[Finding(claim=f"第 {len(state.findings) + 1} 条")],
            tokens=self.tokens_per_round,
            note="烧钱",
        )


@dataclass(slots=True)
class _Alternating:
    _turn: int = 0

    def pick(
        self,
        state: SharedState,
        *,
        last_role: str = "",
        validated: bool = False,
        continue_role: str = "",
    ) -> str | None:
        name = ("a", "b")[self._turn % 2]
        self._turn += 1
        return name


@dataclass(slots=True)
class _Sequence:
    limit: int = 200
    _turn: int = 0

    def pick(
        self,
        state: SharedState,
        *,
        last_role: str = "",
        validated: bool = False,
        continue_role: str = "",
    ) -> str | None:
        if self._turn >= self.limit:
            return None
        name = f"r{self._turn}"
        self._turn += 1
        return name

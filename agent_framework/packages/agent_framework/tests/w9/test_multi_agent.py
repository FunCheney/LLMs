"""W9 D3：多智能体编排（三道守卫、环路指纹、预算累计、共享状态版本号）。

这三个守卫的共同点是**都只在"出事那一刻"才有用**，所以不能用"平时跑得通"
来证明。每条守卫都配一条反向用例：把触发条件造出来，断言真的中止了。

同样不能省的是一条**正常流程**用例。只有反向用例的话，"守卫写得太狠、
把正常流程也砍了"这个 bug 反而测不出来——validator 判 `needs_fix` 时
也不产出任何东西，跟"角色空转"长得一模一样。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from agent_framework.multi_agent import (
    ROLE_RESEARCHER,
    ROLE_SUPERVISOR,
    ROLE_VALIDATOR,
    Finding,
    Handoff,
    HandoffLedger,
    LoopDetected,
    ResearcherRole,
    RoleAction,
    RoleResult,
    SharedState,
    StaleStateError,
    Supervisor,
    SupervisorConfig,
    SupervisorRole,
    ValidatorRole,
)
from agent_framework.planner import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_PENDING,
)
from agent_framework.retrieval.acl import Principal
from agent_framework.skills import (
    PermissionGuard,
    SandboxPolicy,
    SkillExecutor,
    SkillRegistry,
)
from agent_framework.skills.servers import SearchServer, WorkspaceServer
from agent_framework.skills.servers.upstream import SearchHit

EDITOR = Principal(subject_id="editor", groups=frozenset({"editor"}))


# --------------------------------------------------------------------------- #
# 测试环境
# --------------------------------------------------------------------------- #


class FakeSearch:
    """上游：每次返回一条带 URL 的命中。

    **必须带 URL**：`_citation_of` 只认 http(s)，没有 URL 的观察会被
    validator 判成"缺引用"，然后我们就分不清"守卫拦住了"与"根本没证据"。
    """

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


def _supervisor(
    executor: SkillExecutor,
    *,
    config: SupervisorConfig | None = None,
    roles: dict[str, object] | None = None,
    scheduler: object = None,
    questions: tuple[str, ...] = ("问题一", "问题二"),
) -> Supervisor:
    return Supervisor(
        executor=executor,
        config=config,
        roles=roles,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        principal=EDITOR,
        stage="research",
        questions=questions,
    )


# --------------------------------------------------------------------------- #
# 1. 环路指纹
# --------------------------------------------------------------------------- #


class TestLoopFingerprint:
    def test_key_ignores_step_no_and_artifacts(self) -> None:
        """指纹**只看 (from, to, task)**。

        带上 `step_no` 每次都不一样 → 永远不会重复 → 检测等于没有；
        带上 `artifacts` → 同一件事换个说法重问就逃掉了，而那是要抓的。
        """

        a = Handoff("a", "b", "做一件", artifacts=("f1",), step_no=1)
        b = Handoff("a", "b", "做一件", artifacts=("f2", "f3"), step_no=7)
        assert a.key == b.key

        c = Handoff("a", "b", "做另一件", step_no=1)
        assert a.key != c.key

    def test_same_edge_repeat_is_detected(self) -> None:
        ledger = HandoffLedger(repeat_threshold=2)
        ledger.record(Handoff("a", "b", "做一件"))
        with pytest.raises(LoopDetected) as exc:
            ledger.record(Handoff("a", "b", "做一件"))
        assert "重复" in str(exc.value)
        # 触发的那条 handoff 要留在流水里：它是证据。
        assert len(ledger.handoffs) == 2

    def test_two_point_pingpong_with_different_tasks(self) -> None:
        """A→B→A→B，**每次任务都不一样**。

        只查"同边重复"会漏掉它（每条边的指纹都不相同），这正是最典型的
        "互踢皮球"：每个人换句话把球踢回去。第 3 条交接时角色序列已经拼出
        `a→b→a→b`，所以断在这里，不必等第 4 条。
        """

        ledger = HandoffLedger(repeat_threshold=2)
        ledger.record(Handoff("a", "b", "第 0 个说法"))
        ledger.record(Handoff("b", "a", "第 1 个说法"))
        with pytest.raises(LoopDetected) as exc:
            ledger.record(Handoff("a", "b", "第 2 个说法"))

        assert "角色循环" in str(exc.value)

    def test_same_role_in_a_row_is_not_a_cycle(self) -> None:
        """同一个角色连着被派（研究员逐个查子问题）**不是互踢皮球**。

        这条是踩出来的：最初没有这个限制，两个子问题的正常流程在第 4 轮
        被误判成环，报告只剩一半还标着 `plan_loop`。
        """

        ledger = HandoffLedger(repeat_threshold=2)
        for i in range(4):
            ledger.record(Handoff(ROLE_RESEARCHER, ROLE_RESEARCHER, f"检索：T（第 {i} 轮）"))
        assert ledger.loop_reason() is None

    def test_normal_flow_has_no_false_positive(self) -> None:
        """researcher 查 3 个不同子问题 + 1 次校验：**不是环**。"""

        ledger = HandoffLedger(repeat_threshold=2)
        ledger.record(Handoff("start", ROLE_RESEARCHER, "检索：T（第 1 轮）"))
        ledger.record(Handoff(ROLE_RESEARCHER, ROLE_RESEARCHER, "检索：T（第 2 轮）"))
        ledger.record(Handoff(ROLE_RESEARCHER, ROLE_RESEARCHER, "检索：T（第 3 轮）"))
        ledger.record(Handoff(ROLE_RESEARCHER, ROLE_VALIDATOR, "校验引用：T"))
        assert ledger.loop_reason() is None


# --------------------------------------------------------------------------- #
# 2. 正常流程（守卫不能把正常流程也砍了）
# --------------------------------------------------------------------------- #


class TestHappyPath:
    async def test_research_then_validate_completes(self, tmp_path: Path) -> None:
        executor, upstream = _env(tmp_path)
        result = await _supervisor(executor).run("对比两个方案")

        assert result.status == STATUS_COMPLETED
        assert result.ok
        assert "https://example.com/" in result.answer
        assert len(upstream.calls) == 2
        # 两个子问题各查一次 + 一次校验 = 3 轮交接
        assert result.trace.turns == 3

    async def test_answer_partitions_cited_and_uncited(self, tmp_path: Path) -> None:
        """没有 URL 的观察进"未采信"区，**不进结论**。"""

        executor, _ = _env(tmp_path, url=False)
        result = await _supervisor(executor).run("对比两个方案")

        assert "未采信（缺出处）" in result.answer
        assert "没有拿到带出处的结论" in result.answer
        # 拿不到引用 → 校验不通过 → 不是 completed
        assert result.status != STATUS_COMPLETED
        assert not result.ok


# --------------------------------------------------------------------------- #
# 3. 守卫一：步数
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _EmptyRole:
    """空转角色：每一轮都不产出任何东西。"""

    name: str = "spinner"

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        return RoleResult(note="我很忙，但什么也没做")


@dataclass(slots=True)
class _StubbornScheduler:
    """永远派同一个空转角色。用来逼出"步数"与"无进展"两条守卫。

    刻意**忽略** `continue_role`：这个调度器代表"坏掉的调度"，它不听话本身
    就是被测对象。
    """

    role: str = "spinner"

    def pick(
        self,
        state: SharedState,
        *,
        last_role: str = "",
        validated: bool = False,
        continue_role: str = "",
    ) -> str | None:
        return self.role


class TestStepGuard:
    async def test_repeated_empty_role_stops_on_no_progress(
        self, tmp_path: Path
    ) -> None:
        """同一个角色连着两轮空手 → 停。

        这条守的是**只记 warning 不中断**那个坑：空转角色的 `act` 不产出
        任何东西，步数如果按"工具调用次数"数，这里会永远数到 0。
        """

        executor, _ = _env(tmp_path)
        sup = _supervisor(
            executor,
            config=SupervisorConfig(max_steps=50),
            roles={"spinner": _EmptyRole()},
            scheduler=_StubbornScheduler(),
        )
        result = await sup.run("做点什么")

        assert result.stop_reason == "no_progress"
        assert result.status == STATUS_ABORTED
        assert sup.trace.rounds == 2
        assert sup.trace.tool_calls == 0

    async def test_step_guard_is_a_real_ceiling(self, tmp_path: Path) -> None:
        """`max_steps` 是硬上限，不是"建议"。"""

        executor, _ = _env(tmp_path)
        sup = _supervisor(
            executor,
            config=SupervisorConfig(max_steps=3),
            roles={"spinner": _EmptyRole()},
            scheduler=_StubbornScheduler(),
        )
        result = await sup.run("做点什么")

        assert sup.trace.rounds <= 3
        assert result.stop_reason in {"max_steps", "no_progress"}


# --------------------------------------------------------------------------- #
# 4. 守卫二：预算**累计**（不每跳重置）
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _CostlyRole:
    """每轮固定申报一份成本。用它把"每跳重置预算"这个 bug 逼出来。"""

    tokens_per_round: int = 100
    name: str = "costly"

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        return RoleResult(
            findings=[Finding(claim=f"第 {len(state.findings) + 1} 条")],
            tokens=self.tokens_per_round,
            note="烧钱",
        )


@dataclass(slots=True)
class _SequenceScheduler:
    """依次派出 r0、r1、r2……**每个角色只用一次**。

    这样环路指纹一定不会触发，测到的就只会是预算守卫。两个守卫分开测，
    否则"到底是谁拦住的"说不清，两个守卫都等于没测。
    """

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


def _many_costly(rounds: int) -> dict[str, object]:
    return {f"r{i}": _CostlyRole(name=f"r{i}") for i in range(rounds)}


class TestBudgetGuard:
    async def test_budget_accumulates_across_handoffs(self, tmp_path: Path) -> None:
        """预算跨 handoff **累计**。

        如果每一跳都把预算重置成配置值，每一跳的"余额"看起来都会充足，
        总额却是配置值的 N 倍。这正是要防的假象。
        """

        executor, _ = _env(tmp_path)
        sup = _supervisor(
            executor,
            config=SupervisorConfig(max_steps=50, max_budget_tokens=250),
            roles=_many_costly(50),
            scheduler=_SequenceScheduler(),
        )
        result = await sup.run("做点什么")

        assert result.stop_reason == "max_tokens"
        # 3 轮 × 100 = 300：第 4 轮开始前发现 300 >= 250 才停。
        assert sup.trace.rounds == 3
        assert sup.trace.tokens == 300
        assert result.trace.tokens == 300

    async def test_bigger_budget_runs_longer(self, tmp_path: Path) -> None:
        """反证：把上限从 250 放大到 1000，轮数从 3 变成 10。

        "每跳重置"的实现在这两个用例里都会跑满 `max_steps`，轮数一样；
        只有**累计**才会出现"上限越大、轮数越多"。
        """

        executor, _ = _env(tmp_path)
        sup = _supervisor(
            executor,
            config=SupervisorConfig(max_steps=50, max_budget_tokens=1000),
            roles=_many_costly(50),
            scheduler=_SequenceScheduler(),
        )
        result = await sup.run("做点什么")

        assert result.stop_reason == "max_tokens"
        assert sup.trace.rounds == 10
        assert sup.trace.tokens == 1000

    async def test_autonomy_budget_and_handoff_budget_are_not_the_same_number(
        self,
    ) -> None:
        """两个上限名字像但含义不同：一个是角色轮数，一个是 token。

        `SupervisorConfig.max_steps` 数的是"角色行动一轮"（可能含多次工具调用），
        `AutonomyBudget.max_steps` 数的是执行步数。混用会让两边的成本对不上。
        """

        config = SupervisorConfig(max_steps=12, max_budget_tokens=20_000)
        with pytest.raises(ValueError):
            SupervisorConfig(max_steps=0).validate()
        with pytest.raises(ValueError):
            SupervisorConfig(max_budget_tokens=0).validate()
        with pytest.raises(ValueError):
            SupervisorConfig(repeat_threshold=1).validate()
        config.validate()  # 默认值合法


# --------------------------------------------------------------------------- #
# 5. 守卫三：环路（集成）
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _PingPongRole:
    """不产出、只把球踢回去的角色。两个配成一对就是互踢皮球。"""

    name: str = "a"

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        # 有产出（一条无引用发现），所以不会被"无进展"守卫先拦住；
        # 这一条专测环路守卫本身。
        return RoleResult(
            findings=[Finding(claim=f"{self.name} 说还得再看看", role=self.name)],
            note="再看看",
        )


@dataclass(slots=True)
class _AlternatingScheduler:
    names: tuple[str, str] = ("a", "b")
    _turn: int = 0

    def pick(
        self,
        state: SharedState,
        *,
        last_role: str = "",
        validated: bool = False,
        continue_role: str = "",
    ) -> str | None:
        name = self.names[self._turn % 2]
        self._turn += 1
        return name


class TestLoopGuardIntegration:
    async def test_pingpong_aborts_within_fingerprint(self, tmp_path: Path) -> None:
        """两个角色互相 handoff → 在指纹检测处中止，**不是**跑满 MAX_STEPS。"""

        executor, _ = _env(tmp_path)
        sup = _supervisor(
            executor,
            config=SupervisorConfig(max_steps=50),
            roles={"a": _PingPongRole(name="a"), "b": _PingPongRole(name="b")},
            scheduler=_AlternatingScheduler(),
        )
        result = await sup.run("做点什么")

        assert result.stop_reason == "plan_loop"
        # 第 4 条交接才构成 A→B 的重复，所以只跑了 3 轮。
        assert sup.trace.rounds == 3
        assert sup.trace.rounds < 50
        # 有半成品 → partial，不是 aborted：拿到的东西是真的。
        assert result.status == STATUS_PARTIAL

    async def test_supervisor_role_cannot_execute(self, tmp_path: Path) -> None:
        """supervisor 被调度成执行角色时直接报错，而不是"顺手也检索一下"。"""

        executor, _ = _env(tmp_path)
        sup = _supervisor(
            executor,
            scheduler=_StubbornScheduler(role=ROLE_SUPERVISOR),
        )
        result = await sup.run("做点什么")

        assert result.stop_reason == "error"
        assert "只该调度" in result.detail

    async def test_unregistered_role_is_an_error(self, tmp_path: Path) -> None:
        executor, _ = _env(tmp_path)
        sup = _supervisor(executor, scheduler=_StubbornScheduler(role="ghost"))
        result = await sup.run("做点什么")

        assert result.stop_reason == "error"
        assert "未注册" in result.detail


# --------------------------------------------------------------------------- #
# 6. 共享状态版本号
# --------------------------------------------------------------------------- #


class TestSharedStateVersion:
    def test_stale_commit_is_rejected(self) -> None:
        """并发"读-改-写"必须被检出——否则一方的工在无报错的情况下消失。"""

        state = SharedState(task="t")
        snapshot_a = state.version
        snapshot_b = state.version  # 两个角色同时读到 0

        state.commit("a", findings=[Finding("a 的发现")], expected_version=snapshot_a)
        with pytest.raises(StaleStateError) as exc:
            state.commit(
                "b", findings=[Finding("b 的发现")], expected_version=snapshot_b
            )

        assert "版本已变" in str(exc.value)
        # b 的写入被拒绝，a 的还在
        assert [item.claim for item in state.findings] == ["a 的发现"]

    def test_commit_without_expected_version_is_allowed(self) -> None:
        """单线程场景可以不传；但**并发场景一定得传**，这是调用方的责任。"""

        state = SharedState(task="t")
        state.commit("a", tokens=10)
        state.commit("b", tokens=5)
        assert state.version == 2
        assert state.budget_used_tokens == 15

    def test_history_can_replay_who_changed_what(self) -> None:
        state = SharedState(task="t")
        state.commit("a", findings=[Finding("x")])
        state.commit("b", findings=[Finding("y")])
        assert state.history == [(1, "a", 1), (2, "b", 1)]

    async def test_supervisor_reports_conflict_instead_of_dropping_it(
        self, tmp_path: Path
    ) -> None:
        """`act` 期间有人抢先写了状态 → supervisor 报冲突，不静默重试。"""

        executor, _ = _env(tmp_path)
        state_holder: dict[str, SharedState] = {}

        @dataclass(slots=True)
        class _RacingRole:
            name: str = ROLE_RESEARCHER

            async def act(
                self, task: str, state: SharedState, *, budget_left: int
            ) -> RoleResult:
                state_holder["state"] = state
                # 模拟"另一路在 await 期间提交了状态"
                state.commit("intruder", tokens=1)
                return RoleResult(tokens=1)

        sup = _supervisor(executor, roles={ROLE_RESEARCHER: _RacingRole()})
        result = await sup.run("做点什么")

        assert result.stop_reason == "error"
        assert "版本已变" in result.detail


# --------------------------------------------------------------------------- #
# 7. 角色职责单一
# --------------------------------------------------------------------------- #


class TestRoleBoundaries:
    def test_supervisor_role_has_no_tools(self) -> None:
        """supervisor 角色**没有 executor、没有 runner**——想越界也没工具可用。

        "职责单一"如果只写在文档里，第一次赶工就会被破。这里断言它压根没有
        可调用的东西。
        """

        role = SupervisorRole()
        assert not hasattr(role, "executor")
        assert not hasattr(role, "runner")
        assert not hasattr(role, "_runner")

    async def test_supervisor_role_refuses_to_act(self) -> None:
        role = SupervisorRole()
        with pytest.raises(NotImplementedError):
            await role.act("t", SharedState(task="t"), budget_left=100)

    def test_pick_order(self) -> None:
        """四条判断的顺序：没证据 → 缺引用 → 没校验 → 收尾。"""

        role = SupervisorRole()
        state = SharedState(task="t")

        assert role.pick(state) == ROLE_RESEARCHER

        state.commit("r", findings=[Finding("有引用", citation="https://x")])
        assert role.pick(state, validated=False) == ROLE_VALIDATOR
        assert role.pick(state, validated=True) is None

        # 缺引用时回研究员补；已经在研究员手里还补不上 → 交校验判失败
        state.commit("r", findings=[Finding("没引用")])
        assert role.pick(state, last_role="validator") == ROLE_RESEARCHER
        assert role.pick(state, last_role=ROLE_RESEARCHER) == ROLE_VALIDATOR

    async def test_roles_do_not_touch_state(self) -> None:
        """角色 `act` **不许有副作用**：状态只由 supervisor 提交。"""

        state = SharedState(task="t")
        roles = (
            ResearcherRole(questions=("q1",)),
            ValidatorRole(),
            SupervisorRole(),
        )
        for role in roles[:2]:
            before = state.version
            await role.act("t", state, budget_left=100)
            assert state.version == before
            assert state.findings == []

    async def test_researcher_skips_already_asked(self) -> None:
        state = SharedState(task="t", asked=["q1"])
        result = await ResearcherRole(questions=("q1", "q2")).act(
            "t", state, budget_left=100
        )
        assert [item.arguments["query"] for item in result.actions] == ["q2"]

    async def test_researcher_reports_nothing_when_exhausted(self) -> None:
        state = SharedState(task="t", asked=["q1"])
        result = await ResearcherRole(questions=("q1",)).act("t", state, budget_left=100)
        assert result.actions == []
        assert result.asked == []

    async def test_validator_flags_uncited(self) -> None:
        state = SharedState(task="t", findings=[Finding("没引用")])
        result = await ValidatorRole().act("t", state, budget_left=100)
        assert result.needs_fix
        assert "没有引用" in result.note

    async def test_validator_passes_with_citation(self) -> None:
        state = SharedState(
            task="t", findings=[Finding("有引用", citation="https://x")]
        )
        result = await ValidatorRole().act("t", state, budget_left=100)
        assert not result.needs_fix


# --------------------------------------------------------------------------- #
# 8. 需要人批的动作 → pending
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _PublisherRole:
    """想直接发布。控制权矩阵里"是否发布"归人，所以必须挂起。"""

    name: str = "publisher"

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        return RoleResult(
            actions=[RoleAction(tool="publish_post", arguments={}, question="q")],
            note="我想发布",
        )


class TestHumanGate:
    async def test_publish_action_suspends(self, tmp_path: Path) -> None:
        executor, _ = _env(tmp_path)
        sup = _supervisor(
            executor,
            roles={"publisher": _PublisherRole()},
            scheduler=_StubbornScheduler(role="publisher"),
        )
        result = await sup.run("发布一篇")

        assert result.status == STATUS_PENDING
        assert "人工审批" in result.detail
        # 没执行 → 没有轨迹记录
        assert sup.trace.tool_calls == 0

    async def test_search_action_does_not_suspend(self, tmp_path: Path) -> None:
        """反向：普通检索不该被"是否发布"这条拦下来。"""

        executor, _ = _env(tmp_path)
        result = await _supervisor(executor).run("对比两个方案")
        assert result.status == STATUS_COMPLETED


# --------------------------------------------------------------------------- #
# 9. D5 需要的轨迹字段
# --------------------------------------------------------------------------- #


class TestTraceForEval:
    async def test_trace_carries_records_and_handoff_fingerprints(
        self, tmp_path: Path
    ) -> None:
        """D5 的四个过程指标要从这里取数，字段缺一个就算不出来。"""

        executor, _ = _env(tmp_path)
        result = await _supervisor(executor).run("对比两个方案")

        trace = result.trace
        assert trace.step_count == 2  # 两次 web_search
        assert trace.turns == 3  # 三次角色交接
        assert len(trace.plan_fingerprints) == 3
        assert all(record.tool == "web_search" for record in trace.records)
        assert trace.tokens > 0

    async def test_steps_used_is_tool_calls_not_rounds(self, tmp_path: Path) -> None:
        """`steps_done` 与 `turns` 不是同一个数——混用会让成本对不上。"""

        executor, _ = _env(tmp_path)
        result = await _supervisor(executor).run("对比两个方案")
        assert result.steps_done == 2
        assert result.steps_total == 3

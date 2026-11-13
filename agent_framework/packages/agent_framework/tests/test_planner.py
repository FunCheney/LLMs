"""W9 D1：规划器地基（状态机、三道上限、死循环、控制权矩阵）。

这些用例的共同点：**每一条都能在不联网、不调模型的条件下断言**。
守卫类逻辑如果只能在真机上验证，最后一定会被 skip 掉，然后就等于没有。
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from agent_framework.planner import (
    CONTROL_MATRIX,
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    AutonomyBudget,
    BudgetMeter,
    ControlViolation,
    Decision,
    Plan,
    PlanExecutePlanner,
    PlanStep,
    PolicyReasoner,
    PolicyTask,
    ReActPlanner,
    ScriptedReasoner,
    StepRunner,
    StepStateError,
    StepStatus,
    StopReason,
    TaskPolicy,
    matrix,
)
from agent_framework.planner.base import plan_fingerprint
from agent_framework.retrieval.acl import Principal
from agent_framework.skills import PermissionGuard, SandboxPolicy, SkillExecutor, SkillRegistry
from agent_framework.skills.servers import SearchServer, WorkspaceServer
from agent_framework.skills.servers.upstream import SearchHit

EDITOR = Principal(subject_id="editor", groups=frozenset({"editor"}))


# --------------------------------------------------------------------------- #
# 测试环境
# --------------------------------------------------------------------------- #


class CountingSearch:
    """数调用次数的上游。**这是"重规划不重跑已完成步骤"的唯一判据**：
    只看 Plan 的状态字段会漏掉"状态对了但副作用发生了两次"。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.calls.append(query)
        url = f"https://example.com/{len(self.calls)}"
        return [SearchHit(f"{query} 的结果", url, "snippet", 1)]


def _env(tmp_path: Path, *, stage: str = "draft") -> tuple[SkillExecutor, CountingSearch]:
    upstream = CountingSearch()
    sandbox = SandboxPolicy(write_roots=(tmp_path,), command_prefixes=("echo",))
    workspace = WorkspaceServer(root=tmp_path, sandbox=sandbox)
    executor = SkillExecutor(
        registry=SkillRegistry(),
        clients={
            "search": SearchServer(upstream).client(),
            "workspace": workspace.client(),
        },
        sandbox=sandbox,
        guard=PermissionGuard(approver=_approve),
    )
    return executor, upstream


async def _approve(skill: object, arguments: object, summary: str) -> str:
    return "approved"


def _runner(
    executor: SkillExecutor, *, stage: str = "draft", principal: Principal = EDITOR
) -> StepRunner:
    return StepRunner(executor, principal=principal, stage=stage)


def _step(step_id: str, tool: str | None, **arguments: object) -> PlanStep:
    return PlanStep(id=step_id, goal=f"做 {step_id}", tool=tool, arguments=arguments)


# --------------------------------------------------------------------------- #
# 1. 计划状态机
# --------------------------------------------------------------------------- #


class TestPlanStateMachine:
    def test_ready_steps_empty_until_dependency_done(self) -> None:
        plan = Plan(
            task="t",
            steps=[
                _step("s1", "web_search", query="aa"),
                replace(_step("s2", "web_search", query="b"), depends_on=("s1",)),
            ],
        )
        assert [step.id for step in plan.ready_steps()] == ["s1"]

        running = plan.with_status("s1", StepStatus.RUNNING)
        done = running.with_status("s1", StepStatus.DONE)
        assert [step.id for step in done.ready_steps()] == ["s2"]

    def test_illegal_transition_rejected(self) -> None:
        plan = Plan(task="t", steps=[_step("s1", None)])
        done = plan.with_status("s1", StepStatus.RUNNING).with_status("s1", StepStatus.DONE)
        # done 是终态：允许回到 running 会让"完成过又重跑"变成可能，
        # 而这件事同时污染幂等与成本统计。
        with pytest.raises(StepStateError):
            done.with_status("s1", StepStatus.RUNNING)

    def test_dependency_on_missing_step_blocks_forever(self) -> None:
        """指向不存在 id 的依赖必须**阻塞**而不是被忽略。

        忽略的话，一个拼错的依赖会让步骤提前执行，而错误永远不会暴露。
        """

        plan = Plan(
            task="t",
            steps=[replace(_step("s2", None), depends_on=("typo-id",))],
        )
        assert plan.ready_steps() == []
        assert not plan.is_finished()

    def test_fingerprint_ignores_status(self) -> None:
        plan = Plan(task="t", steps=[_step("s1", "x"), _step("s2", "y")])
        moved = plan.with_status("s1", StepStatus.RUNNING)
        # 状态进了指纹，"重复计划检测"就永远不会命中——这恰好是它要防的失效。
        assert plan.fingerprint == moved.fingerprint
        assert plan_fingerprint(plan.steps) == plan.fingerprint


# --------------------------------------------------------------------------- #
# 2. 三道上限（每一条都必须真的中断）
# --------------------------------------------------------------------------- #


class TestBudgetGuards:
    async def test_max_steps_returns_partial(self, tmp_path: Path) -> None:
        executor, upstream = _env(tmp_path)
        # 故意给一个永远做不完的计划：5 步但只允许 2 步。
        steps = [_step(f"s{i}", "web_search", query=f"q{i}") for i in range(5)]
        plan = Plan(task="t", steps=steps)
        reasoner = ScriptedReasoner(
            plan=plan,
            actions=[
                Decision(kind="tool", tool="web_search", arguments={"query": f"q{i}"}, tokens=5)
                for i in range(5)
            ],
        )
        planner = PlanExecutePlanner(
            reasoner=reasoner, runner=_runner(executor), budget=AutonomyBudget(max_steps=2)
        )
        result = await planner.run("t")

        assert result.status == STATUS_PARTIAL
        assert result.stop_reason == StopReason.MAX_STEPS.value
        # 关键：不是"记一条 warning 继续跑"——只执行了 2 步。
        assert result.trace.step_count == 2
        assert len(upstream.calls) == 2

    async def test_max_tokens_returns_partial(self, tmp_path: Path) -> None:
        executor, _ = _env(tmp_path)
        plan = Plan(task="t", steps=[_step("s1", "web_search", query="aa")])
        reasoner = ScriptedReasoner(
            plan=plan,
            actions=[
                Decision(kind="tool", tool="web_search", arguments={"query": "aa"}, tokens=10_000)
            ],
            tokens_per_call=10_000,
        )
        planner = PlanExecutePlanner(
            reasoner=reasoner, runner=_runner(executor), budget=AutonomyBudget(max_tokens=50)
        )
        result = await planner.run("t")
        assert result.status == STATUS_PARTIAL
        assert result.stop_reason == StopReason.MAX_TOKENS.value

    async def test_max_cost_returns_partial(self, tmp_path: Path) -> None:
        executor, _ = _env(tmp_path)
        planner = PlanExecutePlanner(
            reasoner=ScriptedReasoner(plan=Plan(task="t", steps=[])),
            runner=_runner(executor),
            budget=AutonomyBudget(max_cost=Decimal("0.01")),
        )
        # 直接从计量器上扣，模拟工具/模型调用产生的成本
        planner.meter.charge_cost(Decimal("0.02"))
        result = await planner.run("t")
        assert result.status == STATUS_PARTIAL
        assert result.stop_reason == StopReason.MAX_COST.value

    async def test_max_seconds_returns_partial(self, tmp_path: Path) -> None:
        """用假时钟触发，不 sleep。

        `sleep` 触发超时会让用例慢且偶发，最后被 skip——而 skip 掉的守卫
        等于不存在。
        """

        executor, _ = _env(tmp_path)
        now = [0.0]

        def clock() -> float:
            return now[0]

        plan = Plan(task="t", steps=[_step("s1", "web_search", query="aa")])
        planner = PlanExecutePlanner(
            reasoner=ScriptedReasoner(
                plan=plan,
                actions=[Decision(kind="tool", tool="web_search", arguments={"query": "aa"})],
            ),
            runner=_runner(executor),
            budget=AutonomyBudget(max_seconds=1.0),
            clock=clock,
        )
        now[0] = 5.0
        result = await planner.run("t")
        assert result.status == STATUS_PARTIAL
        assert result.stop_reason == StopReason.MAX_SECONDS.value

    def test_budget_check_order_is_fixed(self) -> None:
        now = [0.0]
        meter = BudgetMeter(
            AutonomyBudget(max_steps=1, max_tokens=1, max_cost=Decimal("0"), max_seconds=1.0),
            clock=lambda: now[0],
        )
        meter.count_step()
        meter.charge_tokens(10)
        meter.charge_cost(Decimal("1"))
        now[0] = 10.0
        # 四项同时超限时报告"步数"：顺序固定才能让日志做趋势。
        assert meter.exhausted() == StopReason.MAX_STEPS


# --------------------------------------------------------------------------- #
# 3. 重规划与死循环
# --------------------------------------------------------------------------- #


class TestReplanAndLoop:
    async def test_failed_step_triggers_replan(self, tmp_path: Path) -> None:
        executor, _ = _env(tmp_path)
        plan = Plan(task="t", steps=[_step("s1", "no_such_tool")])
        reasoner = ScriptedReasoner(
            plan=plan,
            actions=[Decision(kind="tool", tool="no_such_tool")],
            replan_plan=Plan(task="t", steps=[]),
        )
        planner = PlanExecutePlanner(reasoner=reasoner, runner=_runner(executor))
        result = await planner.run("t")

        assert reasoner.replan_calls == 1
        assert result.trace.replans == 1

    async def test_replan_does_not_rerun_done_steps(self, tmp_path: Path) -> None:
        """重规划后，已完成步骤不得重跑。

        这是"重规划"最隐蔽的成本：状态字段看起来对，但副作用发生了两次。
        只断言 Plan 的 status 会漏掉它，所以这里数的是**上游被调了几次**。
        """

        executor, upstream = _env(tmp_path)
        plan = Plan(
            task="t",
            steps=[
                _step("s1", "web_search", query="第一次"),
                _step("s2", "no_such_tool"),
            ],
        )
        # 重规划把失败的 s2 换掉，但**保留 s1**
        replanned = Plan(
            task="t",
            steps=[
                _step("s1", "web_search", query="第一次"),
                _step("s2b", "web_search", query="换一种"),
            ],
        )
        reasoner = ScriptedReasoner(
            plan=plan,
            actions=[
                Decision(kind="tool", tool="web_search", arguments={"query": "第一次"}),
                Decision(kind="tool", tool="no_such_tool"),
                Decision(kind="tool", tool="web_search", arguments={"query": "换一种"}),
            ],
            replan_plan=replanned,
        )
        planner = PlanExecutePlanner(reasoner=reasoner, runner=_runner(executor))
        result = await planner.run("t")

        assert result.status == STATUS_COMPLETED
        # s1 只搜了一次。若 _merge_replan 丢了状态，这里会是 2。
        assert upstream.calls == ["第一次", "换一种"]

    async def test_repeated_plan_aborts(self, tmp_path: Path) -> None:
        """重规划产出同一份计划 → 判定原地打转，中止。

        `ScriptedReasoner.replan_plan=None` 表示"原样返回"，这是构造死循环
        最短的路径：现实里模型改写一遍 goal 但步骤结构不变，指纹也相同。
        """

        executor, _ = _env(tmp_path)
        plan = Plan(task="t", steps=[_step("s1", "no_such_tool")])
        reasoner = ScriptedReasoner(
            plan=plan,
            actions=[Decision(kind="tool", tool="no_such_tool")] * 6,
        )
        planner = PlanExecutePlanner(
            reasoner=reasoner, runner=_runner(executor), budget=AutonomyBudget(max_replans=4)
        )
        result = await planner.run("t")

        assert result.status == STATUS_PARTIAL
        assert result.stop_reason == StopReason.PLAN_LOOP.value
        # 早停：4 次重规划额度没用完就停了，因为指纹已经重复。
        assert result.trace.replans < 4

    async def test_max_replans_stops(self, tmp_path: Path) -> None:
        executor, _ = _env(tmp_path)
        plan = Plan(task="t", steps=[_step("s1", "no_such_tool")])
        # 每次重规划都换一个新步骤 id，指纹变化 → 不会触发 PLAN_LOOP，
        # 只能靠 max_replans 兜住。两种守卫测的是不同的东西。
        replans = [
            Plan(task="t", steps=[_step(f"s1-{i}", "no_such_tool")]) for i in range(1, 6)
        ]
        reasoner = ScriptedReasoner(
            plan=plan,
            actions=[Decision(kind="tool", tool="no_such_tool")] * 6,
            replan_plan=replans[0],
        )
        planner = PlanExecutePlanner(
            reasoner=reasoner, runner=_runner(executor), budget=AutonomyBudget(max_replans=1)
        )
        result = await planner.run("t")
        assert result.status == STATUS_PARTIAL
        assert result.stop_reason == StopReason.MAX_REPLANS.value

    async def test_cyclic_dependency_aborts_not_hangs(self, tmp_path: Path) -> None:
        """依赖成环必须**立刻**报错，不能等到预算耗尽——那是另一种"卡住"。"""

        executor, _ = _env(tmp_path)
        plan = Plan(
            task="t",
            steps=[
                replace(_step("a", None), depends_on=("b",)),
                replace(_step("b", None), depends_on=("a",)),
            ],
        )
        planner = PlanExecutePlanner(
            reasoner=ScriptedReasoner(plan=plan), runner=_runner(executor)
        )
        result = await planner.run("t")
        assert result.status != STATUS_COMPLETED
        assert "卡死" in result.detail


# --------------------------------------------------------------------------- #
# 4. 执行不越界（规划器不许重写执行）
# --------------------------------------------------------------------------- #


class TestExecutionBoundary:
    async def test_invisible_tool_is_denied_and_recorded(self, tmp_path: Path) -> None:
        """看不见的工具被调用 → 记录成 denied，且**不是** ok。

        `invalid_call_rate` 靠这个区分：如果规划器把 denied 当成功，
        过程指标会显示"零无效调用"，而实际上每一步都被挡了。
        """

        executor, _ = _env(tmp_path, stage="research")
        plan = Plan(task="t", steps=[_step("s1", "publish_post")])
        reasoner = ScriptedReasoner(
            plan=plan,
            actions=[
                Decision(
                    kind="tool", tool="publish_post", arguments={"title": "t", "body": "b" * 40}
                )
            ],
            replan_plan=Plan(task="t", steps=[]),
        )
        planner = PlanExecutePlanner(reasoner=reasoner, runner=_runner(executor, stage="research"))
        result = await planner.run("t")

        assert result.trace.step_count == 1
        assert result.trace.records[0].code == "tool_not_visible"
        assert result.trace.records[0].is_invalid

    async def test_visible_tools_reflect_stage(self, tmp_path: Path) -> None:
        executor, _ = _env(tmp_path)
        research = _runner(executor, stage="research").visible_tools()
        draft = _runner(executor, stage="draft").visible_tools()
        assert "publish_post" not in research
        assert "write_brief" in draft
        # 控制权矩阵里"工具是否可见"归 code：模型拿到的只是这份视图。
        assert matrix().owner_of("工具是否可见") == "code"


# --------------------------------------------------------------------------- #
# 5. 控制权矩阵不是文档
# --------------------------------------------------------------------------- #


class TestControlMatrix:
    def test_seven_decisions_and_model_only_owns_three(self) -> None:
        assert len(CONTROL_MATRIX) == 7
        model_owned = [name for name, owner in CONTROL_MATRIX.items() if owner == "model"]
        assert len(model_owned) == 3
        assert set(model_owned) == {"下一步做什么", "调用哪个工具", "工具参数"}

    def test_unknown_decision_defaults_to_code(self) -> None:
        # 默认归代码而不是归模型：新增决定会先被拒绝，再补登记。
        # 反过来的默认值会悄悄扩大模型权限。
        assert matrix().owner_of("某个还没登记的决定") == "code"

    def test_assert_code_owned_raises_for_human_decision(self) -> None:
        with pytest.raises(ControlViolation):
            matrix().assert_code_owned("是否发布")


def test_control_matrix_has_no_dead_entries() -> None:
    """矩阵里每个决定都要在运行时被查询过。

    只在文档里出现的矩阵，等于没有矩阵——所以在源码里找 `owner_of(` /
    `human_required(` / `model_may_decide(` 的实参，要求每个键至少出现一次。
    """

    root = Path(__file__).resolve().parents[1] / "src" / "agent_framework"
    sources = list((root / "planner").glob("*.py")) + list(
        (root / "multi_agent").glob("*.py")
    ) if (root / "multi_agent").exists() else list((root / "planner").glob("*.py"))

    text = "\n".join(path.read_text("utf-8") for path in sources)
    # controls.py 自己不算：那里定义的矩阵被自己读到不算"被使用"。
    text = text.replace((root / "planner" / "controls.py").read_text("utf-8"), "")

    missing = [name for name in CONTROL_MATRIX if name not in text]
    assert not missing, f"这些决定只登记没被查询：{missing}"


# --------------------------------------------------------------------------- #
# 6. ReAct 对照组
# --------------------------------------------------------------------------- #


class TestReactBaseline:
    async def test_react_completes_on_finish(self, tmp_path: Path) -> None:
        executor, upstream = _env(tmp_path)
        reasoner = ScriptedReasoner(
            plan=Plan(task="t"),
            actions=[
                Decision(kind="tool", tool="web_search", arguments={"query": "aa"}),
                Decision(kind="finish", text="答案是 A"),
            ],
        )
        planner = ReActPlanner(reasoner=reasoner, runner=_runner(executor))
        result = await planner.run("t")

        assert result.status == STATUS_COMPLETED
        assert "答案是 A" in result.answer
        assert upstream.calls == ["aa"]
        # ReAct 不产出计划：指纹列表为空是它的特征，不是缺数据。
        assert result.trace.plan_fingerprints == []

    async def test_react_stops_at_max_steps(self, tmp_path: Path) -> None:
        executor, _ = _env(tmp_path)
        reasoner = ScriptedReasoner(
            plan=Plan(task="t"),
            actions=[
                Decision(kind="tool", tool="web_search", arguments={"query": f"q{i}"})
                for i in range(10)
            ],
        )
        planner = ReActPlanner(
            reasoner=reasoner, runner=_runner(executor), budget=AutonomyBudget(max_steps=3)
        )
        result = await planner.run("t")
        assert result.status == STATUS_PARTIAL
        assert result.stop_reason == StopReason.MAX_STEPS.value
        assert result.trace.step_count == 3

    async def test_react_without_plan_cannot_replan(self, tmp_path: Path) -> None:
        """ReAct 的失败步只能重试，不能重规划——这是它与 P&E 的本质差别。"""

        executor, _ = _env(tmp_path)
        reasoner = ScriptedReasoner(
            plan=Plan(task="t"),
            actions=[Decision(kind="tool", tool="no_such_tool")] * 3,
        )
        planner = ReActPlanner(
            reasoner=reasoner, runner=_runner(executor), budget=AutonomyBudget(max_steps=3)
        )
        result = await planner.run("t")
        assert result.trace.replans == 0


# --------------------------------------------------------------------------- #
# 7. 策略 reasoner + 计划状态 vs 上下文窗口
# --------------------------------------------------------------------------- #


def _policy_task(*, recall: bool) -> PolicyTask:
    steps: list[PlanStep] = [
        _step("s1", "web_search", query="基准报告"),
        _step("s2", "web_search", query="转化周期"),
        _step("s3", "web_search", query="渠道"),
        _step("s4", "web_search", query="预算占比"),
    ]
    if recall:
        steps.append(_step("s5", "web_search", query="复核"))
    return PolicyTask(
        id="t1",
        question="竞品基准",
        steps=tuple(steps),
        recall_from={4: 0} if recall else {},
    )


class TestPolicyReasoner:
    async def test_policy_plan_execute_keeps_early_result(self, tmp_path: Path) -> None:
        """P&E 的可见状态覆盖整份计划，第 5 步能引用第 1 步的结果。"""

        executor, upstream = _env(tmp_path)
        task = _policy_task(recall=True)
        planner = PlanExecutePlanner(
            reasoner=PolicyReasoner(policy=TaskPolicy(task)),
            runner=_runner(executor),
            observation_window=8,
        )
        result = await planner.run(task.question)

        assert result.status == STATUS_COMPLETED
        # 第 5 步的 query 里带上了第 1 步的真实产出，而不是"未知"。
        assert any("基准报告" in query for query in upstream.calls[1:])
        assert not any("未知" in query for query in upstream.calls)

    async def test_policy_react_loses_early_result_when_window_small(self, tmp_path: Path) -> None:
        """窗口只有 1 条时，ReAct 看不到第 1 步的结果，只能填"未知"。

        这是本周唯一诚实的"机制差异"：不是 ReAct 更笨，是它把中间结果
        放在会被挤掉的上下文窗口里。
        """

        executor, upstream = _env(tmp_path)
        task = _policy_task(recall=True)
        planner = ReActPlanner(
            reasoner=PolicyReasoner(policy=TaskPolicy(task)),
            runner=_runner(executor),
            observation_window=1,
        )
        result = await planner.run(task.question)

        assert result.status == STATUS_COMPLETED
        assert any("未知" in query for query in upstream.calls)

    async def test_policy_react_recovers_with_wide_window(self, tmp_path: Path) -> None:
        """把窗口放大到覆盖全程，ReAct 就不再丢早期结果。

        **这条用例的存在是为了防止结论被讲错**：如果 ReAct 只是"窗口不够"，
        那 P&E 的优势就来自"结构化保存中间结果"，而不是来自"多智能体更聪明"。
        没有这条对照，前面那条很容易被写成后者。
        """

        executor, upstream = _env(tmp_path)
        task = _policy_task(recall=True)
        planner = ReActPlanner(
            reasoner=PolicyReasoner(policy=TaskPolicy(task)),
            runner=_runner(executor),
            observation_window=16,
        )
        result = await planner.run(task.question)

        assert result.status == STATUS_COMPLETED
        assert not any("未知" in query for query in upstream.calls)
        assert any("基准报告" in query for query in upstream.calls[1:])

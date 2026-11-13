"""W11 D3：统一执行模型——九态、非法迁移、旧枚举收敛、事件重放。

这组用例守五条：

1. **非法迁移抛错**，不是静默忽略（静默会让"状态对不上"变成最难查的一类 bug）；
2. **run 状态可以从 steps 推导**，分叉时 `reconcile` 纠正并留痕——
   这是"任务显示 running，但某一步卡在等审批"的结构性解法；
3. **`WAITING_APPROVAL` 是独立状态**，不是 running 的子状态（否则 D1 的图全部失真）；
4. **旧枚举必须全部有映射**——穷举 W2/W9/W10 三套状态，缺一个就红，
   这是"真的收敛了"与"又加了一套"的分界线；
5. **事件流能独立重建状态**，不依赖任何"表的当前值"。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from agent_framework.error_codes import KNOWN_CODES, is_caller_code, is_reliability_code
from agent_framework.runtime.model import (
    LEGACY_MAPPINGS,
    Attempt,
    Event,
    IllegalTransition,
    Run,
    RunState,
    Step,
    allowed_from_run,
    allowed_from_step,
    clone,
    state_from_legacy,
)

T0 = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def new_run(*steps: tuple[str, str]) -> Run:
    run = Run(run_id="run-1", goal="写一篇关于 MCP 的文章")
    for step_id, goal in steps:
        run.add_step(step_id, goal)
    return run


# --------------------------------------------------------------------------- #
# 状态机
# --------------------------------------------------------------------------- #


def test_nine_states_are_the_whole_vocabulary() -> None:
    assert {item.value for item in RunState} == {
        "created",
        "planning",
        "running",
        "waiting_tool",
        "waiting_approval",
        "tool_failed",
        "completed",
        "failed",
        "cancelled",
    }


def test_terminal_states_have_no_outgoing_edges() -> None:
    """COMPLETED / CANCELLED 没有出边。

    `COMPLETED → RUNNING` 如果能走，"完成过又重跑"会让幂等与成本统计同时失真。
    `FAILED` 是例外：续跑就是"失败的 run 接着跑"，见下一条。
    """

    for state in (RunState.COMPLETED, RunState.CANCELLED):
        assert allowed_from_run(state) == frozenset()
        assert allowed_from_step(state) == frozenset()


def test_failed_can_be_resumed_but_completed_cannot() -> None:
    """失败可以续跑（同一个 run_id、同一条事件流），完成不能复活。"""

    assert RunState.RUNNING in allowed_from_run(RunState.FAILED)
    assert RunState.RUNNING not in allowed_from_run(RunState.COMPLETED)
    assert RunState.RUNNING in allowed_from_step(RunState.FAILED)


def test_step_never_enters_planning() -> None:
    """计划是 run 级的产物，步骤不会被"规划"。"""

    for state in RunState:
        assert RunState.PLANNING not in allowed_from_step(state), state


def test_illegal_transition_raises_with_the_allowed_set() -> None:
    run = new_run(("step-1", "检索"))
    run.start()
    run.mark_step("step-1", RunState.RUNNING)
    run.mark_step("step-1", RunState.COMPLETED)

    with pytest.raises(IllegalTransition) as excinfo:
        run.mark_step("step-1", RunState.RUNNING)

    message = str(excinfo.value)
    assert "completed" in message and "running" in message
    # 报错里必须带"允许哪些"，否则值班要去看源码
    assert "允许" in message


def test_run_cannot_skip_planning() -> None:
    run = Run(run_id="run-1")
    with pytest.raises(IllegalTransition):
        run.advance(RunState.RUNNING)


def test_step_cannot_start_before_run_starts() -> None:
    """run 还在 CREATED 就有 step 跑起来 → 拒绝，而不是让 reconcile 悄悄把 run 推进。

    这条是"原子性"的守门人：拒绝之前不能留下半截状态。
    """

    run = new_run(("step-1", "检索"))
    with pytest.raises(IllegalTransition):
        run.mark_step("step-1", RunState.RUNNING)

    assert run.step("step-1").state is RunState.CREATED, "被拒的迁移不该留下痕迹"


def test_cancelling_keeps_finished_steps_finished() -> None:
    run = new_run(("step-1", "检索"), ("step-2", "成文"))
    run.start()
    run.mark_step("step-1", RunState.RUNNING)
    run.mark_step("step-1", RunState.COMPLETED)
    run.mark_step("step-2", RunState.RUNNING)

    run.cancel(at=at(10), reason="用户取消")

    assert run.step("step-1").state is RunState.COMPLETED
    assert run.step("step-2").state is RunState.CANCELLED
    assert run.state is RunState.CANCELLED


# --------------------------------------------------------------------------- #
# 状态推导：run 不该是人手写的
# --------------------------------------------------------------------------- #


def test_run_state_is_derived_from_steps() -> None:
    run = new_run(("step-1", "检索"), ("step-2", "发布"))
    run.start()
    run.mark_step("step-1", RunState.RUNNING)
    assert run.state is RunState.RUNNING

    run.mark_step("step-1", RunState.COMPLETED)
    run.mark_step("step-2", RunState.RUNNING)
    run.mark_step("step-2", RunState.WAITING_APPROVAL)

    # 最关键的一条：有一步在等人，run 就不许显示 running
    assert run.state is RunState.WAITING_APPROVAL


def test_reconcile_fixes_a_hand_edited_state_and_leaves_a_trace() -> None:
    """状态被人改错（或旧版本写的）时，reconcile 纠正并留 `reconciled` 事件。"""

    run = new_run(("step-1", "发布"))
    run.start()
    run.mark_step("step-1", RunState.RUNNING)
    run.mark_step("step-1", RunState.WAITING_APPROVAL)

    run.state = RunState.RUNNING  # 模拟"某张表的当前值"被写脏
    events_before = len(run.events)

    assert run.reconcile(at=at(5)) is RunState.WAITING_APPROVAL
    assert len(run.events) == events_before + 1
    assert run.events[-1].kind == "reconciled"
    assert run.events[-1].data["from"] == "running"


def test_reconcile_refuses_an_unreachable_state() -> None:
    """推导出不可达状态时要报错，而不是硬写进去。"""

    run = new_run(("step-1", "检索"))
    run.start()
    run.mark_step("step-1", RunState.RUNNING)
    run.mark_step("step-1", RunState.COMPLETED)
    run.mark_step("step-1", RunState.COMPLETED) if False else None

    # 人为制造一个不可能的组合：run 已 COMPLETED，但步骤回到 running
    run.step("step-1").state = RunState.RUNNING
    with pytest.raises(IllegalTransition):
        run.reconcile()


def test_waiting_approval_is_not_counted_as_stuck() -> None:
    """`WAITING_APPROVAL` 必须是独立状态：混进 running 会让"等人"被算成"卡住"。"""

    run = new_run(("step-1", "发布"))
    run.start()
    run.mark_step("step-1", RunState.RUNNING)
    run.mark_step("step-1", RunState.WAITING_APPROVAL)

    assert run.state is RunState.WAITING_APPROVAL
    assert run.state is not RunState.RUNNING
    assert RunState.WAITING_APPROVAL.value == "waiting_approval"


# --------------------------------------------------------------------------- #
# 旧枚举收敛
# --------------------------------------------------------------------------- #


def test_every_legacy_state_has_a_mapping() -> None:
    """穷举三套旧状态机。缺一个就红——这是"收敛"与"又加一套"的分界。"""

    from agent_framework.guardrails.hitl import APPROVED, EXPIRED, PENDING, REJECTED
    from agent_framework.planner.base import StepStatus
    from agent_framework.queue.base import TaskStatus

    for status in TaskStatus:
        assert state_from_legacy("task_status", status.value)
    for status in StepStatus:
        assert state_from_legacy("step_status", status.value)
    for status in (PENDING, APPROVED, REJECTED, EXPIRED):
        assert state_from_legacy("approval", status)


def test_unknown_legacy_state_raises_instead_of_guessing() -> None:
    """静默兜底（未知一律当 running）会让"某个状态没接上"看起来像"任务还在跑"。"""

    with pytest.raises(KeyError):
        state_from_legacy("step_status", "retrying")
    with pytest.raises(KeyError):
        state_from_legacy("no_such_system", "running")


def test_approval_expired_maps_to_failed_not_running() -> None:
    """过期默认拒绝（W10 红线）：映射到 FAILED，不是 RUNNING。"""

    assert state_from_legacy("approval", "expired") is RunState.FAILED
    assert state_from_legacy("approval", "pending") is RunState.WAITING_APPROVAL


def test_mapping_tables_only_produce_real_states() -> None:
    for table in LEGACY_MAPPINGS.values():
        for value in table.values():
            assert isinstance(value, RunState)


# --------------------------------------------------------------------------- #
# 码表复用（W4 的那一份）
# --------------------------------------------------------------------------- #


def test_error_code_table_is_not_duplicated() -> None:
    """`Attempt` 复用 `error_codes` 那份码表；`ERROR_HTTP` 里的码必须都被归类。"""

    from agent_framework.llm.errors import ERROR_HTTP

    http_codes = {code for _, code in ERROR_HTTP.values()}
    assert http_codes <= KNOWN_CODES, http_codes - KNOWN_CODES


def test_attempt_classifies_faults_without_its_own_rules() -> None:
    upstream = Attempt(n=1, started_at=T0, error_code="UPSTREAM_TIMEOUT")
    caller = Attempt(n=1, started_at=T0, error_code="BAD_REQUEST")
    ok = Attempt(n=1, started_at=T0)

    assert upstream.ok is False and upstream.is_upstream_fault is True
    assert caller.is_caller_fault is True and caller.is_upstream_fault is False
    assert ok.ok is True and ok.is_upstream_fault is False

    # 与共享码表一致，不自己判
    for attempt in (upstream, caller, ok):
        assert attempt.is_upstream_fault is is_reliability_code(attempt.error_code)
        assert attempt.is_caller_fault is is_caller_code(attempt.error_code)


def test_circuit_open_is_a_reliability_code_not_a_caller_one() -> None:
    """熔断打开不是"我们传错了参数"，是上游不健康。"""

    assert is_reliability_code("CIRCUIT_OPEN") is True
    assert is_caller_code("CIRCUIT_OPEN") is False


# --------------------------------------------------------------------------- #
# 事件流：只追加、可重放
# --------------------------------------------------------------------------- #


def build_sample_run() -> Run:
    run = new_run(("step-1", "检索"), ("step-2", "成文"))
    run.start(at=at(0))
    run.mark_step("step-1", RunState.RUNNING, at=at(1))
    run.record_attempt(
        "step-1", started_at=at(1), finished_at=at(2), args_digest="d-1", latency_ms=1000
    )
    run.mark_step("step-1", RunState.COMPLETED, at=at(2))
    run.mark_step("step-2", RunState.RUNNING, at=at(3))
    run.record_attempt(
        "step-2",
        started_at=at(3),
        finished_at=at(33),
        args_digest="d-2",
        error_code="UPSTREAM_TIMEOUT",
        latency_ms=30000,
    )
    run.mark_step(
        "step-2", RunState.FAILED, at=at(33), error_code="UPSTREAM_TIMEOUT", reason="超时"
    )
    return run


def test_events_are_append_only_and_sequenced() -> None:
    run = build_sample_run()
    seqs = [event.seq for event in run.events]
    assert seqs == list(range(1, len(seqs) + 1))


def test_state_can_be_rebuilt_from_events_alone() -> None:
    """只读事件重建 —— 不依赖"某张表的当前值"。"""

    run = build_sample_run()
    rebuilt = Run.from_events(
        run_id=run.run_id,
        steps=[(step.step_id, step.goal) for step in run.steps],
        events=run.events,
        created_at=run.created_at,
        goal=run.goal,
    )

    assert rebuilt.state is run.state is RunState.FAILED
    assert {s.step_id: s.state for s in rebuilt.steps} == {
        "step-1": RunState.COMPLETED,
        "step-2": RunState.FAILED,
    }


def test_replay_ignores_stale_step_states() -> None:
    """重放前先把 steps 复位，否则"重放"会变成"叠加"。"""

    run = build_sample_run()
    rebuilt = Run.from_events(
        run_id=run.run_id,
        steps=[(step.step_id, step.goal) for step in run.steps],
        events=[],
    )
    assert all(step.state is RunState.CREATED for step in rebuilt.steps)
    assert rebuilt.state is RunState.CREATED


def test_event_json_roundtrip() -> None:
    run = build_sample_run()
    for event in run.events:
        assert Event.from_json(event.to_json()) == event


def test_run_json_roundtrip_and_clone_is_deep() -> None:
    run = build_sample_run()
    copy = clone(run)

    assert copy.as_dict() == run.as_dict()
    copy.mark_step("step-2", RunState.RUNNING, at=at(40))
    assert run.step("step-2").state is RunState.FAILED, "clone 必须深拷贝"


def test_step_is_a_plain_object_with_attempts() -> None:
    step = Step(step_id="s", goal="g")
    assert step.done is False and step.finished is False
    run = build_sample_run()
    assert len(run.step("step-2").attempts) == 1
    assert run.step("step-2").last_attempt() is not None


# --------------------------------------------------------------------------- #
# 看板口径：等人 ≠ 卡住
# --------------------------------------------------------------------------- #


def test_stuck_runs_exclude_waiting_approval() -> None:
    """把"等人"算进"卡住"，SLO 的卡住率会在每次正常审批等待时上升，
    真正卡住的任务被淹掉——D1 的告警随之失效。"""

    from agent_framework.runtime.model import state_breakdown, stuck_runs

    blocked = new_run(("step-1", "发布"))
    blocked.run_id = "run-approval"
    blocked.start()
    blocked.mark_step("step-1", RunState.RUNNING)
    blocked.mark_step("step-1", RunState.WAITING_APPROVAL)

    running = new_run(("step-1", "检索"))
    running.run_id = "run-running"
    running.start()
    running.mark_step("step-1", RunState.RUNNING)

    assert stuck_runs([blocked, running]) == [running]
    assert state_breakdown([blocked, running]) == {
        "waiting_approval": 1,
        "running": 1,
    }

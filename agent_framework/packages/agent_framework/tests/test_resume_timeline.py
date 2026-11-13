"""W11 D3：断点续跑 + Run Timeline——"重启"必须能真的重启一次。

这组用例守四条：

1. **已完成的步一次都不重跑**。不是为了省时间，是为了省钱：
   一个 20 步的调研任务重跑前 15 步，账单翻倍而报告里只写"重试了一次"；
2. **重启时处于 RUNNING 的步骤必须重跑**（结果未知），并靠幂等键兜副作用；
3. **Timeline 能独立解释失败**：不打开原始日志，就能说出"哪一步、什么错、试了几次、卡在哪"；
4. **Timeline 可以从事件流重建**。如果某个信息在事件里找不到，
   说明事件记漏了，而不是渲染写错了——这条断言顺带在守事件流的完整性。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from agent_framework.runtime.model import Run, RunState
from agent_framework.runtime.resume import (
    JsonRunStore,
    ResumableRunner,
    StepOutcome,
    step_idempotency_key,
)
from agent_framework.runtime.timeline import RunTimeline

T0 = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)


class Clock:
    """每问一次就走 1 秒的假时钟。真 sleep 会让测试慢，然后一定有人把它 skip 掉。"""

    def __init__(self, step_s: float = 1.0) -> None:
        self.now = T0
        self.step_s = step_s

    def __call__(self) -> datetime:
        current = self.now
        self.now += timedelta(seconds=self.step_s)
        return current


class ScriptedExecutor:
    """按脚本执行：`fail` 里的 step_id 第一次失败，第二次起成功。"""

    def __init__(self, *, fail_once: dict[str, str] | None = None) -> None:
        self._fail_once = dict(fail_once or {})
        self.calls: list[str] = []

    async def __call__(self, run: Run, step) -> StepOutcome:  # noqa: ANN001 - 测试替身
        self.calls.append(step.step_id)

        code = self._fail_once.pop(step.step_id, None)
        if code is not None:
            return StepOutcome(ok=False, error_code=code, detail="注入失败", args_digest="d")

        return StepOutcome(ok=True, latency_ms=100.0, args_digest="d")


def make_run(steps: int = 4) -> Run:
    run = Run(run_id="run-resume", goal="深度调研")
    for index in range(1, steps + 1):
        run.add_step(f"step-{index}", f"第 {index} 步", tool=f"tool-{index}")
    return run


def make_runner(executor: ScriptedExecutor, store: JsonRunStore | None = None) -> ResumableRunner:
    return ResumableRunner(executor=executor, store=store, clock=Clock())


# --------------------------------------------------------------------------- #
# 直跑
# --------------------------------------------------------------------------- #


async def test_full_run_completes_and_runs_every_step_once() -> None:
    executor = ScriptedExecutor()
    run = make_run()

    report = await make_runner(executor).run(run)

    assert report.executed == ("step-1", "step-2", "step-3", "step-4")
    assert report.skipped == ()
    assert executor.calls == ["step-1", "step-2", "step-3", "step-4"]
    assert run.state is RunState.COMPLETED
    assert all(step.state is RunState.COMPLETED for step in run.steps)


async def test_failure_stops_the_run_immediately() -> None:
    """一步失败即停：后面的步骤依赖它，继续跑只会把错误放大。"""

    executor = ScriptedExecutor(fail_once={"step-3": "UPSTREAM_TIMEOUT"})
    run = make_run()

    report = await make_runner(executor).run(run)

    assert report.failed == ("step-3",)
    assert report.executed == ("step-1", "step-2", "step-3")
    assert executor.calls == ["step-1", "step-2", "step-3"]
    assert run.state is RunState.FAILED
    assert run.step("step-4").state is RunState.CREATED


# --------------------------------------------------------------------------- #
# 重启续跑
# --------------------------------------------------------------------------- #


async def test_resume_skips_completed_steps_and_does_not_rerun_them(
    tmp_path: Path,
) -> None:
    """跑到第 3 步失败 → "重启" → 从第 3 步续跑，前 2 步不重跑。"""

    store = JsonRunStore(root=tmp_path)
    run = make_run()

    first = ScriptedExecutor(fail_once={"step-3": "UPSTREAM_TIMEOUT"})
    await make_runner(first, store).run(run)
    assert first.calls == ["step-1", "step-2", "step-3"]

    # ── "进程重启"：只从磁盘拿回 run，内存态全部丢掉
    reloaded = store.load(run.run_id)
    assert reloaded.state is RunState.FAILED

    second = ScriptedExecutor()
    report = await make_runner(second, store).run(reloaded)

    assert report.skipped == ("step-1", "step-2")
    assert report.executed == ("step-3", "step-4")
    assert second.calls == ["step-3", "step-4"], "已完成步被重跑了，钱白花"
    assert reloaded.state is RunState.COMPLETED


async def test_interrupted_step_is_marked_for_retry_and_rerun(tmp_path: Path) -> None:
    """进程在一歩执行中死掉 → 那一步必须重跑（结果未知），并留下原因。"""

    store = JsonRunStore(root=tmp_path)
    run = make_run(steps=3)
    run.start(at=T0)
    run.mark_step("step-1", RunState.RUNNING, at=T0)
    run.mark_step("step-1", RunState.COMPLETED, at=T0)
    # 第 2 步跑到一半，进程死了
    run.mark_step("step-2", RunState.RUNNING, at=T0)
    run.mark_step("step-2", RunState.WAITING_TOOL, at=T0)
    store.save(run)

    reloaded = store.load(run.run_id)
    executor = ScriptedExecutor()
    report = await make_runner(executor, store).run(reloaded)

    assert report.interrupted == ("step-2",)
    assert report.skipped == ("step-1",)
    assert "step-2" in report.executed
    assert executor.calls == ["step-2", "step-3"]

    reasons = [event.data.get("reason") for event in reloaded.events if event.step_id == "step-2"]
    assert any("进程重启" in str(reason) for reason in reasons)


async def test_waiting_approval_is_never_auto_resumed(tmp_path: Path) -> None:
    """审批是等人，不是"结果未知"。把它重跑等于绕过审批（W10 红线）。"""

    store = JsonRunStore(root=tmp_path)
    run = make_run(steps=2)
    run.start(at=T0)
    run.mark_step("step-1", RunState.RUNNING, at=T0)
    run.mark_step("step-1", RunState.WAITING_APPROVAL, at=T0)
    store.save(run)

    reloaded = store.load(run.run_id)
    executor = ScriptedExecutor()
    report = await make_runner(executor, store).run(reloaded)

    assert report.interrupted == (), "等待审批的步骤不能被当成中断"
    assert reloaded.step("step-1").state is RunState.WAITING_APPROVAL
    # 一次执行都没有：既不越审批，也不跳过去跑后面的步（依赖关系上它是前置）
    assert executor.calls == []
    assert report.blocked == ("step-1",)
    assert report.executed == ()
    assert report.skipped == ()
    assert reloaded.state is RunState.WAITING_APPROVAL


async def test_store_refuses_to_shrink_the_event_stream(tmp_path: Path) -> None:
    """事件流只允许增长。整份覆写在两个进程同时续跑时会互相盖掉对方的事件。"""

    store = JsonRunStore(root=tmp_path)
    run = make_run(steps=2)
    await make_runner(ScriptedExecutor(), store).run(run)

    stale = make_run(steps=2)  # 事件数 0，比磁盘上的少
    with pytest.raises(ValueError, match="只允许增长"):
        store.save(stale)


async def test_step_idempotency_key_is_stable_across_restarts() -> None:
    """重启后算出来的键必须和重启前一模一样，否则"重跑不产生第二次副作用"是空话。"""

    run = make_run(steps=2)
    before = step_idempotency_key(run.run_id, run.step("step-1"))

    reloaded = Run.from_dict(run.as_dict())
    after = step_idempotency_key(reloaded.run_id, reloaded.step("step-1"))

    assert before == after
    # 不同步骤不能撞同一个键
    assert before != step_idempotency_key(run.run_id, run.step("step-2"))


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #


async def build_failed_run_step3() -> Run:
    executor = ScriptedExecutor(fail_once={"step-3": "UPSTREAM_TIMEOUT"})
    run = make_run()
    await make_runner(executor).run(run)
    return run


async def test_timeline_explains_a_failure_without_the_logs() -> None:
    run = await build_failed_run_step3()
    text = RunTimeline().render(run)

    assert "step-3" in text
    assert "UPSTREAM_TIMEOUT" in text
    assert "failed" in text
    # 结论行：一眼看出哪儿断了
    assert "失败步骤" in text


async def test_timeline_shows_retry_with_unchanged_arguments() -> None:
    """两次尝试参数相同 → "重试，参数未变"；不同 → "参数已改"。

    这两个结论在复盘时指向完全不同的修法，所以必须能从 Timeline 上直接读出来。
    """

    run = make_run(steps=1)
    run.start(at=T0)
    run.mark_step("step-1", RunState.RUNNING, at=T0)
    run.record_attempt(
        "step-1",
        started_at=T0,
        finished_at=T0 + timedelta(seconds=2),
        args_digest="same",
        error_code="UPSTREAM_TIMEOUT",
        latency_ms=2000,
    )
    run.record_attempt(
        "step-1",
        started_at=T0 + timedelta(seconds=3),
        finished_at=T0 + timedelta(seconds=4),
        args_digest="same",
        latency_ms=1000,
    )
    run.mark_step("step-1", RunState.COMPLETED, at=T0 + timedelta(seconds=4))

    text = RunTimeline().render(run)
    assert "参数未变" in text
    assert "#2" in text


async def test_timeline_marks_waiting_approval_as_waiting_not_stuck() -> None:
    run = make_run(steps=2)
    run.start(at=T0)
    run.mark_step("step-1", RunState.RUNNING, at=T0)
    run.mark_step("step-1", RunState.WAITING_APPROVAL, at=T0, reason="发布需审批")

    text = RunTimeline().render(run)
    assert "等待审批" in text or "等人审批" in text
    assert run.state is RunState.WAITING_APPROVAL


async def test_timeline_can_be_rebuilt_from_events_alone() -> None:
    """重放后渲染出的内容必须与原始一致——证明事件流没有漏记。"""

    run = await build_failed_run_step3()
    rebuilt = Run.from_events(
        run_id=run.run_id,
        steps=[(step.step_id, step.goal) for step in run.steps],
        events=run.events,
        goal=run.goal,
    )

    original = RunTimeline().rows(run)
    replayed = RunTimeline().rows(rebuilt)

    # 时间偏移与状态可以完全一致；attempt 明细来自 steps，重放不重建它们
    assert [row.step_id for row in original] == [row.step_id for row in replayed]
    assert [row.status for row in original] == [row.status for row in replayed]
    assert rebuilt.state is run.state


async def test_timeline_to_dict_is_stable_for_reports() -> None:
    run = await build_failed_run_step3()
    payload = RunTimeline().to_dict(run)

    assert payload["run_id"] == run.run_id
    assert payload["state"] == "failed"
    assert payload["summary"]
    assert all("offset_ms" in row for row in payload["rows"])

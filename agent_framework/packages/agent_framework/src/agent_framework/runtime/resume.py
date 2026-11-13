"""断点续跑（W11 D3）：重启之后，从第 4 步接着跑，前 3 步不重跑。

## 为什么"跳过已完成步"必须靠事件，而不是靠"表的当前值"

如果把 `steps[].state` 存成一张表的当前值，重启后你只知道"第 3 步是 completed"，
不知道"它完成于什么时候、用的什么参数、重试过没有"。续跑时想做任何判断
（比如"这步是 10 分钟前跑的，缓存该不该信"）都无从下手。

所以落盘的是**事件流**（`Run.events`），`run.state` 只是它的投影。
`Run.replay()` 只读事件就能重建，续跑因此不依赖任何"被谁手改过"的字段。

## 重启时还在 RUNNING 的步骤，必须重跑

这是最容易写错的一条。进程在 `RUNNING` 状态死掉时，那次调用**结果未知**：
它可能已经写完了文件，也可能一个字节都没发出去。

- 当成成功 → 静默丢一次副作用（数据缺一块，且没人知道）；
- 当成失败且不再重试 → 任务永远卡住；
- **当成"要重跑，副作用由幂等键兜"** → 唯一正确的选择。

所以恢复时把 `RUNNING` / `WAITING_TOOL` 的步骤标成 `TOOL_FAILED`（可重试），
每一步带一个稳定的幂等键 `(run_id, step_id, 参数指纹)`——重跑不会产生第二次副作用。

## 已经完成的步，一次都不能重跑

不是为了省时间，是为了**省钱**：W5 的检索步一次几毛到几块，一个 20 步的调研任务
重跑前 15 步，账单会翻倍，而报告里只会显示"重试了一次"。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .model import Attempt, Run, RunState, Step, utcnow

__all__ = [
    "JsonRunStore",
    "ResumableRunner",
    "ResumeReport",
    "StepOutcome",
    "StepExecutor",
    "interrupted_steps",
    "step_idempotency_key",
]

# 重启时"结果未知"的状态。`WAITING_APPROVAL` **不在**这里：
# 审批是等人，人是会回来点按钮的，把它重跑等于绕过审批（W10 红线）。
_INTERRUPTED = frozenset({RunState.RUNNING, RunState.WAITING_TOOL})


def step_idempotency_key(run_id: str, step: Step, *, tool: str | None = None) -> str:
    """一步的幂等键。稳定、可复算——重启后算出来必须和重启前一样。"""

    from ..resilience.idempotency import digest_of

    return digest_of(
        {
            "run": run_id,
            "step": step.step_id,
            "goal": step.goal,
            "tool": tool if tool is not None else step.tool,
        }
    )


def interrupted_steps(run: Run) -> list[Step]:
    """重启时处于"结果未知"的步骤。"""

    return [step for step in run.steps if step.state in _INTERRUPTED]


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """一次步骤执行的结果。`error_code` 沿用 `error_codes` 那份码表。"""

    ok: bool
    error_code: str | None = None
    latency_ms: float | None = None
    detail: str = ""
    args_digest: str | None = None


class StepExecutor(Protocol):
    """执行一步。真实实现是"调工具/调模型"，测试里是脚本。"""

    async def __call__(self, run: Run, step: Step) -> StepOutcome: ...


@dataclass(slots=True)
class ResumeReport:
    """续跑报告。`skipped` 与 `executed` 必须分成两个元组——

    合成一个列表之后，"这次只跑了 2 步"到底是"续跑生效"还是"任务只剩 2 步"，
    在报告里看不出来。
    """

    run_id: str
    skipped: tuple[str, ...] = ()
    executed: tuple[str, ...] = ()
    interrupted: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    # 卡在等人审批的步骤。**必须单独一档**：它不是"跳过"，也不是"失败"——
    # 它是"现在什么都不该做，等一个人"。混进 skipped 会让报告看起来一切正常。
    blocked: tuple[str, ...] = ()

    @property
    def resumed(self) -> bool:
        return bool(self.skipped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "skipped": list(self.skipped),
            "executed": list(self.executed),
            "interrupted": list(self.interrupted),
            "failed": list(self.failed),
            "blocked": list(self.blocked),
        }


@dataclass
class JsonRunStore:
    """一个 run 一个 JSON 文件。够用、可读、可入 git 做回归样本。

    生产上应该是"事件追加到日志/DB"，而不是整份覆写：整份覆写在两个进程
    同时续跑同一个 run 时会互相盖掉对方的事件。这里保留整份写，
    但 `save()` 只允许**单向增长**（事件数不能变少），越界直接抛。
    """

    root: Path
    clock: Callable[[], datetime] = utcnow

    def path_of(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def save(self, run: Run) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path_of(run.run_id)

        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            if len(previous.get("events") or ()) > len(run.events):
                raise ValueError(
                    f"run {run.run_id} 的事件数从 {len(previous['events'])} 掉到 "
                    f"{len(run.events)}；事件流只允许增长"
                )

        target.write_text(
            json.dumps(run.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return target

    def load(self, run_id: str) -> Run:
        return Run.from_dict(json.loads(self.path_of(run_id).read_text(encoding="utf-8")))

    def exists(self, run_id: str) -> bool:
        return self.path_of(run_id).exists()


@dataclass
class ResumableRunner:
    """按步执行，随时可停、随时可续。"""

    executor: StepExecutor
    store: JsonRunStore | None = None
    clock: Callable[[], datetime] = utcnow
    _attempts: dict[str, int] = field(default_factory=dict, repr=False)

    async def run(self, run: Run) -> ResumeReport:
        now = self.clock()

        # 1. 先把"结果未知"的步骤标成可重试。这一步必须在执行之前做完，
        #    否则第一个 RUNNING 步骤会卡在非法迁移上（RUNNING 不能直接回 RUNNING）。
        interrupted = [step.step_id for step in interrupted_steps(run)]
        if interrupted and run.state not in {RunState.CREATED, RunState.PLANNING}:
            for step_id in interrupted:
                run.mark_step(
                    step_id,
                    RunState.TOOL_FAILED,
                    at=now,
                    reason="进程重启：上一次调用结果未知，需重跑（副作用由幂等键兜）",
                )

        if run.state in {RunState.CREATED, RunState.PLANNING}:
            run.start(at=now)

        report = ResumeReport(run_id=run.run_id, interrupted=tuple(interrupted))

        for step in run.steps:
            # 每一步都重新读一次时钟。用同一个 `now` 走到底的话，时间轴上
            # 所有"开始"都挤在 00:00，看不出哪一步在等外部、等了多久。
            now = self.clock()

            if step.done:
                # 2. 已完成的步：只留一条事件，绝不重跑。
                run.note(
                    "resumed_skip",
                    at=now,
                    step_id=step.step_id,
                    data={"reason": "已完成步不重跑（成本）"},
                )
                report.skipped += (step.step_id,)
                continue

            if step.state is RunState.WAITING_APPROVAL:
                # 等审批的步骤**绝不自动执行**（W10 红线）。也不能跳过去跑后面的步，
                # 因为在依赖关系上它是前置。
                run.note(
                    "blocked_on_approval",
                    at=now,
                    step_id=step.step_id,
                    data={"reason": "等待人工审批，续跑不越过审批"},
                )
                report.blocked += (step.step_id,)
                break

            outcome = await self._execute(run, step, now)
            report.executed += (step.step_id,)

            if not outcome.ok:
                report.failed += (step.step_id,)
                # 一步失败即停：后面的步骤依赖它，继续跑只会把错误放大。
                break

            self._save(run)

        self._save(run)
        run.reconcile(at=now)
        self._save(run)
        return report

    # -- 内部 --------------------------------------------------------------- #

    async def _execute(self, run: Run, step: Step, now: datetime) -> StepOutcome:
        # 从"任何非 RUNNING"进 RUNNING：CREATED（首次）/ TOOL_FAILED（重启恢复）/
        # FAILED（续跑重试）。这三种情况在状态机上都是合法边。
        if step.state is not RunState.RUNNING:
            run.mark_step(step.step_id, RunState.RUNNING, at=now, reason="start")

        if step.tool:
            # 有工具就先进 waiting_tool：这个状态存在的唯一意义是
            # "监控能看出这一步在等外部，而不是在我们自己的代码里空转"。
            run.mark_step(step.step_id, RunState.WAITING_TOOL, at=now, reason=step.tool)

        started = self.clock()
        try:
            outcome = await self.executor(run, step)
        except Exception as exc:
            # 执行器自己炸了也必须落一条 Attempt，否则 Timeline 上这一步
            # 会显示"跑了，但没有记录"，比失败更难查。
            finished = self.clock()
            self._mark_tool_returned(run, step, finished)
            run.record_attempt(
                step.step_id,
                started_at=started,
                finished_at=finished,
                error_code=None,
                latency_ms=(finished - started).total_seconds() * 1000,
                detail=f"执行器异常：{type(exc).__name__}: {exc}",
            )
            run.mark_step(
                step.step_id, RunState.FAILED, at=finished, reason=f"{type(exc).__name__}"
            )
            raise

        finished = self.clock()
        self._mark_tool_returned(run, step, finished)
        latency = outcome.latency_ms
        if latency is None:
            latency = (finished - started).total_seconds() * 1000

        run.record_attempt(
            step.step_id,
            started_at=started,
            finished_at=finished,
            args_digest=outcome.args_digest,
            error_code=outcome.error_code,
            latency_ms=latency,
            detail=outcome.detail,
        )

        if outcome.ok:
            run.mark_step(step.step_id, RunState.COMPLETED, at=finished, reason="done")
        else:
            run.mark_step(
                step.step_id,
                RunState.FAILED,
                at=finished,
                error_code=outcome.error_code,
                reason=outcome.detail,
            )
        return outcome

    @staticmethod
    def _mark_tool_returned(run: Run, step: Step, at: datetime) -> None:
        """工具返回 → 回到 RUNNING，再落终态。

        少了这一步，`waiting_tool → completed` 就是一条不存在的边；
        更重要的是，Timeline 上会看不出"这一步在等外部等了多久"。
        """

        if step.state is RunState.WAITING_TOOL:
            run.mark_step(step.step_id, RunState.RUNNING, at=at, reason="tool returned")

    def _save(self, run: Run) -> None:
        if self.store is not None:
            self.store.save(run)


def attempts_of(run: Run) -> Mapping[str, tuple[Attempt, ...]]:
    """每步的尝试序列（报告里按步列"试了几次"）。"""

    return {step.step_id: tuple(step.attempts) for step in run.steps}

#!/usr/bin/env python
"""断点续跑 + Run Timeline 演练（W11 D3）。

    python evals/scripts/resume_drill.py

做三件事，每件都打印可核对的数字：

1. 跑一个 4 步任务，第 3 步注入 `UPSTREAM_TIMEOUT` → 任务失败；
2. **丢掉内存态**（只从 JSON 文件读回），续跑 → 前 2 步不重跑，从第 3 步接着跑；
3. 把续跑后的 run 渲染成 Timeline，并验证"只凭 Timeline 能讲清全过程"。

时钟是假的（每问一次走 1 秒），所以输出可复现——真 sleep 会让这个脚本没法进门禁。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework.runtime.model import Run, RunState  # noqa: E402
from agent_framework.runtime.resume import (  # noqa: E402
    JsonRunStore,
    ResumableRunner,
    StepOutcome,
)
from agent_framework.runtime.timeline import RunTimeline  # noqa: E402

RUN_ID = "drill-run"
T0 = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)


class TickClock:
    """每问一次走 1 秒。固定步长 = 输出可复现。"""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        current = self.now
        self.now += timedelta(seconds=1)
        return current


class ScriptedExecutor:
    def __init__(self, fail_once: dict[str, str] | None = None) -> None:
        self._fail_once = dict(fail_once or {})
        self.calls: list[str] = []

    async def __call__(self, run: Run, step) -> StepOutcome:  # noqa: ANN001
        self.calls.append(step.step_id)
        code = self._fail_once.pop(step.step_id, None)
        if code is not None:
            return StepOutcome(ok=False, error_code=code, detail="注入超时", args_digest="args-v1")
        return StepOutcome(ok=True, latency_ms=100.0, args_digest="args-v1")


def build_run() -> Run:
    # 显式给 created_at / at：混用 `utcnow()` 与本脚本的假时钟，
    # 时间轴会出现"事件比创建还早"，那是脚本的 bug，不是模型的。
    run = Run(run_id=RUN_ID, goal="调研 MCP 协议在 2026 年的落地情况", created_at=T0)
    run.add_step("step-1", "检索近 30 天的资料", tool="search_web", at=T0)
    run.add_step("step-2", "抓取 3 篇一手文档", tool="fetch_page", at=T0)
    run.add_step("step-3", "抽取事实与引用", tool="extract_facts", at=T0)
    run.add_step("step-4", "成文并自查引用密度", tool="compose", at=T0)
    return run


async def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="w11-d3-"))
    try:
        store = JsonRunStore(root=root)

        print("== 1. 第一趟：第 3 步注入 UPSTREAM_TIMEOUT ==\n")
        run = build_run()
        first = ScriptedExecutor(fail_once={"step-3": "UPSTREAM_TIMEOUT"})
        report1 = await ResumableRunner(first, store, TickClock()).run(run)

        print(f"executed = {report1.executed}")
        print(f"failed   = {report1.failed}")
        print(f"run.state= {run.state.value}")
        print(f"executor 实际调用: {first.calls}")
        print(f"落盘文件: {store.path_of(RUN_ID)}（{store.path_of(RUN_ID).stat().st_size} 字节）")

        print("\n== 2. '进程重启'：只从磁盘读回，内存态全丢 ==\n")
        reloaded = store.load(RUN_ID)
        print(f"读回后 run.state = {reloaded.state.value}；事件 {len(reloaded.events)} 条")

        second = ScriptedExecutor()
        # 续跑的时钟从"最后一个事件之后"开始，时间轴才不会倒退
        resume_clock = TickClock(start=reloaded.events[-1].at + timedelta(seconds=1))
        report2 = await ResumableRunner(second, store, resume_clock).run(reloaded)

        print(f"skipped  = {report2.skipped}   ← 已完成步，一次都没重跑")
        print(f"executed = {report2.executed}")
        print(f"blocked  = {report2.blocked}")
        print(f"executor 实际调用: {second.calls}")
        print(f"续跑后 run.state = {reloaded.state.value}")

        print("\n== 3. Timeline（续跑后的完整历史）==\n")
        print(RunTimeline().render(reloaded))

        print("\n== 4. 断言 ==\n")
        checks = [
            ("第一趟停在第 3 步", report1.failed == ("step-3",)),
            ("第一趟只跑了 3 步", first.calls == ["step-1", "step-2", "step-3"]),
            ("续跑跳过前 2 步", report2.skipped == ("step-1", "step-2")),
            ("续跑只跑后 2 步", second.calls == ["step-3", "step-4"]),
            ("已完成步没有重跑", "step-1" not in second.calls and "step-2" not in second.calls),
            ("最终状态 completed", reloaded.state is RunState.COMPLETED),
        ]
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not all(ok for _, ok in checks):
            return 1

        print("\n结论：断点续跑生效，成本不重复支付；Timeline 能独立解释全过程。")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(__import__("asyncio").run(main()))

"""Run Timeline（W11 D3）：把一次执行压成一张能看的表。

## 判据

拿任何一条失败 trace，**不打开日志文件**、只看 Timeline 就能说出
"哪一步、什么错、重试了几次、现在卡在哪"。做不到就说明状态没收敛，还是四套并存。

## 为什么从 `run.events` 渲染，而不是从 `run.steps`

steps 是**当前值**，events 是**历史**。从 steps 渲染的话，"第 2 步第一次超时、
第二次成功"只能看到最后一次尝试的痕迹，而"为什么重试"恰好藏在第一次里面。
从 events 渲染还有个副作用好处：它能顺带证明事件流本身是够用的——
如果某个信息在 events 里找不到，说明事件记漏了，而不是渲染写错了。

## 时间轴从第一个 event 起算

不用 `run.created_at`：创建与真正开始之间可能隔着几分钟排队，
从创建起算会让"第 1 步用了 4 分钟"这种结论在报告里凭空出现。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..error_codes import is_caller_code
from .model import Attempt, Event, Run

__all__ = ["Line", "RunTimeline"]

# 每一列的宽度。写成常量而不是 f-string 里现算，
# 是为了让"加一列"这件事只改一个地方（表头与数据行共用同一组宽度）。
_WIDTHS: tuple[int, ...] = (10, 10, 4, 24, 14, 9)
_HEADER = ("t", "step", "try", "action", "status", "latency")


def _join(cells: tuple[str, ...]) -> str:
    return "  ".join(f"{cell:<{width}}" for cell, width in zip(cells, _WIDTHS, strict=True))


@dataclass(frozen=True, slots=True)
class Line:
    """一行 = 一次尝试，或一次状态变化（没有尝试的那种）。"""

    offset_ms: int
    step_id: str
    attempt: int | None
    action: str
    status: str
    latency_ms: float | None = None
    error_code: str | None = None
    note: str = ""

    def render(self) -> str:
        minutes, millis = divmod(self.offset_ms, 60_000)
        seconds, ms = divmod(millis, 1_000)
        clock = f"{minutes:02d}:{seconds:02d}.{ms // 100:01d}"

        step = self.step_id or "-"
        attempt = f"#{self.attempt}" if self.attempt is not None else "-"
        latency = f"{self.latency_ms:.0f}ms" if self.latency_ms is not None else "-"

        line = _join((clock, step, attempt, self.action, self.status, latency))
        if self.error_code:
            line += f"  [{self.error_code}]"
        if self.note:
            line += f"  ({self.note})"
        return line.rstrip()


class RunTimeline:
    """无状态渲染器。构造参数只有"要不要显示毫秒偏移"这类纯展示选项。"""

    def __init__(self, *, include_events: bool = False) -> None:
        # 默认只画"尝试"与"关键状态"，不画每一次 step_state：
        # 全画出来之后，一个 8 步任务有 40 行，而人只关心失败与等待。
        self._include_events = include_events

    def rows(self, run: Run) -> list[Line]:
        events = run.events
        if not events:
            return []

        # 起点取**最早**的事件而不是第 1 条：事件按追加顺序排列，而 `at` 可能
        # 因为"续跑时注入了一个更早的假时钟"出现回退。取最小值只会让
        # 前几行的时间偏大，取第一条会让整张表的偏移全错。
        origin: datetime = min(event.at for event in events)
        rows: list[Line] = []

        for event in events:
            if event.kind == "attempt":
                rows.append(self._attempt_line(run, event, origin))
            elif event.kind in {"step_added", "run_state", "reconciled"}:
                if self._include_events:
                    rows.append(self._event_line(event, origin))
            elif event.kind == "step_state":
                line = self._state_line(run, event, origin)
                if line is not None:
                    rows.append(line)
            elif event.kind in {"resumed_skip", "blocked_on_approval"}:
                rows.append(self._skip_line(run, event, origin))
        return rows

    def render(self, run: Run) -> str:
        rows = self.rows(run)
        header = _join(_HEADER).rstrip()
        lines = [header, "-" * len(header)]
        lines.extend(row.render() for row in rows)
        lines.append("")
        lines.append(self.summary(run))
        return "\n".join(lines)

    def summary(self, run: Run) -> str:
        """一句话结论：卡在哪 / 为什么结束。"""

        attempts = sum(len(step.attempts) for step in run.steps)
        retries = sum(1 for step in run.steps if len(step.attempts) > 1)

        if run.state.value == "waiting_approval":
            waiting = [s.step_id for s in run.steps if s.state.value == "waiting_approval"]
            return f"卡在等人审批：{', '.join(waiting) or '（未定位到步骤）'}"
        if run.state.value == "waiting_tool":
            waiting = [s.step_id for s in run.steps if s.state.value == "waiting_tool"]
            return f"卡在等工具返回：{', '.join(waiting) or '（未定位到步骤）'}"

        unfixed: list[str] = []
        for step in run.steps:
            last = step.last_attempt()
            if step.state.value in {"failed", "tool_failed"} and last is not None:
                unfixed.append(f"{step.step_id}({last.error_code or '无码'})")
        if unfixed:
            detail = "; ".join(unfixed)
            return (
                f"{run.state.value}：失败步骤 {detail}；"
                f"共 {attempts} 次尝试，{retries} 步重试过"
            )
        return f"{run.state.value}：{len(run.steps)} 步 / {attempts} 次尝试 / {retries} 步重试"

    # -- 行构造 ------------------------------------------------------------- #

    def _attempt_line(self, run: Run, event: Event, origin: datetime) -> Line:
        step_id = event.step_id or "-"
        attempt = self._attempt_of(run, step_id, event.attempt)
        error_code = (attempt.error_code if attempt else None) or event.data.get("error_code")

        return Line(
            offset_ms=self._offset(event.at, origin),
            step_id=step_id,
            attempt=event.attempt,
            action=(attempt.tool if attempt and attempt.tool else "-"),
            status="ok" if not error_code else "failed",
            latency_ms=attempt.latency_ms if attempt else None,
            error_code=error_code,
            note=self._retry_note(run, step_id, event.attempt, error_code),
        )

    def _state_line(self, run: Run, event: Event, origin: datetime) -> Line | None:
        """状态行只为"需要人注意"的状态画一条：等待、失败、终态。"""

        if event.state is None or event.step_id is None:
            return None
        if event.state.value not in {
            "waiting_tool",
            "waiting_approval",
            "tool_failed",
            "failed",
            "cancelled",
        }:
            return None

        step = run.find_step(event.step_id)
        return Line(
            offset_ms=self._offset(event.at, origin),
            step_id=event.step_id,
            attempt=None,
            action=step.tool if step and step.tool else "-",
            status=event.state.value,
            note=str(event.data.get("reason") or ""),
        )

    def _skip_line(self, run: Run, event: Event, origin: datetime) -> Line:
        """续跑跳过 / 等审批。这两行是"钱花在哪儿"的唯一线索，必须画出来。"""

        step = run.find_step(event.step_id or "")
        status = "skipped" if event.kind == "resumed_skip" else "blocked"
        note = (
            str(event.data.get("reason") or "")
            if event.kind == "blocked_on_approval"
            else "已完成，未重跑"
        )
        return Line(
            offset_ms=self._offset(event.at, origin),
            step_id=event.step_id or "-",
            attempt=None,
            action=step.tool if step and step.tool else "-",
            status=status,
            note=note,
        )

    def _event_line(self, event: Event, origin: datetime) -> Line:
        return Line(
            offset_ms=self._offset(event.at, origin),
            step_id=event.step_id or "-",
            attempt=event.attempt,
            action=event.kind,
            status=event.state.value if event.state else "-",
        )

    # -- 小工具 ------------------------------------------------------------- #

    @staticmethod
    def _offset(at: datetime, origin: datetime) -> int:
        return max(0, int((at - origin).total_seconds() * 1000))

    @staticmethod
    def _attempt_of(run: Run, step_id: str, n: int | None) -> Attempt | None:
        if n is None:
            return None
        step = run.find_step(step_id)
        if step is None:
            return None
        for attempt in step.attempts:
            if attempt.n == n:
                return attempt
        return None

    @staticmethod
    def _retry_note(run: Run, step_id: str, n: int | None, error_code: Any) -> str:
        """回答"为什么重试"。

        参数指纹相同 → "重试，参数未变"（说明重试是被允许的，不是改了才跑通）；
        不同 → "改了参数再试"。这两个结论在复盘时指向完全不同的修法。
        """

        if n is None or n <= 1:
            return ""

        step = run.find_step(step_id)
        if step is None:
            return ""

        current = next((a for a in step.attempts if a.n == n), None)
        previous = next((a for a in step.attempts if a.n == n - 1), None)
        if current is None or previous is None:
            return ""

        same = current.args_digest == previous.args_digest
        reason = "重试" if error_code else "重试后成功"
        return f"{reason}，参数{'未变' if same else '已改'}"

    def to_dict(self, run: Run) -> dict[str, Any]:
        """给报告/接口用的结构化版本（`render()` 是给人看的）。"""

        return {
            "run_id": run.run_id,
            "state": run.state.value,
            "summary": self.summary(run),
            "rows": [
                {
                    "offset_ms": row.offset_ms,
                    "step_id": row.step_id,
                    "attempt": row.attempt,
                    "action": row.action,
                    "status": row.status,
                    "latency_ms": row.latency_ms,
                    "error_code": row.error_code,
                    "note": row.note,
                }
                for row in self.rows(run)
            ],
        }


def is_callers_fault(error_code: str | None) -> bool:
    """便利函数：这次失败是不是"我们自己传错了"。Timeline 报告里要分开统计。"""

    return is_caller_code(error_code)

"""统一执行模型（W11 D3）。

四个对象 + 一张九态状态机 + 时间线 + 断点续跑。设计动机见 `model.py` 的模块文档，
这里只说明**边界**：

- 本包**不执行**任何东西。执行归 `planner/`（单轮）与 `workflow/`（长任务）。
- 本包**不持久化**。`resume.JsonRunStore` 只是给演练与测试一个可读的落点，
  生产上的持久化在队列/工作流那一侧。
- 本包**不新造错误码**。`Attempt.error_code` 用 `agent_framework.error_codes` 那份。
"""

from .model import (
    LEGACY_MAPPINGS,
    Attempt,
    Event,
    IllegalTransition,
    Run,
    RunState,
    Step,
    allowed_from_run,
    allowed_from_step,
    is_terminal,
    state_breakdown,
    state_from_legacy,
    stuck_runs,
)
from .resume import (
    JsonRunStore,
    ResumableRunner,
    ResumeReport,
    StepExecutor,
    StepOutcome,
    step_idempotency_key,
)
from .timeline import Line, RunTimeline

__all__ = [
    "LEGACY_MAPPINGS",
    "Attempt",
    "Event",
    "IllegalTransition",
    "JsonRunStore",
    "Line",
    "ResumableRunner",
    "ResumeReport",
    "Run",
    "RunState",
    "RunTimeline",
    "Step",
    "StepExecutor",
    "StepOutcome",
    "allowed_from_run",
    "allowed_from_step",
    "is_terminal",
    "state_breakdown",
    "state_from_legacy",
    "stuck_runs",
    "step_idempotency_key",
]

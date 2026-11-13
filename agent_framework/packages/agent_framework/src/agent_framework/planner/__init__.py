"""规划与多智能体（W9）。

分层：

```
reasoners.py   大脑（脚本化 / 策略 / 真模型）—— 三种编排共用，保证对比只有一个变量
base.py        Plan/PlanStep 状态机 + Reasoner 协议 + StepRunner（唯一执行出口）
controls.py    三道上限 + 控制权矩阵（七个决定，模型只拿三个）
react.py       朴素 ReAct —— 对照组，刻意保留它的缺陷
plan_execute.py  主范式：计划 → 执行 → 失败重规划
research_planner.py  Agentic Search：缺口驱动的多轮检索
```

`multi_agent/` 在同一个包里以子模块形式存在（supervisor + 三角色 + handoff），
它复用的仍是这里的 `StepRunner` 与 `AutonomyBudget`——**多 Agent 不是另一套执行，
只是另一种"谁来决定下一步"**。
"""

from __future__ import annotations

from .base import (
    DECISION_FINISH,
    DECISION_TOOL,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_PENDING,
    AgentState,
    Decision,
    Plan,
    Planner,
    PlannerBase,
    PlanStep,
    Reasoner,
    RunTrace,
    StepRecord,
    StepRunner,
    StepStateError,
    StepStatus,
    TaskResult,
    args_digest,
    plan_fingerprint,
)
from .controls import (
    CONTROL_MATRIX,
    OWNER_CODE,
    OWNER_HUMAN,
    OWNER_MODEL,
    AutonomyBudget,
    BudgetMeter,
    ControlMatrix,
    ControlViolation,
    StopReason,
    matrix,
)
from .plan_execute import PlanExecutePlanner
from .react import ReActPlanner
from .reasoners import LLMReasoner, PolicyReasoner, PolicyTask, ScriptedReasoner, TaskPolicy

__all__ = [
    "CONTROL_MATRIX",
    "DECISION_FINISH",
    "DECISION_TOOL",
    "OWNER_CODE",
    "OWNER_HUMAN",
    "OWNER_MODEL",
    "STATUS_ABORTED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_PARTIAL",
    "STATUS_PENDING",
    "AgentState",
    "AutonomyBudget",
    "BudgetMeter",
    "ControlMatrix",
    "ControlViolation",
    "Decision",
    "LLMReasoner",
    "Plan",
    "PlanExecutePlanner",
    "PlanStep",
    "Planner",
    "PlannerBase",
    "PolicyReasoner",
    "PolicyTask",
    "ReActPlanner",
    "Reasoner",
    "RunTrace",
    "ScriptedReasoner",
    "StepRecord",
    "StepRunner",
    "StepStateError",
    "StepStatus",
    "StopReason",
    "TaskPolicy",
    "TaskResult",
    "args_digest",
    "matrix",
    "plan_fingerprint",
]

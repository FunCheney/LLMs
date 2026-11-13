"""多智能体编排（W9 D3）。

四个文件各管一件事，边界刻意画得很硬：

| 文件 | 只负责 | 不该有的 |
|---|---|---|
| `shared_state` | 共享工作区 + 版本号 | 任何调度判断 |
| `handoff` | 交接消息 + 环路指纹 | 谁该被调度的逻辑 |
| `roles` | 三种角色的单一职责 | 直接调工具、直接写状态 |
| `supervisor` | 调度 + 记账 + 三道守卫 | 检索、写作、任何业务逻辑 |

**为什么不用现成的多 Agent 框架。** 见 `docs/w9/langgraph-vs-inhouse.md`：
D4 会用 LangGraph 写一份对照实现，量的是同一件事在两种框架下"守卫写在哪、
默认值给谁"。这里的实现把三道守卫放在编排层的最外层，是因为它们防的三类
失败（膨胀 / 预算重置 / 互踢皮球）都属于**编排的错**，不是某个角色的错。
"""

from __future__ import annotations

from .handoff import Handoff, HandoffLedger, LoopDetected
from .roles import (
    ROLE_RESEARCHER,
    ROLE_SUPERVISOR,
    ROLE_VALIDATOR,
    AgentRole,
    ResearcherRole,
    RoleAction,
    RoleResult,
    SupervisorRole,
    ValidatorRole,
)
from .shared_state import Finding, SharedState, StaleStateError
from .supervisor import Supervisor, SupervisorConfig

__all__ = [
    "ROLE_RESEARCHER",
    "ROLE_SUPERVISOR",
    "ROLE_VALIDATOR",
    "AgentRole",
    "Finding",
    "Handoff",
    "HandoffLedger",
    "LoopDetected",
    "ResearcherRole",
    "RoleAction",
    "RoleResult",
    "SharedState",
    "StaleStateError",
    "Supervisor",
    "SupervisorConfig",
    "SupervisorRole",
    "ValidatorRole",
]

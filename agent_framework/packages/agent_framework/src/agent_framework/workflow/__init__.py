"""长任务编排包（W2 只放接口缝，W9 接 Temporal）。

`workflow/` 存在的意义是**把边界写进代码**：`session/` 只存取、`llm/` 只调模型、
`workflow/` 只编排。三者互不 import，跨层只能通过数据契约（`ChatCall` / `Message`）。
"""

from __future__ import annotations

from .research_workflow import (
    ResearchActivities,
    ResearchPlan,
    ResearchReport,
    ResearchRequest,
    ResearchWorkflow,
)

__all__ = [
    "ResearchActivities",
    "ResearchPlan",
    "ResearchReport",
    "ResearchRequest",
    "ResearchWorkflow",
]

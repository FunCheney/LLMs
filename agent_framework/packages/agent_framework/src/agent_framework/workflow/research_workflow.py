"""长任务编排的**接口缝**（W2 D1 只定签名，W9 才实现）。

为什么现在写这个文件
----------------
`docs/w2/adr-w2-session-boundary.md` 的核心结论是"`/v1/chat` 直连，长调研走
workflow"。如果这条结论只写在文档里，W9 动手时一定会有人把 `SessionStore` 和
`LLMGateway` 直接焊进 workflow，或者反过来把 `/v1/chat` 塞进 workflow 换"能续跑"。
先把 activity 的**签名**钉住，边界就有了可执行的形状。

为什么这里没有 `import temporalio`
--------------------------------
1. 本周不装 Temporal（`w1-tasks.md` T1.2 记着"W2 再加回"，但 D1 只要求接口缝）；
2. 库代码一旦硬依赖编排引擎，所有离线用例都要拉起一个 worker 才能跑；
3. 真正的编排语义（超时、重试策略、补偿）应该在 W9 写 workflow 时**显式声明**，
   而不是靠这里偷偷替调用方决定。

所以这里只有：数据契约 + 三个 activity 的 `Protocol` + 一个占位 workflow 的函数签名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "ResearchActivities",
    "ResearchPlan",
    "ResearchReport",
    "ResearchRequest",
    "ResearchWorkflow",
]


@dataclass(frozen=True, slots=True)
class ResearchRequest:
    """一次调研请求的输入。

    `request_id` 必带：它就是 workflow id 的来源。没有它，重投递会启动第二个
    workflow，同一个调研跑两遍、花两份钱——这类事故在 W9 之前不会暴露。
    """

    request_id: str
    question: str
    max_sources: int = 5
    language: str = "zh"


@dataclass(frozen=True, slots=True)
class ResearchPlan:
    """规划的产物：待查的子问题列表。"""

    sub_questions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """汇总产物。`citations` 与 `sub_questions` 一一对应，便于溯源。"""

    summary: str
    citations: list[str] = field(default_factory=list)
    failed_sub_questions: list[str] = field(default_factory=list)


class ResearchActivities(Protocol):
    """workflow 可调用的 activity。

    三条纪律（W9 实现时必须遵守）：

    1. **每个 activity 必须幂等**：Temporal 的是"至少一次"投递，activity 会被重放。
       写库/发消息的动作必须由调用方带幂等键（沿用 `SessionStore.append` 的做法）。
    2. **activity 里不许出现重试循环**：重试策略由 workflow 声明。activity 内部再
       写一层 `for attempt in range(3)`，实际 attempts 会变成 3×配置值，成本和
       日志都对不上（与 ADR-003「重试只在 client」是同一条原则的延伸）。
    3. **返回值必须可序列化**：只用 dataclass + 基础类型，不要塞 `LLMResponse`。
       W1 的归一化结果已经是内部 `Message`，跨进程边界前先投影成纯数据。
    """

    async def plan(self, request: ResearchRequest) -> ResearchPlan:
        """把问题拆成子问题。通常是一次 cheap 档 LLM 调用。"""
        ...

    async def gather(self, sub_question: str, *, top_k: int) -> list[str]:
        """检索单条子问题的证据（W6 的 RAG 从这里接进来）。"""
        ...

    async def synthesize(
        self, request: ResearchRequest, plan: ResearchPlan, evidence: list[str]
    ) -> ResearchReport:
        """汇总成带引用的报告。通常是一次 strong 档 LLM 调用。"""
        ...


class ResearchWorkflow(Protocol):
    """占位 workflow：签名定稿，实现留到 W9。

    为什么用 `Protocol` 而不是继承 Temporal 的 `@workflow.defn`：
    本周没有任何代码需要 import 它，写成 Protocol 才能让
    `run_smoke` / `pytest` 在**没有 Temporal** 的机器上照常跑绿。
    """

    async def run(self, request: ResearchRequest) -> ResearchReport:
        """plan → gather（并发，受 `limits` 约束）→ synthesize。

        W9 的实现要点（先记在这里，避免那时候重新讨论）：
        - `gather` 并发度不得绕过 `resilience.ConcurrencyLimiter`，否则调研会把
          线上 `/v1/chat` 的配额挤干；
        - 每个阶段一个 `span()`，让一条 trace 能穿起"规划/检索/汇总"三次调用；
        - 失败的子问题进 `failed_sub_questions` 而不是整单失败：调研的价值在于
          "拿到 4/5 的答案"，全有全无是最差的产品决策。
        """
        raise NotImplementedError("W9 实现：本周只钉签名（见 docs/w2/w2-tasks.md D1）")

"""安全护栏四层（W10 D3/D4）：输入 → 执行（带 HITL）→ 输出 → 行为监控。

## 为什么是"四层"而不是"一个大检查"

每层管的**对象**不同，把它们合成一个函数之后，失败原因就只剩一句话：

| 层 | 管什么 | 失败语义 | 典型漏检 |
|---|---|---|---|
| 输入 | 进来的是不是一段可以处理的文本 | 400 / 422 | 只查用户输入、漏了检索块 |
| 执行 | 这个动作要不要人批 | 202 + 待审 id | 未审批静默执行 |
| 输出 | 产物能不能交付 | 422 | 只查长度、漏了悬空编号 |
| 行为 | 频率是否异常 | 告警（不拦断） | 单次全合规、合起来出事 |

最后一列是重点：**每一层都有它唯一能看见的东西**。这也是为什么
"四层"不是一个组织代码的方式，而是四个不同的观测点。

## 编排顺序与归属（ADR-008 的复现）

顺序固定为 输入 → 执行 → 输出 → 行为，理由：

1. 输入层最早，因为它最便宜（正则 + 长度判断），能在花钱之前拦掉明显恶意；
2. 执行层必须在输出层之前，因为"能不能做"比"做得怎么样"优先级更高——
   反过来的话，一次未审批的发布也会先生成完整产物再被卡住，白花钱；
3. 行为层在最后且**不拦断**：频率异常往往发生在批量场景，拦断会误伤，
   它的消费者是告警与熔断器（W11 D2），不是 HTTP 状态码。

这四条链路的**每一条都必须在所有路径上生效**——所以顺序与归属收在
`pipeline.py` 的 `GuardrailPipeline` 一个入口里，调用方只能通过 `run()`
拿结果。ADR-008 说的"横切关注点需要一个所有路径都必经的归属点"，
在这里就是那个 `run()`：只要新链路走它，就不可能只过两层；
不走它，就等于在 code review 里声明"这条路我不设防"。
"""

from __future__ import annotations

from .behavior import (
    BEHAVIOR_RULES,
    EVENT_APPROVAL_PENDING,
    EVENT_BUDGET_SPENT,
    EVENT_DANGEROUS_CALL,
    EVENT_STEP,
    EVENT_TOOL_CALL,
    BehaviorAlert,
    BehaviorMonitor,
    BehaviorRule,
    Event,
)
from .hitl import (
    APPROVED,
    EXPIRED,
    PENDING,
    REJECTED,
    ApprovalAuditEntry,
    ApprovalError,
    ApprovalGate,
    ApprovalRequest,
    ApprovalRequired,
    arguments_digest,
    skill_of,
)
from .injection import (
    ALLOWED_SOURCES,
    LAYER_PREAMBLE,
    RULE_NAMES,
    SOURCE_MEMORY,
    SOURCE_RETRIEVAL,
    SOURCE_TOOL,
    SOURCE_USER,
    GuardVerdict,
    InjectionGuard,
    InjectionHit,
    isolate,
    layer_prompt,
)
from .input import BlockedInput, InputCheck, InputGuard, InputLimits
from .output import (
    OutputCheck,
    OutputGuard,
    OutputPolicy,
    OutputRejected,
    citation_markers,
    dangling_markers,
    split_sentences,
    unpublished_claims,
)
from .pipeline import (
    STAGE_APPROVAL,
    STAGE_BEHAVIOR,
    STAGE_EXECUTE,
    STAGE_INPUT,
    STAGE_OUTPUT,
    GuardedRun,
    GuardrailPipeline,
    GuardStep,
)

__all__ = [
    "ALLOWED_SOURCES",
    "APPROVED",
    "BEHAVIOR_RULES",
    "BlockedInput",
    "EVENT_APPROVAL_PENDING",
    "EVENT_BUDGET_SPENT",
    "EVENT_DANGEROUS_CALL",
    "EVENT_STEP",
    "EVENT_TOOL_CALL",
    "EXPIRED",
    "GuardVerdict",
    "GuardedRun",
    "GuardrailPipeline",
    "GuardStep",
    "InjectionGuard",
    "InjectionHit",
    "InputCheck",
    "InputGuard",
    "InputLimits",
    "LAYER_PREAMBLE",
    "OutputCheck",
    "OutputGuard",
    "OutputPolicy",
    "OutputRejected",
    "PENDING",
    "REJECTED",
    "RULE_NAMES",
    "SOURCE_MEMORY",
    "SOURCE_RETRIEVAL",
    "SOURCE_TOOL",
    "SOURCE_USER",
    "STAGE_APPROVAL",
    "STAGE_BEHAVIOR",
    "STAGE_EXECUTE",
    "STAGE_INPUT",
    "STAGE_OUTPUT",
    "ApprovalAuditEntry",
    "ApprovalError",
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalRequired",
    "BehaviorAlert",
    "BehaviorMonitor",
    "BehaviorRule",
    "Event",
    "arguments_digest",
    "citation_markers",
    "dangling_markers",
    "isolate",
    "layer_prompt",
    "skill_of",
    "split_sentences",
    "unpublished_claims",
]

"""上下文工程（W7）：预算 → 工作记忆 → 长期记忆 → 退化缓解的 Context 流水线。

- `schema.py`        ContextItem / ContextBudget / 六类排除原因（Debug Report 的数据源）
- `budget.py`        allocate()：槽位预算分配 + 确定性裁剪 + 排除原因
- `builder.py`       ContextBuilder：分配结果 → 标准 list[Message]（不改 gateway）
- `degradation.py`   长上下文退化缓解：U 形注意力排布（消息序列级）
- `drift.py`         Drift 三指标（目标漂移 / 证据丢失 / 重复劳动，纯规则）

与 W6 `retrieval/context_assemble` 的分工：那边管**检索块的局部预算**
（去冗余 → 裁剪 → 首尾放置，编号之前）；这边管**整条 payload 的全局预算**
（system/memory/history/tool 各占多少、超了先裁谁、为什么）。局部预算的
产物（`render_prompt` 的编号文本）作为 retrieved 槽位的内容进全局分配。

排布纪律：system 只收 trust=system 的内容；external 信任的内容无论多相关
都不能进 system 槽位（allocate 直接抛错）——W10 注入防护在这里提前立住。
"""

from .budget import DEFAULT_TOKENIZER, allocate, default_tokenizer
from .builder import MEMORY_BLOCK_HEADER, ContextBuilder, build_messages, render_memory_block
from .degradation import order_payload
from .drift import (
    GOAL_ECHO_THRESHOLD,
    SOURCE_LOSS_THRESHOLD,
    DriftReport,
    check_drift,
    terms,
)
from .schema import (
    AllocatedContext,
    ContextBudget,
    ContextItem,
    ContextKind,
    Exclusion,
    ExclusionReason,
    TrustLevel,
)

__all__ = [
    "AllocatedContext",
    "ContextBudget",
    "ContextBuilder",
    "ContextItem",
    "ContextKind",
    "DEFAULT_TOKENIZER",
    "DriftReport",
    "Exclusion",
    "ExclusionReason",
    "GOAL_ECHO_THRESHOLD",
    "MEMORY_BLOCK_HEADER",
    "SOURCE_LOSS_THRESHOLD",
    "TrustLevel",
    "allocate",
    "build_messages",
    "check_drift",
    "default_tokenizer",
    "order_payload",
    "render_memory_block",
    "terms",
]

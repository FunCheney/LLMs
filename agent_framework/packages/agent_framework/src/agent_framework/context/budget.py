"""token 预算分配器（W7 D1）：按槽位裁剪，裁剪顺序确定，排除原因可解释。

核心算法只有两条，都写死（不许配置"自定义裁剪策略"）：

1. **槽位内**：超预算时按 `(score, authority, id)` 从低到高裁。
   先裁分数低的；分数相同裁权威度低的；再相同按 id 确定顺序——
   确定性不是锦上添花，是 replay 复现的前提（W6 的 citation 抽样就是靠
   逐位复现排查出判据错误的）。
2. **槽位间**：不挪用。每个槽位独立裁到自己的上限，剩余额度**不**借给
   别的槽位。挪用会让"retrieved 没进第 6 条"的解释依赖当轮 memory 的
   占用——Debug Report 就不再是"配置问题"而是"运行时问题"，不可复算。

tokenizer 与 W6 `context_assemble.estimate_tokens` 同一份实现（ADR-W7-01：
字符粗估 1.6 chars/token，不引 tiktoken——它在模型间不通用，引进来只会
让人以为预算是精确的）。计数口径与真实发送的一致性由单测
`test_budget_accounting` 用同一 tokenizer 核对。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..retrieval.context_assemble import estimate_tokens
from .schema import (
    AllocatedContext,
    ContextBudget,
    ContextItem,
    ContextKind,
    Exclusion,
    ExclusionReason,
)

__all__ = ["DEFAULT_TOKENIZER", "allocate", "default_tokenizer"]

# 预算分配的"槽位预算表"：kind → ContextBudget 字段名。
# 用映射而不是 getattr(kind.name)：kind 与字段名一旦对不上，
# 某个槽位就会静默不设防——映射表至少让两者必须在同一处对齐。
_SLOT_FIELD: dict[ContextKind, str] = {
    ContextKind.SYSTEM: "system",
    ContextKind.RETRIEVED: "retrieved",
    ContextKind.MEMORY: "memory",
    ContextKind.HISTORY: "history",
    ContextKind.TOOL_RESULT: "tool_results",
}

# 计数口径的唯一定义处。allocate/builder/单测全部引用它，
# 不允许各自 `len(text) / 1.6`——两处口径必漂移（W7 任务清单 2 号返工）。
DEFAULT_TOKENIZER: Callable[[str], int] = estimate_tokens


def default_tokenizer(text: str) -> int:
    """包一层具名函数，mypy/IDE 能正确显示签名。"""
    return DEFAULT_TOKENIZER(text)


def allocate(
    items: Sequence[ContextItem],
    budget: ContextBudget,
    tokenizer: Callable[[str], int] = DEFAULT_TOKENIZER,
) -> AllocatedContext:
    """把候选分配进预算。返回入选项 + 每槽用量 + 全部排除原因。

    判定顺序（每道闸只处理它负责的原因，先拦先记）：

    1. 上游已出局（`excluded != None`）：照抄进报告，`decided_by=upstream`；
    2. 过期（`is_expired`）：`filtered_expired`；
    3. 信任级违规：system 槽位出现 external 信任的内容 → **拒绝整次分配**
       （抛 ValueError）。这是攻击面而不是预算问题，静默裁掉等于把注入
       尝试降级成了"预算不足"；
    4. 槽位内超预算：`budget_squeezed`（从低分往高裁）。
    """
    budget.validate()

    exclusions: list[Exclusion] = []
    candidates: list[ContextItem] = []

    for item in items:
        # 闸 1：上游已出局。不覆盖原因（纪律：一个候选只有一个记录者）。
        if item.excluded is not None:
            exclusions.append(
                Exclusion(
                    item_id=item.id,
                    kind=item.kind,
                    reason=item.excluded,
                    note=item.note,
                    decided_by="upstream",
                )
            )
            continue
        # 闸 2：生命周期。
        if item.is_expired():
            exclusions.append(
                Exclusion(
                    item_id=item.id,
                    kind=item.kind,
                    reason=ExclusionReason.FILTERED_EXPIRED,
                    note=item.source,
                )
            )
            continue
        # 闸 3：信任级（system 槽位拒 external，见 schema 顶部纪律 1）。
        if item.kind is ContextKind.SYSTEM and item.trust.value == "external":
            raise ValueError(
                f"信任级违规：external 内容 {item.id!r} 试图进入 system 槽位。"
                f"这是注入路径，不是预算问题——检查上游把网页/工具返回标成了 system"
            )
        candidates.append(item)

    # 按槽位分组（保输入顺序）。
    by_slot: dict[ContextKind, list[ContextItem]] = {kind: [] for kind in _SLOT_FIELD}
    for item in candidates:
        by_slot[item.kind].append(item)

    kept_ids: set[int] = set()
    used: dict[str, int] = {}
    for kind, slot_items in by_slot.items():
        limit = getattr(budget, _SLOT_FIELD[kind])
        # 预算内从高分往低放，放不下的整条出局。不"跳着塞小的"，
        # 理由与 W6 前缀裁剪相同：事后无法解释取舍。
        costs = {id(item): tokenizer(item.content) for item in slot_items}
        ordered = sorted(slot_items, key=lambda it: (-it.score, -it.authority, it.id))
        slot_used = 0
        for item in ordered:
            cost = costs[id(item)]
            if slot_used + cost > limit:
                exclusions.append(
                    Exclusion(
                        item_id=item.id,
                        kind=kind,
                        reason=ExclusionReason.BUDGET_SQUEEZED,
                        note=f"slot={_SLOT_FIELD[kind]} used={slot_used} cost={cost}",
                    )
                )
                continue
            kept_ids.add(id(item))
            slot_used += cost
        used[_SLOT_FIELD[kind]] = slot_used

    # 恢复输入顺序输出（builder 按 kind 取用，顺序确定性同样重要）。
    kept_order = [item for item in candidates if id(item) in kept_ids]
    return AllocatedContext(
        items=tuple(kept_order),
        used=used,
        exclusions=tuple(exclusions),
        budget=budget,
    )

"""长上下文退化缓解（W7 D4）：让"带记忆的长 payload"仍把关键信息放在
注意力高位。

与 W6 `order_for_attention` 的关系——不是替代，是分层：

- W6 只排**检索块**（编号之前），不动消息结构；
- 本模块排**整条消息序列**（编号之后），规则更粗但影响面更大。

排布原则（Liu et al. 2023, "Lost in the Middle" 的工程化结论）：

    system（头）→ 历史/工具（中段）→ 记忆 → 资料 → 问题（尾）

注意力呈 U 形：首尾利用率高、中段低。所以"必须被模型看到的信息"
（system、问题、高置信记忆、top 资料）占首尾；"上下文胶水"
（历史、工具结果）让位中段。这个排布已经实现在 `builder.build_messages`，
本模块提供 `order_payload` 作为**消息序列级**的通用重排入口，供
不经过 builder 的链路（W9 workflow）复用同一策略。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..llm.types import Message, MessageRole

__all__ = ["HEAD_ROLES", "TAIL_KINDS", "order_payload"]

# 必须占头部/尾部的角色。system 恒在头；最后一条 user（当前问题）恒在尾。
HEAD_ROLES = frozenset({MessageRole.SYSTEM})
TAIL_KINDS = frozenset({MessageRole.USER})


def order_payload(messages: Sequence[Message]) -> list[Message]:
    """把消息重排成 U 形注意力分布：system 头 + 最后一条 user 尾 + 其余居中。

    只动**顺序**，不动内容、不改 role——重排后消息语义必须与重排前
    可互换（tool 消息与其调用消息的相对位置保持不变：`Message` 里
    tool_call_id 的成组关系靠相邻位置，乱序会把工具结果配错调用）。

    实现：头 = system 前缀；尾 = 最后一条 user；中段 = 其余按原顺序。
    中段内部**不做**首尾交错（W6 对检索块做过 1,3,5,…,6,4,2 交错）：
    历史消息有严格的时序语义，交错会把"第 3 轮答在先、第 2 轮问在后"，
    对话直接读不懂。检索块没有时序语义，所以可以交错——这是两者
    只能分层实现、不能合并的根本原因。
    """
    if len(messages) <= 2:
        return list(messages)

    head: list[Message] = []
    for message in messages:
        if message.role in HEAD_ROLES:
            head.append(message)

    # 尾部：最后一条 user 消息（当前问题）。从后往前找第一条即可。
    tail: Message | None = None
    for message in reversed(messages):
        if message.role in TAIL_KINDS:
            tail = message
            break

    used_head = {id(m) for m in head}
    tail_id = id(tail) if tail is not None else None
    middle = [
        message
        for message in messages
        if id(message) not in used_head and id(message) != tail_id
    ]

    return head + middle + ([tail] if tail is not None else [])

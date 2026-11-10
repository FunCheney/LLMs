"""Drift 三指标（W7 D4）：目标漂移 / 证据丢失 / 重复劳动。

抓的是这类故障：长任务跑到第 8 步，模型还在流畅解题，但**目标已经被
摘要压缩没了**、或**第 2 步的证据全被挤出窗口**——表现是"答得很顺，
答的是另一件事"。

三条设计纪律（都来自 W4 门禁的同一套理由）：

1. **纯规则，不调模型。** 检测器本身必须零成本、零延迟、可复现，
   否则它自己就成了新的不确定性来源。
2. **命中只重注入，不中断。** 报警后把目标与证据重新塞回首尾
   （`reinjection`），任务继续跑——误报的代价是多花一点 token，
   中断的代价是丢掉已完成的 8 步。
3. **三指标各抓一个正交故障**，阈值写死在常量里：阈值是"报警线"
   不是"调参旋钮"，要调就得连报告一起改（和 W6 治理权重同一纪律）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "GOAL_ECHO_THRESHOLD",
    "SOURCE_LOSS_THRESHOLD",
    "REPEAT_ACTION_WINDOW",
    "DriftReport",
    "check_drift",
    "terms",
]

# goal_echo < 0.3 → 目标关键词在当前 payload 里几乎消失。
GOAL_ECHO_THRESHOLD = 0.3
# 窗口内证据 < 50% → 之前引用过的 chunk 大半被挤出。
SOURCE_LOSS_THRESHOLD = 0.5
# 最近 3 步内同一 (tool, args_hash) 出现 ≥ 2 次 → 死循环前兆。
REPEAT_ACTION_WINDOW = 3
REPEAT_ACTION_COUNT = 2


def terms(text: str) -> set[str]:
    """关键词集合：ASCII 词 + 中文二元组（bigram）。

    与 embedding 判定的分工：这里要的是**字面重合度**——"目标还在不在
    窗口里"是字面问题，语义相近不算数（摘要复述目标也必须提到它的词）。
    bigram 而不是单字：单字在中文里几乎处处重合，会稀释分母。
    """
    result: set[str] = set(re.findall(r"[a-z0-9_]+", (text or "").lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        if len(run) == 1:
            result.add(run)
        else:
            result.update(run[i : i + 2] for i in range(len(run) - 1))
    return result


@dataclass(frozen=True)
class DriftReport:
    """一次漂移检测的结果。字段直接进 `context_drift` trace 事件。"""

    # 目标关键词集合在 payload 中的重合度（0–1）。
    goal_echo: float = 1.0
    # 之前引用过的证据 chunk 仍在窗口内的比例（0–1）。无历史引用时为 1.0。
    source_retention: float = 1.0
    # 检测窗口内同一 (tool, args_hash) 的最大重复次数。
    max_repeat: int = 0
    # 命中的报警（goal_lost / source_loss / repeat_action），可能多项同时。
    alerts: tuple[str, ...] = ()
    # 建议的重注入内容（命中任一报警时非空；调用方决定是否执行）。
    reinject: tuple[str, ...] = field(default_factory=tuple)

    @property
    def drifted(self) -> bool:
        return bool(self.alerts)


def check_drift(
    goal: str,
    payload_texts: Sequence[str],
    cited_chunk_ids: Sequence[str] | None = None,
    in_window_ids: Sequence[str] | None = None,
    recent_actions: Sequence[tuple[str, str]] | None = None,
) -> DriftReport:
    """三个纯规则指标。

    - `goal`：任务原始目标文本（任务开始时存一份，永不参与压缩）。
    - `payload_texts`：当前 payload 里所有文本块（含摘要/资料/记忆）。
    - `cited_chunk_ids`：之前轮次引用过的证据 chunk_id 集合。
    - `in_window_ids`：当前窗口内仍存在的 chunk_id 集合。
    - `recent_actions`：最近 N 步的 (tool 名, args_hash) 序列。
    """
    goal_terms = terms(goal)
    payload_terms: set[str] = set()
    for text in payload_texts:
        payload_terms |= terms(text)

    echo = (
        len(goal_terms & payload_terms) / len(goal_terms) if goal_terms else 1.0
    )

    retention = 1.0
    if cited_chunk_ids is not None and in_window_ids is not None:
        cited = set(cited_chunk_ids)
        if cited:
            retention = len(cited & set(in_window_ids)) / len(cited)

    max_repeat = 0
    if recent_actions:
        window = list(recent_actions[-REPEAT_ACTION_WINDOW:])
        for action in set(window):
            max_repeat = max(max_repeat, window.count(action))

    alerts: list[str] = []
    reinject: list[str] = []
    if echo < GOAL_ECHO_THRESHOLD:
        alerts.append("goal_lost")
        reinject.append(f"[任务目标重注入] {goal}")
    if retention < SOURCE_LOSS_THRESHOLD:
        alerts.append("source_loss")
        reinject.append("[证据重注入] 之前引用过的资料已部分挤出窗口，见尾部资料块")
    if max_repeat >= REPEAT_ACTION_COUNT:
        alerts.append("repeat_action")
        # 死循环不重注入（重注入解决不了循环），只报警——终止策略是
        # W9 多 Agent 的职责，这里提前留下信号。
        reinject = [r for r in reinject if not r.startswith("[证据")]

    return DriftReport(
        goal_echo=echo,
        source_retention=retention,
        max_repeat=max_repeat,
        alerts=tuple(alerts),
        reinject=tuple(reinject),
    )

"""handoff 协议与环路检测（W9 D3）。

**预算随 handoff 传递、不重置**，这是本周最容易被写错的一行。三个角色各拿
一份"自己的预算"，总成本就是配置值的三倍；而看板上每一跳都显示"余额充足"。
所以 `budget_left_tokens` 是**同一个数在递减**，不是每跳重新赋值。

**环路指纹防的是"两个角色互踢皮球"**（ROADMAP 点名）。它的表现很隐蔽：
Outcome 层只是"成功率略降"，而 `path_repeat_rate` 会从 0.05 跳到 0.6。
所以检测必须发生在编排层，而不是等评测报告出来才发现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

__all__ = ["Handoff", "HandoffLedger", "LoopDetected"]


class LoopDetected(RuntimeError):
    """出现环路。**中止并说明是哪条边在重复**，不能静默重试。

    静默重试的代价：每一轮都在花真钱，而日志里只有一串看起来正常的交接。
    """


@dataclass(frozen=True, slots=True)
class Handoff:
    """一次角色交接。它是一个**消息**，不是函数调用——接收方可以拒绝。"""

    from_role: str
    to_role: str
    task: str
    # 中间产物 id（引用 / 证据编号）。传 id 而不是正文：正文会让 handoff
    # 悄悄变成第二条上下文通道，预算就管不住了。
    artifacts: tuple[str, ...] = ()
    # 剩余预算。**递减传递**，不是每跳重置。
    budget_left_tokens: int = 0
    # 全局步号。上限由 supervisor 判，但每跳都带着它，接收方能看见"还剩几步"。
    step_no: int = 0

    @property
    def key(self) -> str:
        """环路指纹：`(from, to, task)`。

        **不含 step_no 与 artifacts**：带 step_no 每次都不一样，指纹永远不重复，
        检测就等于没有；带 artifacts 会让"同一件事被换了个说法重问"逃过检测，
        而那恰恰是要抓的。
        """

        raw = f"{self.from_role}\x00{self.to_role}\x00{self.task.strip()}"
        return sha256(raw.encode("utf-8")).hexdigest()[:12]

    def as_dict(self) -> dict[str, object]:
        return {
            "from": self.from_role,
            "to": self.to_role,
            "task": self.task,
            "budget_left_tokens": self.budget_left_tokens,
            "step_no": self.step_no,
            "key": self.key,
        }


@dataclass(slots=True)
class HandoffLedger:
    """handoff 流水 + 两种环路判据。"""

    handoffs: list[Handoff] = field(default_factory=list)
    # 同一条边出现几次就判环。2 = "同一个请求被提了第二次"。
    # 阈值设 3 看起来更宽容，实际会让第二次重复整个跑完——而指纹早已说明不会有新结果。
    repeat_threshold: int = 2

    def record(self, handoff: Handoff) -> None:
        """记一条交接。**判出环就直接抛**，不返回一个"你自己看着办"的标记。

        抛异常而不是返回布尔：调用方只要写了 `record` 就没法"忘了检查"。
        这条 handoff 会被留在流水里——它是证据，擦掉只会让事后复盘少一条线索。
        """

        self.handoffs.append(handoff)
        reason = self.loop_reason()
        if reason is not None:
            raise LoopDetected(reason)

    def loop_reason(self) -> str | None:
        """有环返回原因，无环返回 None。两种环都要抓：

        1. **同边重复**：A→B 的同一个任务被提了 ≥2 次（最常见）。
        2. **两点来回**：A→B→A→B，每条边各只出现一次，但整体在原地转。
           只查"同边重复"会漏掉它，而它是最典型的"互踢皮球"。
        """

        counts: dict[str, int] = {}
        for handoff in self.handoffs:
            counts[handoff.key] = counts.get(handoff.key, 0) + 1
            if counts[handoff.key] >= self.repeat_threshold:
                return (
                    f"同一条 handoff 重复 {counts[handoff.key]} 次："
                    f"{handoff.from_role}→{handoff.to_role}「{handoff.task[:40]}」"
                )

        roles = [item.from_role for item in self.handoffs] + (
            [self.handoffs[-1].to_role] if self.handoffs else []
        )
        return _find_cycle(roles)

    def as_list(self) -> list[dict[str, object]]:
        return [item.as_dict() for item in self.handoffs]


def _find_cycle(path: list[str]) -> str | None:
    """在角色序列里找长度 2..4 的重复片段。

    两个刻意的限制：

    1. **不做通用图环检测**。真实的互踢皮球是两三个角色在小范围里来回，
       长度超过 4 的"环"实际上已经是正常的长流程了，抓它只会误报。
    2. **窗口里必须有两个以上不同角色**。`[r, r]` 这种"同一个角色连着被派
       两次"，不是互踢皮球——那是研究员在逐个查子问题，或者调度器自己卡住了。
       后者归"无进展"与步数守卫管；在这里判环会让正常的多轮检索直接断掉
       （实测：两个子问题的正常流程在第 4 轮被误杀）。
    """

    if len(path) < 4:
        return None
    for size in (2, 3, 4):
        for start in range(0, len(path) - 2 * size + 1):
            window = path[start : start + size]
            following = path[start + size : start + 2 * size]
            if len(set(window)) < 2:
                continue
            if window == following:
                return f"角色循环：{' → '.join(window)} 出现了两轮"
    return None

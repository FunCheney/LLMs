"""上下文工程的数据契约（W7 D1）：预算分配的单位不是字符串，是 ContextItem。

为什么需要这一层：W6 之前的"喂给模型什么"是各链路自己拼的——回答链路拼
`render_prompt`，队列链路拼 messages，评测链路又拼一遍。拼装逻辑分散意味着
三件事没人能回答：**预算多少、超了先裁谁、这条为什么没进来**。W7 把拼装
收敛成一条流水线：

    候选 ContextItem → allocate（按槽位预算裁剪 + 记排除原因）
                      → builder（组装成标准 list[Message]）

三条纪律（都有对应的单测断言）：

1. **信任级先于相关性**：`trust="external"` 的内容（网页、竞品资料、工具返回）
   无论多相关都不能进 system 槽位。这是 W10 注入防护在 Context 层的同一条
   原则，提前在这里立住，W10 只需补检测器。
2. **排除原因互斥且穷尽**：每个被排除的候选有且只有一个原因（`ExclusionReason`），
   由**第一道拦下它的闸门**负责记录——后道闸门不许覆盖。两个原因并存等于
   两份互相矛盾的 Debug Report。
3. **生命周期显式**：`expires_at` 让"过期"在预算层可判定，而不是靠调用方
   各自记得查（谁忘谁负责——但"谁忘"本身就是不可运维的答案）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..llm.types import Message

__all__ = [
    "ContextBudget",
    "ContextItem",
    "ContextKind",
    "ExclusionReason",
    "TrustLevel",
    "now_utc",
]


class ContextKind(StrEnum):
    """ContextItem 的槽位归属。一个 item 只属于一个槽位，槽位决定预算与裁剪优先级。"""

    SYSTEM = "system"
    # RAG 命中的资料（W6 的引用链路产物）。
    RETRIEVED = "retrieved"
    # 长期记忆召回项（W7 D3）。
    MEMORY = "memory"
    # 工作记忆 / 对话历史（W7 D2）。
    HISTORY = "history"
    # 工具返回结果（W9 前主要是 RAG 之外的检索工具）。
    TOOL_RESULT = "tool_result"


class TrustLevel(StrEnum):
    """内容的信任级。

    - `system`：我们自己写的 prompt（角色设定、引用规则）。
    - `user`：用户会话中产生的内容（含从会话提取的长期记忆）。
    - `external`：网页 / 竞品资料 / 工具返回。**永不可当指令**，也不可进
      system 槽位（见 builder 的校验与 schema 顶部纪律 1）。
    """

    SYSTEM = "system"
    USER = "user"
    EXTERNAL = "external"


class ExclusionReason(StrEnum):
    """候选没进上下文的六类原因。**互斥且穷尽**——每个候选最多命中一个。

    前三类发生在 `allocate()` 之前的上游（由调用方带进来，`allocate` 只负责
    不覆盖）；后三类由预算/压缩层自己判定。每类都指向一个明确的"该去改哪里"，
    见 docs/w7/context-budget-spec.md 的对照表。
    """

    # 根本没被召回（W5 检索：chunk / 改写 / RRF）。
    NOT_RETRIEVED = "not_retrieved"
    # ACL 过滤掉了（W6 ACL，不是 bug，是权限生效）。
    FILTERED_PERMISSION = "filtered_permission"
    # 过期被治理降权 / 剔除（W6 知识治理）。
    FILTERED_EXPIRED = "filtered_expired"
    # 召回了但名次不够（W5 重排：排到第 8 位只取前 5）。
    RANKED_OUT = "ranked_out"
    # 排上了但预算被别的槽位 / 同槽高位项挤掉（本包 budget.py）。
    BUDGET_SQUEEZED = "budget_squeezed"
    # 压缩 / 摘要时被合并掉了（W7 D2 摘要策略）。
    COMPRESSED_AWAY = "compressed_away"


def now_utc() -> datetime:
    """统一取"现在"的入口。测试要推进时间时只需替换这一个函数。"""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ContextItem:
    """进入预算分配的最小单位。

    五件套（来源 / 版本 / 权威度 / 信任级 / 生命周期）不是元数据装饰：
    每一件都对应一条预算层的判定——来源用于 Debug Report 定位、版本用于
    "同文档不同版该留谁"、权威度参与裁剪排序、信任级决定能不能进 system、
    生命周期决定过期判定。
    """

    # 调试与归因用的稳定标识（如 chunk_id / memory key）。同类候选里不必唯一，
    # 但 Debug Report 靠它回答"这条为什么没进上下文"。
    id: str
    # 正文。对 history / tool_result 类 item，正文同时挂在 `message` 上
    # （builder 会透传原 Message，保住 role / tool_call_id）。
    content: str
    kind: ContextKind
    # 哪个文档 / 哪次会话 / 哪个工具：Debug Report 与审计要能指回去。
    source: str
    # 文档版本 / prompt 版本。没有就 None，**别造假**——假的版本号比没有
    # 更糟：排障会沿它去找一个不存在的版本。
    version: str | None = None
    # 权威度 0–1：官方 > 自有 > 未知站点（复用 W6 治理的质量分）。
    # 只在同槽位内参与裁剪排序，不跨槽位比较（跨槽位由固定优先级决定）。
    authority: float = 0.5
    trust: TrustLevel = TrustLevel.USER
    # 生命周期终点。None = 不过期（如品牌调性这类长期项）。
    expires_at: datetime | None = None
    # 槽位内相关性（0–1，检索分 / 记忆置信度归一后的值）。同槽位超预算时
    # 低分先裁。缺省 0.5 让"没给分的调用方"仍能得到确定性行为。
    score: float = 0.5
    # 原始消息（history / tool_result 用）。builder 透传它而不是重新拼 role，
    # 否则 tool_call_id、provider_meta 这些契约字段会在拼装中丢失。
    message: Message | None = None
    # 上游已判定的排除原因。非 None 的 item 视为**已出局**，`allocate()`
    # 只把它记进报告，不再参与预算（也不覆盖原因——见纪律 2）。
    excluded: ExclusionReason | None = None
    # 额外备注（进日志不进 prompt）。例如 ACL 拒绝时记"哪条策略"。
    note: str = ""

    def is_expired(self, at: datetime | None = None) -> bool:
        """生命周期判定。没有 expires_at 的项永远不过期。"""
        if self.expires_at is None:
            return False
        return self.expires_at <= (at or now_utc())


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """各槽位的 token 上限（粗估口径，见 ADR-W7-01：与 W6 estimate_tokens 同源）。

    两个不变量（`validate()` 在 allocate 前强制）：

    1. 每个槽位 ≥ 0，且五槽之和 ≤ `window`（模型输入窗口减去 reserve_output）。
       破了这条，"预算"就只是五个互不相干的数字，总输入照样爆窗口。
    2. `reserve_output` 不可侵占。长输入把它挤掉，`max_tokens` 被压到 0，
       厂商直接 400——这是 W7 任务清单里点名的第一号返工原因。
    """

    # 模型输入窗口减 reserve_output 之后的"可用于输入"总量。
    # 默认 8000：五槽默认之和 7000（400+3000+600+2000+1000）必须在窗口内，
    # 否则 validate() 直接拒绝——那是配置错误，不该留到运行时再裁。
    window: int = 8000
    system: int = 400
    retrieved: int = 3000
    memory: int = 600
    history: int = 2000
    tool_results: int = 1000
    # 给模型输出留的余量。不进任何输入槽位，也不许被挪用。
    reserve_output: int = 600

    def validate(self) -> None:
        """结构错误在分配前炸，不留到"第一次真实请求 400"。"""
        slots = self.system + self.retrieved + self.memory + self.history + self.tool_results
        if self.window <= 0:
            raise ValueError(f"window 必须大于 0，当前 {self.window}")
        if self.reserve_output < 0:
            raise ValueError(f"reserve_output 不能为负，当前 {self.reserve_output}")
        if slots > self.window:
            raise ValueError(
                f"五槽之和 {slots} 超过窗口 {self.window}：槽位预算加起来都不成立，"
                f"逐项调小，或把 window 调大（并确认模型窗口真有那么大）"
            )
        for name in ("system", "retrieved", "memory", "history", "tool_results"):
            if getattr(self, name) < 0:
                raise ValueError(f"槽位 {name} 不能为负")


@dataclass(frozen=True, slots=True)
class Exclusion:
    """Debug Report 的一行：哪个候选、被哪道闸、为什么拦下。"""

    # 出局候选的 ContextItem.id。
    item_id: str
    kind: ContextKind
    reason: ExclusionReason
    # 附加定位信息（如 ACL 策略名、摘要覆盖到第几轮）。不带正文——
    # 被拦掉的可能是无权内容，把它写进报告等于用报告绕过权限。
    note: str = ""
    # 判定来源：上游带进来（upstream）还是本层判定（budget）。
    # 用于核对"六类原因只有一个记录者"的纪律是否被破坏。
    decided_by: str = "budget"


@dataclass(frozen=True, slots=True)
class AllocatedContext:
    """`allocate()` 的结果：入选项 + 逐槽位用量 + 全部出局原因。

    Debug Report 是**免费的**：排除原因在裁剪时顺手记录，不另建一套埋点
    （另一套必然和这里对不上——W7 任务清单第 7 号返工原因）。
    """

    # 入选项，按槽位分组保持分配后的确定性顺序。
    items: tuple[ContextItem, ...] = ()
    # 每个槽位实际使用的 token 数（粗估）。
    used: dict[str, int] = field(default_factory=dict)
    # 全部出局项（含上游带进来的三类）。
    exclusions: tuple[Exclusion, ...] = ()
    # 本次使用的预算（原样带回，报告可复算）。
    budget: ContextBudget | None = None

    @property
    def total_used(self) -> int:
        return sum(self.used.values())

    def by_kind(self, kind: ContextKind) -> list[ContextItem]:
        return [item for item in self.items if item.kind == kind]

    def explain(self, item_id: str) -> str:
        """回答"这条为什么没进上下文"。返回值直接可进日志/工单。

        找不到任何记录 = 这个 id 根本不在本次候选里——这本身就是答案
        （该去上游查"为什么没送进来"），不能含糊成 "unknown"。
        """
        for exclusion in self.exclusions:
            if exclusion.item_id == item_id:
                return f"{exclusion.reason.value}:{exclusion.note}" if exclusion.note else (
                    exclusion.reason.value
                )
        for item in self.items:
            if item.id == item_id:
                return "included"
        return "not_in_candidates"

    def as_dict(self) -> dict[str, Any]:
        """给日志/trace 的投影。字段固定，W11 成本看板直接读。"""
        return {
            "used": dict(self.used),
            "total_used": self.total_used,
            "included": [item.id for item in self.items],
            "excluded": [
                {
                    "item_id": exclusion.item_id,
                    "kind": exclusion.kind.value,
                    "reason": exclusion.reason.value,
                    "decided_by": exclusion.decided_by,
                }
                for exclusion in self.exclusions
            ],
        }

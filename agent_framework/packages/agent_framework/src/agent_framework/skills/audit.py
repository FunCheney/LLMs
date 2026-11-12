"""工具调用全审计（W8 D5）：每一次调用都要能回答"谁、调了什么、成没成"。

审计字段的来源是 Gate 8 #6：主体 / 工具 / 风险级 / 结果 / replay 标记，
一个都不能少。少一个就会出现"查不到"的工单：
少了主体 → 无法定责；少了风险级 → 无法判断该不该走审批；
少了 replay 标记 → 分不清"这周真的花了钱"还是"跑的是回放"。

**参数只记摘要，不记全文。** 三个理由，第二个最容易被忽略：

1. 参数里常有隐私与密钥（正文、客户名单、token）。
2. 审计日志的读取方比业务数据宽得多（值班、合规、事后复盘），
   把全文写进去等于用审计日志绕过数据权限——和 W6 "被滤掉的块只记数量
   不记内容"是同一条纪律。
3. 摘要足够定位：`args_digest` 能判"两次调用是不是同一份参数"，
   `arg_keys` 能判"模型这次多传了什么"，而这正是排障要问的两个问题。

参数摘要用**规范化后的哈希**（`replay.canonical_args`），所以
`{"a":1,"b":2}` 与 `{"b":2,"a":1}` 得到同一个 digest——顺序不同不该是
"两次不同的调用"。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..observability import current_request_id, current_trace_id
from .contract import ToolOutcome
from .replay import canonical_args

logger = logging.getLogger(__name__)

__all__ = ["AuditLog", "ToolAudit", "args_digest"]

# 进程内保留多少条。1000 条 ≈ 几百 KB，足够"点开最近一次任务看全部调用"，
# 又不至于把内存当数据库用。真正的长期存储在 W11 的 trace 管道。
DEFAULT_CAPACITY = 1000

# 参数摘要长度。12 位十六进制 = 48 bit，同一任务内撞概率可忽略；
# 再长就没人会去比对，等于白占日志。
DIGEST_LEN = 12


def args_digest(arguments: dict[str, Any] | None) -> str:
    return hashlib.sha256(canonical_args(arguments).encode()).hexdigest()[:DIGEST_LEN]


@dataclass(frozen=True, slots=True)
class ToolAudit:
    """一次工具调用的审计条目。字段固定，W11 的看板直接读。"""

    # 本次动作的稳定 id（幂等与对账都用它）。
    action_id: str = ""
    # 内部工具名（`server__tool`）与对应的 Skill 名。两者都要：
    # 一个 Skill 可能包多个工具，一个工具也可能被多个 Skill 复用。
    tool: str = ""
    skill: str = ""
    # 结果状态：ok / tool_not_visible / permission_denied / sandbox_denied / ...
    status: str = ""
    # 风险级（read / write / irreversible）。
    risk: str = "read"
    # 主体 id。组不进审计（组会变，留 id 才能回溯"当时是谁"）。
    principal: str = ""
    # 参数摘要与参数名（**不含值**）。
    args_digest: str = ""
    arg_keys: tuple[str, ...] = ()
    # 耗时与回放标记。
    latency_ms: float = 0.0
    replayed: bool = False
    cost: float = 0.0
    # 审批状态（approved / pending / rejected / ""=不需要）。
    approval: str = ""
    # 任务阶段（可见面与风险判定都依赖它，排障时要能对上）。
    stage: str = ""
    # 备注（命中的规则、回放键、截断句柄）。**只放机器可读的短语**。
    notes: tuple[str, ...] = field(default_factory=tuple)
    request_id: str = ""
    trace_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tool": self.tool,
            "skill": self.skill,
            "status": self.status,
            "risk": self.risk,
            "principal": self.principal,
            "args_digest": self.args_digest,
            "arg_keys": list(self.arg_keys),
            "latency_ms": round(self.latency_ms, 3),
            "replayed": self.replayed,
            "cost": round(self.cost, 6),
            "approval": self.approval,
            "stage": self.stage,
            "notes": list(self.notes),
            "request_id": self.request_id or "-",
            "trace_id": self.trace_id or "-",
        }


class AuditLog:
    """进程内环形缓冲 + 结构化日志。

    两份输出是刻意的：环形缓冲供**测试与 eval 断言**（要能拿到条目对象），
    结构化日志供**线上排查**（进程重启后还在）。只做后者则测试只能靠
    `caplog` 解字符串，只做前者则线上什么都查不到。
    """

    def __init__(self, *, capacity: int = DEFAULT_CAPACITY) -> None:
        self._entries: list[ToolAudit] = []
        self._capacity = capacity

    def record(self, entry: ToolAudit, *, outcome: ToolOutcome | None = None) -> ToolAudit:
        """写入一条。缺 request/trace id 时自动补当前上下文，
        免得每个调用点都要记得传（记得传的代码，迟早有一处忘记）。

        `outcome` 只用来补 `latency_ms` / `replayed` / `cost`——这三项
        在调用方那层拿不全。
        """
        enriched = _fill_context(entry, outcome)
        self._entries.append(enriched)
        if len(self._entries) > self._capacity:
            # 丢最旧的：审计的价值在"最近发生了什么"，不在考古。
            del self._entries[: len(self._entries) - self._capacity]

        logger.info("tool_audit %s", json.dumps(enriched.as_dict(), ensure_ascii=False))
        return enriched

    def entries(self) -> tuple[ToolAudit, ...]:
        return tuple(self._entries)

    def find(self, action_id: str) -> ToolAudit | None:
        for entry in reversed(self._entries):
            if entry.action_id == action_id:
                return entry
        return None

    def summary(self) -> dict[str, Any]:
        """按状态/风险/回放聚合。报告里"工具调用"那一节的数字来自这里。"""
        by_status: dict[str, int] = {}
        by_risk: dict[str, int] = {}
        replayed = 0
        cost = 0.0
        for entry in self._entries:
            by_status[entry.status] = by_status.get(entry.status, 0) + 1
            by_risk[entry.risk] = by_risk.get(entry.risk, 0) + 1
            replayed += 1 if entry.replayed else 0
            cost += entry.cost
        return {
            "total": len(self._entries),
            "by_status": by_status,
            "by_risk": by_risk,
            "replayed": replayed,
            "live": len(self._entries) - replayed,
            # 回放记 0 成本，所以这里的 cost 只含真调用——这正是要能分开的原因。
            "cost": round(cost, 6),
        }

    def clear(self) -> None:
        self._entries.clear()


def _fill_context(entry: ToolAudit, outcome: ToolOutcome | None) -> ToolAudit:
    from dataclasses import replace

    updates: dict[str, Any] = {}
    if not entry.request_id:
        updates["request_id"] = current_request_id() or ""
    if not entry.trace_id:
        updates["trace_id"] = current_trace_id() or ""
    if outcome is not None:
        if not entry.latency_ms:
            updates["latency_ms"] = outcome.latency_ms
        if not entry.replayed:
            updates["replayed"] = outcome.replayed
        if not entry.cost:
            updates["cost"] = outcome.cost

    return replace(entry, **updates) if updates else entry

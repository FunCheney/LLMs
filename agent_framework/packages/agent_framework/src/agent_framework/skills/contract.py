"""工具执行的内部契约（W8）：`ToolOutcome` 是"结果 + 元数据"的一个整体。

为什么不直接复用 `llm.types.ToolResult`：那是**模型看到的东西**（content +
is_error），一个字段都不能多——多出来的字段迟早会被某个 adapter 发到厂商
wire 上（W1 的 `INTERNAL_ONLY_KEYS` 就是为此存在的清单）。

但执行层要回答的问题比模型多：这次是回放还是真调用（成本算不算）、
耗时多少、哪个 server 的哪个工具、失败码是什么。这些如果分开返回
（`call_tool() -> ToolResult` + `client.last_meta`），并发下拿到的会是
**别人的**标记——而"回放成本记 0"一旦记错，W11 的成本看板就失真了。
所以把两者绑成一个对象，一次返回。

`code` 用的是稳定字符串而不是异常类型：审计、评测断言、告警规则都要按它
匹配，异常类名一改全崩，而字符串可以只加不改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.types import ToolResult

__all__ = [
    "CODE_INVALID_ARGUMENTS",
    "CODE_NOT_FOUND",
    "CODE_OK",
    "CODE_PERMISSION_DENIED",
    "CODE_REPLAY_MISS",
    "CODE_SANDBOX_DENIED",
    "CODE_TOOL_ERROR",
    "CODE_TOOL_NOT_VISIBLE",
    "CODE_TIMEOUT",
    "CODE_UNVERIFIED",
    "ToolOutcome",
]

# 成功。其余码互斥——一次调用只有一个 `code`（沿用 W7 六类排除原因的同一条纪律：
# 两个原因并存 = 两份互相矛盾的报告）。
CODE_OK = "ok"
CODE_NOT_FOUND = "tool_not_found"
CODE_TOOL_NOT_VISIBLE = "tool_not_visible"
CODE_INVALID_ARGUMENTS = "invalid_arguments"
CODE_PERMISSION_DENIED = "permission_denied"
CODE_SANDBOX_DENIED = "sandbox_denied"
CODE_TIMEOUT = "tool_timeout"
CODE_TOOL_ERROR = "tool_error"
CODE_REPLAY_MISS = "replay_miss"
CODE_UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """一次工具执行的全部事实。"""

    # 回注给模型的那条 ToolResult（tool_call_id 由执行器补上）。
    result: ToolResult
    # 机器可判的状态码（见上面的常量）。**不参与** prompt。
    code: str = CODE_OK
    # 是否来自回放。True 时 cost 必为 0，且报告里必须与真调用分开统计，
    # 否则"这周省了多少钱"会变成"这周没调用过工具"。
    replayed: bool = False
    # 本次调用的**真实**成本（货币由上游决定，这里只抄数字）。
    # 回放恒为 0——不是"免费"，是"没有产生费用"，两者在报表里必须能分开。
    cost: float = 0.0
    # 本次调用的实际耗时（毫秒）。回放也算：它测的是我们自己的链路。
    latency_ms: float = 0.0
    # 归属：哪个 server 的哪个工具（内部名）。
    server: str = ""
    tool: str = ""
    # 工具返回的结构化载荷（验真层要读它，比如 `{"ok": false}`）。
    # None = 结果不是 JSON 对象（纯文本搜索结果就是这种）。
    payload: dict[str, Any] | None = None
    # 审计备注（不进 prompt）：命中的沙箱规则、回放键、截断句柄……
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.code == CODE_OK and not self.result.is_error

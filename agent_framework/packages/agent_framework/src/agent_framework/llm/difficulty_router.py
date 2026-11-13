"""难度分流（W11 D5 / architecture §8）：简单任务别用贵模型。

## 为什么不改业务代码

`Router` 协议只回答"候选是谁"（`route(task)`）。让它能看**这一次请求的内容**，
需要第二个入口 `route_for(call)`——它是**可选**能力：
实现了就走它，没实现就退回 `route(task)`。于是 `TaskRouter` / `StaticRouter`
一行都不用改，`LLMGateway._plan` 只多一个 `getattr`。

这样做的代价是"协议多了一个隐式成员"。接受它，是因为另一个选项
（把 `task` 撑成带上下文的联合类型）会改掉所有 Router 实现与所有测试。

## 分流的判据必须可解释

`difficulty_signals()` 给的是**带权信号的加总**，不是黑箱分数：

| 信号 | 权重 | 为什么它意味着"难" |
|---|---|---|
| `hard_intent` | **2** | 明确要求论证/对比/规划：难度证据单独成立（见下方不对称代价） |
| `needs_tools` | 1 | 便宜档的 tool-call 稳定性最差 |
| `long_context` | 1 | 长上下文里丢约束的概率更高 |
| `format_strict` | 1 | 格式遵循是便宜档最常见的失分点 |

`hard_intent` 权重是 2：一句短而难的"规划一下灰度方案"只拿 1 分会掉进便宜档。
**只用一个分数、不给理由**的分流器无法复盘：误分流时你不知道该调哪个信号。

## 两条硬约束（看起来冲突，边界就在"难度"与"能力"之间）

1. **只按难度排顺序时，候选链一个都不许少**。便宜档排在前面失败了，仍要落到强档——
   因为"这次答不好"是概率问题，不是"这条路不存在"。
2. **能力不足可以缩短链，而且必须缩短**。便宜档不支持 tools 却把请求递给它，
   拿到的是一个 400，然后由 `gateway` 重试到强档——白白多花一次往返，
   而 `_plan` 一开始就知道它不行。这不是"试错"，这是"明知故犯"。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import Catalog
from .router import TaskRouter

__all__ = ["DifficultyRouter", "difficulty_signals"]

# 复杂意图的词面信号。**不是唯一判据**，只是最便宜的那一层。
_HARD_HINTS: tuple[str, ...] = (
    "对比",
    "比较",
    "为什么",
    "论证",
    "分几步",
    "推理",
    "评估",
    "方案",
    "规划",
    "compare",
    "why",
    "trade-off",
    "design",
)

# 结构化输出的信号。
_FORMAT_HINTS: tuple[str, ...] = ("json", "表格", "markdown", "schema", "字段")

_LONG_QUERY_CHARS = 120

# `hard_intent` 的权重是 2，其余是 1。**这是一个刻意的偏置，理由是不对称的代价**：
#
# 一个明确要求"论证 / 对比 / 规划"的请求，其难度证据已经足够单独成立。
# 若把它和"文本长短"放在同一权重上，一句短而难的"规划一下灰度方案"
# 只拿 1 分、够不到阈值 2，就被当成简单请求——这正是 W11 D5 实测
# 6 条硬样本里漏掉 4 条的原因（`evals/scripts/cost_ab.py` 实验 2）。
#
# 两个方向的误判代价并不对称：
#   - 硬请求误走便宜档：**省了钱、可能答错**，而答错没人会在当天发现；
#   - 简单请求误走强档：多花 20 倍单价，但看得见、可回滚、答案仍是对的。
# 所以在没有真实质量数据可标定权重之前，宁可偏向强档。
_HARD_INTENT_WEIGHT = 2


def difficulty_signals(call: Any) -> dict[str, int]:
    """把"难"拆成**带权**的信号表。权重写在一处，方便误分流时逐条复盘。"""

    messages: Sequence[Any] = getattr(call, "messages", ()) or ()
    text = " ".join(
        str(getattr(message, "content", "")) for message in messages
    ).lower()

    return {
        "needs_tools": 1 if getattr(call, "tools", None) else 0,
        "hard_intent": _HARD_INTENT_WEIGHT if any(hint in text for hint in _HARD_HINTS) else 0,
        "long_context": 1 if len(text) > _LONG_QUERY_CHARS else 0,
        "format_strict": 1 if any(hint in text for hint in _FORMAT_HINTS) else 0,
        "explicit_strong": 1 if str(getattr(call, "task", "")) in {"deep_research"} else 0,
    }


@dataclass
class DifficultyRouter:
    """按难度给出候选顺序。`cheap` 与 `strong` 来自 `models.yaml` 的 alias。"""

    catalog: Catalog
    cheap: str
    strong: str
    # 达到这个分数就走强档。阈值是**口径**，写在 dataclass 里而不是散在代码中。
    threshold: int = 2
    # 能力检查（可选）：返回 True 表示"这个 alias 能满足这次的硬需求"。
    # 不传就只按 catalog 里的 alias 存在性判断——这是最弱但不骗人的默认。
    capability_check: Callable[[str, Any], bool] | None = None
    _fallback: TaskRouter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = TaskRouter(self.catalog)

    @property
    def name(self) -> str:
        return "difficulty"

    # -- Router 协议 -------------------------------------------------------- #

    def route(self, task: str) -> list[str]:
        """没有请求上下文时的回落：按 task 的默认路由。"""

        return self._fallback.route(task)

    # -- 可选增强 ---------------------------------------------------------- #

    def route_for(self, call: Any) -> list[str]:
        """按这一次请求的难度排候选。

        难度只改**顺序**；能力不足才**删**候选。两者的区别见模块文档。

        链的构造是"首选 + 另一档 + `models.yaml` 里配置的默认链"再保序去重，
        而不是硬写 `[cheap, strong]`——后者会把 `routes.strong.fallback`
        里配的兜底（如 `gpt-4o-mini`）悄悄丢掉，让"只改顺序"变成"缩短链"。
        """

        score = sum(difficulty_signals(call).values())
        base = self._fallback.route(call.task)  # 默认链；未知 task 的 warning 也在这里
        cheap_ok = self.capability_check is None or self.capability_check(self.cheap, call)

        if not cheap_ok:
            # 便宜档扛不住硬需求时直接挪走，而不是"先试一下再说"：
            # 试一次的代价是一次 400 + 一次重试，而这不是"难度"问题。
            chain = [alias for alias in (self.strong, *base) if alias != self.cheap]
        elif score >= self.threshold:
            chain = [self.strong, *base, self.cheap]
        else:
            chain = [self.cheap, *base, self.strong]

        return [alias for alias in dict.fromkeys(chain) if _known(self.catalog, alias)]

    def explain(self, call: Any) -> dict[str, Any]:
        """给日志/报告用：分数、信号、最终选择。误分流时靠它归因。"""

        signals = difficulty_signals(call)
        score = sum(signals.values())
        picked = self.route_for(call)
        return {
            "signals": signals,
            "score": score,
            "threshold": self.threshold,
            "candidates": picked,
            "reason": "强档" if score >= self.threshold else "便宜档",
        }


def _known(catalog: Catalog, alias: str) -> bool:
    try:
        catalog.get_model(alias)
    except Exception:  # noqa: BLE001 - 未知 alias 就是"这条路不存在"
        return False
    return True

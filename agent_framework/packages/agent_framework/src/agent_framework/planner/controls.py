"""自主性的上限与控制权分配（W9 D1）。

**为什么把"自主"写成上限而不是口号。** "让 Agent 自主规划"这句话没有可验收的
含义；`max_steps=8 / max_cost=0.5 / max_seconds=120` 才有。上限的价值不在
平时（平时用不到），在于失控那一刻**它是真的会中断**——所以这里的
`BudgetMeter.exhausted()` 返回的是一个原因码，规划器拿到它必须停下，
而不是记一条 warning 继续跑。

**为什么控制权要写成矩阵。** 被问"你怎么保证 Agent 不乱来"，答案不该是
"我在 prompt 里加了约束"——prompt 是建议，不是机制。这张矩阵把 7 个决定
分了权，模型只拿到 3 个；而且它在运行时被 `owner_of()` 真的查询，
不是文档里的一段话（`test_control_matrix_has_no_dead_entries` 会盯着这一点）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

__all__ = [
    "CONTROL_MATRIX",
    "OWNER_CODE",
    "OWNER_HUMAN",
    "OWNER_MODEL",
    "AutonomyBudget",
    "BudgetMeter",
    "ControlMatrix",
    "ControlViolation",
    "StopReason",
    "matrix",
]


class StopReason(StrEnum):
    """中止原因。**每一种都必须能被写进日志并让人看懂**。

    没有 `unknown`：一个说不出原因的中止，等于这次运行不可归因。
    """

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    MAX_SECONDS = "max_seconds"
    MAX_REPLANS = "max_replans"
    PLAN_LOOP = "plan_loop"
    NO_PROGRESS = "no_progress"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AutonomyBudget:
    """一次任务允许消耗的四种资源。

    `max_steps` 数的是**执行步数**而不是模型轮次：一轮里模型可以提议多次调用，
    按轮次计数会让"一轮发 5 个工具"绕过限制。

    `max_cost` 用 `Decimal` 而不是 `float`：累加 0.1 三次不等于 0.3，
    预算判定用浮点会出现"看起来没超但账单超了"。
    """

    max_steps: int = 8
    max_tokens: int = 60_000
    max_cost: Decimal = Decimal("0.50")
    max_seconds: float = 120.0
    max_replans: int = 2

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens 必须 >= 1")
        if self.max_cost < 0:
            raise ValueError("max_cost 不能为负")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds 必须 > 0")
        if self.max_replans < 0:
            raise ValueError("max_replans 不能为负")


class BudgetMeter:
    """预算计量器。**所有扣减都经过它**，规划器不许自己加变量。

    时钟注入是为了让 `max_seconds` 在测试里能确定性触发——用 `sleep` 触发超时
    会让用例慢且不稳定，最后大家把它 skip 掉，守卫就等于没测。
    """

    def __init__(
        self,
        budget: AutonomyBudget,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        budget.validate()
        self._budget = budget
        self._clock = clock
        self._started = clock()
        self._steps = 0
        self._tokens = 0
        self._cost = Decimal("0")
        self._replans = 0

    # -- 只读 -------------------------------------------------------------- #

    @property
    def budget(self) -> AutonomyBudget:
        return self._budget

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def cost(self) -> Decimal:
        return self._cost

    @property
    def replans(self) -> int:
        return self._replans

    @property
    def tokens_left(self) -> int:
        return max(0, self._budget.max_tokens - self._tokens)

    def elapsed(self) -> float:
        return float(self._clock() - self._started)

    # -- 扣减 -------------------------------------------------------------- #

    def charge_tokens(self, tokens: int) -> None:
        self._tokens += max(0, int(tokens))

    def charge_cost(self, cost: float | Decimal) -> None:
        self._cost += Decimal(str(cost))

    def count_step(self) -> None:
        self._steps += 1

    def count_replan(self) -> None:
        self._replans += 1

    # -- 判定 -------------------------------------------------------------- #

    def exhausted(self) -> StopReason | None:
        """超限原因；None = 还有额度。

        检查顺序**固定**（步数 → token → 成本 → 墙钟）。同时超两项时报告哪个
        取决于这个顺序，所以它必须是确定的：否则同一个用例两次跑出不同原因，
        日志就没法用来做趋势。
        """

        if self._steps >= self._budget.max_steps:
            return StopReason.MAX_STEPS
        if self._tokens >= self._budget.max_tokens:
            return StopReason.MAX_TOKENS
        if self._cost >= self._budget.max_cost:
            return StopReason.MAX_COST
        if self.elapsed() >= self._budget.max_seconds:
            return StopReason.MAX_SECONDS
        return None

    def replans_exhausted(self) -> bool:
        return self._replans >= self._budget.max_replans

    def snapshot(self) -> dict[str, object]:
        """进日志/评测用。`left` 与 `used` 都给，因为读者关心的是不同的量。"""

        return {
            "steps_used": self._steps,
            "steps_left": max(0, self._budget.max_steps - self._steps),
            "tokens_used": self._tokens,
            "tokens_left": self.tokens_left,
            "cost_used": str(self._cost),
            "seconds": round(self.elapsed(), 3),
            "replans": self._replans,
        }


# --------------------------------------------------------------------------- #
# 控制权矩阵
# --------------------------------------------------------------------------- #

OWNER_MODEL = "model"
OWNER_CODE = "code"
OWNER_HUMAN = "human"


# 七个决定。**模型只拿三个**，而且都是"下一步做什么"这类可以事后审计的选择；
# 一切"能不能"的问题都在代码手里，"要不要对外"在人手里。
CONTROL_MATRIX: dict[str, str] = {
    "下一步做什么": OWNER_MODEL,
    "调用哪个工具": OWNER_MODEL,
    "工具参数": OWNER_MODEL,
    # 可见面是代码算的：模型不该知道自己看不见的工具存在，更不能自己"申请"看见。
    "工具是否可见": OWNER_CODE,
    # 预算/步数/环路：模型说"还能继续"不作数。
    "是否能继续": OWNER_CODE,
    # 越权判定必须是代码：模型自评"我有权限"是最危险的一类幻觉。
    "是否越过权限": OWNER_CODE,
    # 对外发布是人：W10 接真 HITL 之前，这里是 pending 占位。
    "是否发布": OWNER_HUMAN,
}


class ControlViolation(RuntimeError):
    """模型试图决定它不该决定的事。

    抛异常而不是"忽略并纠正"：把"模型想自己终止预算"这件事静默改掉，
    就永远不知道它在多少次运行里试图绕开守卫。
    """


@dataclass(frozen=True, slots=True)
class ControlMatrix:
    """矩阵的查询接口。运行时通过它判权，测试通过它查"有没有死条目"。"""

    owners: Mapping[str, str] = field(default_factory=lambda: dict(CONTROL_MATRIX))

    def owner_of(self, decision: str) -> str:
        try:
            return self.owners[decision]
        except KeyError:
            # 未登记的决定默认归代码：先拒绝，再补登记。
            # 反过来的默认值（归模型）会让新增决定悄悄扩大模型的权限。
            return OWNER_CODE

    def model_may_decide(self, decision: str) -> bool:
        return self.owner_of(decision) == OWNER_MODEL

    def human_required(self, decision: str) -> bool:
        return self.owner_of(decision) == OWNER_HUMAN

    def assert_code_owned(self, decision: str) -> None:
        """给"这条必须由代码判"的地方用。写错归属时**在启动/首次调用就炸**。"""

        owner = self.owner_of(decision)
        if owner != OWNER_CODE:
            raise ControlViolation(f"{decision} 应由代码决定，矩阵里是 {owner}")

    def describe(self) -> list[dict[str, str]]:
        """给 ADR 与 `/v1/plan/config` 用的可读形态。"""

        return [
            {"decision": name, "owner": owner} for name, owner in self.owners.items()
        ]


def matrix() -> ControlMatrix:
    return ControlMatrix()

"""SLO 与误差预算（W11 D1）。

## 这个文件解决的不是"算个数"，是"口径只能有一处"

W11 之前，成功率/延迟/成本这三个数散在三个地方算：`evals/metrics.py` 算评测口径、
`observability.py` 打日志、`regression_gate.py` 判门禁。三者对"p95"的定义是同一个
（`TaskMetrics.latency_p95`，nearest-rank），但**没人把它写成一句可引用的话**。
于是"p95 是 1.2s 还是 1.5s"这种争论，每次都靠翻代码。

所以这里做三件事，按重要性排：

1. **把三个 SLI 的取值口径写成常量**（`SLOS`），并注明它读的是哪个字段；
2. **把误差预算的算式写一次**，告警与报告都调它，不各自实现；
3. **把"延迟达标"转换成"达标请求占比"**，从而只有一种预算算法。

## 第 3 条为什么要这样绕

latency 的 SLO 是"p95 ≤ 1500ms"，形式上是 `lte`，与 availability 的 `gte` 不同。
如果为它单独写一套"预算是多少毫秒"，就会出现两个预算概念：
availability 说"还能坏 1% 的请求"，latency 说"还能慢 50ms"——这两句话没法相加，
也没法一起回答"这个窗口还剩多少余量"。

转成"达标占比"之后两者同构：`SLI = 达标样本 / 总样本`，目标 0.95。
于是 burn rate、剩余预算、告警阈值全部只有一套公式。

**代价**：报"p95 是多少毫秒"时要从样本重算一次（`percentile()`），
不能从预算反推。这是可接受的——报数用样本，判预算用占比。

## 不做什么

- 不接 Prometheus / 不开时序库：当前数据源是 `RequestLog` 的 JSONL 与
  `evals/history/*.json`，一个 `python ops/slo_report.py` 就够了。
  引入时序库要先有"每秒 N 条"的写入压力，现在没有。
- 不做多窗口（1h/6h/3d）燃烧率告警：**单窗口 + 持续时间**已经能挡住抖动，
  多窗口燃烧率是给有 7×24 值班的团队用的，我们只有一个值班的人。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "ALERT_RULES",
    "AlertRule",
    "BudgetStatus",
    "COMPARISON_GTE",
    "COMPARISON_LTE",
    "SLO",
    "SLOS",
    "alert_fires",
    "SLI_THRESHOLDS",
    "SLO_BY_NAME",
    "compliance",
    "error_budget",
    "evaluate",
    "percentile",
    "series_from_eval",
]

COMPARISON_GTE = "gte"
COMPARISON_LTE = "lte"


@dataclass(frozen=True)
class SLO:
    """一条 SLO。`sli` 是**人能读、且能对上字段**的口径描述，不是查询语句。

    写查询语句（PromQL/SQL）在文件里，等于把"口径"和"存储"绑在一起；
    换一次存储就要重写一遍口径，而重写时口径通常会悄悄变。
    """

    name: str
    sli: str
    target: float
    window: str
    comparison: str
    unit: str = "ratio"
    # 口径的边界条件。**必须写**：不写的话，半年后没人知道这条 SLO 算不算 "skipped"。
    discipline: str = ""

    @property
    def is_ratio(self) -> bool:
        return self.unit == "ratio"


SLOS: tuple[SLO, ...] = (
    SLO(
        name="availability",
        sli="RequestLog 中 error_code 为空的请求占比",
        target=0.99,
        window="28d",
        comparison=COMPARISON_GTE,
        discipline=(
            "分母**不含** 4xx 由我方鉴权/护栏拦掉的请求：那些是调用方的问题，"
            "不是我们不可用。含进去的后果是'被扫的时候可用性看起来下降'，"
            "于是真正的故障被淹没在攻击噪声里。"
        ),
    ),
    SLO(
        name="latency_p95",
        sli="首 token 延迟（first_token_ms）的 p95 ≤ 1500ms；流式请求才有该字段",
        target=0.95,
        window="7d",
        comparison=COMPARISON_LTE,
        unit="ms",
        discipline=(
            "DoD 原话是「首 token p95 ≤ 1.5s」，所以取 first_token_ms 而不是整体 latency。"
            "非流式请求没有 first_token_ms——此时**不参与**这条 SLO，"
            "而不是拿 latency_ms 顶替：两者量级差 10 倍，混算等于把长尾抹平。"
        ),
    ),
    SLO(
        name="cost_per_task",
        sli="单任务成本（cost 求和 / 非缓存命中任务数）≤ 2.0 CNY",
        target=0.95,
        window="7d",
        comparison=COMPARISON_LTE,
        unit="CNY",
        discipline=(
            "缓存命中与回放的记录**成本记 0 但要单列**（W8 已定：`replayed`/_meta）；"
            "`usage_estimated=True` 的记录在报告里单独给占比——估算占比 >20% 时"
            "这条 SLO 的结论不可信（W1 Day5 的教训）。"
        ),
    ),
)

SLO_BY_NAME = {slo.name: slo for slo in SLOS}


def percentile(samples: Sequence[float], q: float) -> float | None:
    """nearest-rank 百分位，与 `evals/metrics.py` 同一口径。

    不用插值：插值会造出一个数据里不存在的值（"p95 = 1.4327ms"），
    而值班要的是"最慢的那一批真实请求有多慢"。
    """

    if not samples:
        return None
    if not 0 < q <= 1:
        raise ValueError(f"q 必须在 (0, 1]，收到 {q}")

    ordered = sorted(samples)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


# `lte` 型 SLO 的达标阈值（毫秒 / 元）。与 target（达标占比）**分开存**：
# `target=0.95` 说的是"允许 5% 请求不达标"，`threshold=1500ms` 说的是
# "什么算达标"。把这两件事塞进一个字段的表现是——改阈值的时候把预算也改了。
SLI_THRESHOLDS: dict[str, float] = {
    "latency_p95": 1500.0,
    "cost_per_task": 2.0,
}


def compliance(slo: SLO, value: float | None) -> int | None:
    """单个样本是否达标。`None` 表示这个样本不参与该 SLO（缺失字段）。

    `gte` 型用 `target` 当阈值（availability 的 target 就是"成功率下限"）；
    `lte` 型用 `SLI_THRESHOLDS`（延迟 1500ms / 成本 2 元）。
    """

    if value is None:
        return None

    if slo.comparison == COMPARISON_GTE:
        return 1 if value >= slo.target else 0

    threshold = SLI_THRESHOLDS.get(slo.name)
    if threshold is None:
        raise KeyError(f"{slo.name} 是 lte 型但没登记阈值：补 SLI_THRESHOLDS")
    return 1 if value <= threshold else 0


def evaluate(slo: SLO, samples: Iterable[float | None]) -> float | None:
    """按"达标占比"给 SLI。没有有效样本时返回 None，**不返回 1.0**。

    返回 1.0 会让"没有数据"看起来像"完美"，而这正是监控最经典的骗局：
    链路挂了 → 没有日志 → 可用性 100%。
    """

    graded = [compliance(slo, value) for value in samples]
    valid = [item for item in graded if item is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


@dataclass(frozen=True)
class BudgetStatus:
    """误差预算的一次快照。字段名刻意与 SRE 常用语对齐，便于对外讲。"""

    slo: SLO
    observed: float
    total: int
    allowed_bad: float
    bad: float
    remaining: float
    exhausted: bool
    burn_rate: float
    elapsed: float

    @property
    def remaining_ratio(self) -> float:
        if self.allowed_bad <= 0:
            return 0.0 if self.bad > 0 else 1.0
        return max(0.0, 1.0 - self.bad / self.allowed_bad)

    def line(self) -> str:
        return (
            f"{self.slo.name:<14} SLI={self.observed:.3f} target={self.slo.target:.2f} "
            f"bad={self.bad:.0f}/{self.allowed_bad:.1f} "
            f"剩余={self.remaining_ratio * 100:.1f}% burn={self.burn_rate:.2f}x"
        )


def error_budget(
    slo: SLO,
    observed: float,
    window_total: int,
    *,
    elapsed: float = 1.0,
) -> BudgetStatus:
    """剩余误差预算。

    `elapsed` 是"窗口已经过了多少"（0–1）。它的唯一用途是算 **burn rate**：
    同样的错误数，在窗口第 1 天发生和在最后 1 天发生，紧迫程度完全不同。
    预算余量本身**不**按 elapsed 折算——已消耗的额度不会因为时间过去而变多。
    """

    if window_total <= 0:
        raise ValueError("window_total 必须为正：没有样本就没有预算可言")
    if not 0.0 < elapsed <= 1.0:
        raise ValueError(f"elapsed 必须在 (0, 1]，收到 {elapsed}")

    allowed_bad = window_total * (1.0 - slo.target)
    bad = window_total * (1.0 - observed)
    burn = (bad / allowed_bad / elapsed) if allowed_bad > 0 else math.inf

    return BudgetStatus(
        slo=slo,
        observed=observed,
        total=window_total,
        allowed_bad=allowed_bad,
        bad=bad,
        remaining=max(0.0, allowed_bad - bad),
        exhausted=bad > allowed_bad,
        burn_rate=burn,
        elapsed=elapsed,
    )


# --------------------------------------------------------------------------- #
# 告警规则
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlertRule:
    """一条告警。

    `for_s` 是**持续时间**，不是装饰：没有它，一次 30 秒的抖动会叫醒值班，
    而人被误报叫醒三次之后就会把这个告警静音——那时真实故障也没人看。
    """

    name: str
    slo: str
    # 触发条件：SLI 低于该值（availability）或 burn rate 高于该值。
    threshold: float
    for_s: float
    severity: str = "warn"
    summary: str = ""


ALERT_RULES: tuple[AlertRule, ...] = (
    AlertRule(
        name="availability_burn_fast",
        slo="availability",
        threshold=5.0,  # burn rate 5x
        for_s=300.0,  # 连续 5 分钟
        severity="page",
        summary="可用性误差预算以 5 倍速消耗：28 天预算将在 5.6 天内烧完",
    ),
    AlertRule(
        name="latency_p95_burn",
        slo="latency_p95",
        threshold=3.0,
        for_s=600.0,
        severity="warn",
        summary="首 token 延迟达标率下滑：p95 逼近 1.5s",
    ),
    AlertRule(
        name="cost_per_task_burn",
        slo="cost_per_task",
        threshold=2.0,
        for_s=900.0,
        severity="warn",
        summary="单任务成本超预算：先查是否 fallback 到贵档或缓存失效",
    ),
)


def alert_fires(rule: AlertRule, series: Sequence[tuple[float, float]]) -> bool:
    """`series` 是 `(timestamp_s, value)` 升序。条件**连续**满足 `for_s` 才触发。

    连续判定用"条件成立区间的跨度"，不是"有几个点满足"：
    采样间隔不定时按点数判会把 10 分钟的空窗当成 1 秒。
    """

    if not series:
        return False

    start: float | None = None
    for at, value in series:
        breached = value >= rule.threshold
        if breached and start is None:
            start = at
        elif not breached:
            start = None

    if start is None:
        return False
    # 区间必须真的跨过 for_s，且最后一个点仍在越线状态。
    return (series[-1][0] - start) >= rule.for_s


def series_from_eval(payload: dict[str, object]) -> list[dict[str, float | None]]:
    """从一期 `*-eval.json` 取出四个维度，供报告与告警使用。

    读的是**已经落盘的汇总**（`by_difficulty`），不重算：
    重算会让"评测时的数"与"报告里的数"出现两个来源，而它们本来就该是同一份。
    """

    rows: list[dict[str, float | None]] = []
    by_difficulty = payload.get("by_difficulty")
    if not isinstance(by_difficulty, dict):
        return rows

    for level, block in sorted(by_difficulty.items()):
        if not isinstance(block, dict):
            continue
        rows.append(
            {
                "difficulty": level,
                "success_rate": block.get("success_rate"),
                "latency_p95_ms": block.get("latency_p95_ms"),
                "first_token_p95_ms": block.get("first_token_p95_ms"),
                "cost_per_task": block.get("cost_per_task"),
                "usage_estimated_ratio": block.get("usage_estimated_ratio"),
                "judge_score": block.get("judge_score"),
            }
        )
    return rows

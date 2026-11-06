"""评测指标聚合层（W3 D1）。

**这一层不做任何 LLM 调用，也不计时。** 它只做一件事：
把一批 `EvalResult` 折成 `TaskMetrics`。输入全部来自 `RequestLog` 与 `expect` 断言，
输出是能直接塞进 `evals/history/*.json` 的结构。

三条硬约束（对应 `docs/w3/metric-spec.md`）
------------------------------------------
1. `success_rate` 的分母是 `passed + failed`。`interrupted`（上游 5xx/限流/鉴权）
   与 `skipped`（本档位没测）**都不进分母**——把厂商故障算成"模型答错"，会让团队
   去调 prompt 而不是去看上游状态页。
2. 成本只用 `usage_estimated=False` 的样本求均值。估算值混进实测均值，得到的数
   既不是实测也不是上界，是个没有意义的数。
3. 样本不足时返回 `None`，不返回 `0.0`。"没有样本"和"全错"在趋势图上是两件事，
   把前者画成后者会凭空造出一次塌陷。

`judge_score` 本周恒为 `None`（W4 才接 Judge）。**不许填 0.0**。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "CaseMetrics",
    "CaseStatus",
    "EvalResult",
    "RagasScores",
    "TaskMetrics",
    "aggregate",
    "by_difficulty",
    "by_kind",
    "by_tag",
    "percentile",
]

# p95 至少 5 个样本才计算：3 个点算 p95 只是把最大值改了个名字。
_MIN_P95_SAMPLES = 5


class CaseStatus(StrEnum):
    """一条 case 的终态。分母口径全看它（metric-spec §1）。"""

    PASSED = "passed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"

    @property
    def counts_toward_success(self) -> bool:
        """只有 passed/failed 进 success_rate 分母。"""

        return self in (CaseStatus.PASSED, CaseStatus.FAILED)


@dataclass(slots=True)
class CaseMetrics:
    """单条 case 的原始观测值。字段与 `RequestLog` 一一对应，不做二次加工。"""

    latency_ms: float | None = None
    first_token_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # `None` 表示"没有成本数据"（离线夹具不计价），和 `0.0`（免费/零用量）不同。
    cost: float | None = None
    currency: str | None = None

    usage_estimated: bool = False
    retry_count: int = 0
    error_code: str | None = None
    trace_id: str | None = None
    span_id: str | None = None

    # 工具层：attempts 含失败的调用，errors 是其中子集。
    tool_calls: int = 0
    tool_errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def has_log(self) -> bool:
        """有没有拿到 `RequestLog`。

        离线档的 skipped case 没有日志；有日志但 `latency_ms is None` 说明
        handler 没挂上——那是埋点坏了，不是"这条很快"。
        """

        return self.latency_ms is not None

    @classmethod
    def from_request_log(cls, payload: Mapping[str, Any] | None) -> CaseMetrics:
        """从 `RequestLog.as_dict()` 的 JSON 还原。

        对缺失字段一律给默认值而不是抛错：日志字段随版本增删（W1 的 18 项 →
        W2 的 20 项），读取方必须能读旧日志，否则 W11 的历史曲线一改结构就断。
        """

        if not payload:
            return cls()

        return cls(
            latency_ms=_as_float(payload.get("latency_ms")),
            first_token_ms=_as_float(payload.get("first_token_ms")),
            prompt_tokens=_as_int(payload.get("prompt_tokens")),
            completion_tokens=_as_int(payload.get("completion_tokens")),
            cost=_as_float(payload.get("cost")),
            currency=_as_str(payload.get("currency")),
            usage_estimated=bool(payload.get("usage_estimated", False)),
            retry_count=_as_int(payload.get("retry_count")),
            error_code=_as_str(payload.get("error_code")),
            trace_id=_as_str(payload.get("trace_id")),
            span_id=_as_str(payload.get("span_id")),
        )


@dataclass(frozen=True, slots=True)
class RagasScores:
    """RAGAS 三指标（W4 D2）。
        - faithfulness（忠实度）
        - answer_relevancy（答案相关性）
        - context_precision（上下文精确度）三个指标

    **每个字段都可能是 `None`，且 `None` 与 0.0 含义不同**：
    - `None`：这条 case 根本算不了（没有 references、答案为空）→ 不进均值
    - `0.0`：算了，结果是一条都没被支撑 → 进均值

    把前者当 0 会让"数据集里有一半 case 没引用"看起来像"faithfulness 掉到 0.5"。

    `method` 记录**计算方式**（`lexical` / `llm`）。同一个数字由 LLM 判定得来
    还是由词面匹配得来，可信度差一个档；报告里不带这个字段，读者无法判断
    "0.72 的 faithfulness" 到底值多少。
    """

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None

    method: str = "lexical"
    claims: int = 0
    contexts: int = 0

    @property
    def has_any(self) -> bool:
        return any(
            value is not None
            for value in (
                self.faithfulness,
                self.answer_relevancy,
                self.context_precision,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "method": self.method,
            "claims": self.claims,
            "contexts": self.contexts,
        }


@dataclass(slots=True)
class EvalResult:
    """一条 case 的评测结果：状态 + 观测值 + 失败原因。

    `categories` 由 D5 的 `classify_failures` 填；聚合层只做计数，
    不在这里判断"这算不算幻觉"（判定规则要能单独被测试）。
    """

    case_id: str
    status: CaseStatus
    kind: str = "chat"
    difficulty: int = 1
    tags: tuple[str, ...] = ()
    metrics: CaseMetrics = field(default_factory=CaseMetrics)
    failures: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    detail: str | None = None

    # W4 D2 起：有 references 的 case 才有值。`None` 表示"这条算不了"。
    ragas: RagasScores | None = None

    @property
    def passed(self) -> bool:
        return self.status is CaseStatus.PASSED


@dataclass(slots=True)
class TaskMetrics:
    """一批 case 的聚合指标。字段名与 `evals/history/*.json` 的 metrics 块对齐。"""

    total: int = 0
    passed: int = 0
    failed: int = 0
    interrupted: int = 0
    skipped: int = 0

    success_rate: float | None = None
    interrupted_rate: float | None = None
    skip_rate: float | None = None

    tool_calls: int = 0
    tool_errors: int = 0
    tool_error_rate: float | None = None

    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    first_token_p50_ms: float | None = None
    first_token_p95_ms: float | None = None

    # 分位的样本数：p95 只有 3 个样本时，读者有权知道。
    latency_samples: int = 0
    first_token_samples: int = 0

    cost_total: float | None = None
    cost_per_task: float | None = None
    currency: str | None = None
    usage_estimated_ratio: float = 0.0

    total_tokens: int = 0

    # W4 D1 填：judge 加权均分（1–5）。没开 judge 或全部解析失败时仍是 None。
    judge_score: float | None = None
    judge_samples: int = 0

    # W4 D2 填：RAGAS 三指标，**只对能算的样本取均值**。
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    ragas_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        """落盘用的扁平字典。

        故意不含 `judge_score=None` 之外的任何兜底值：写进 history 的必须是
        "真实观测或明确的 None"，绝不能是 0 或 "-"。
        """

        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "interrupted": self.interrupted,
            "skipped": self.skipped,
            "success_rate": self.success_rate,
            "interrupted_rate": self.interrupted_rate,
            "skip_rate": self.skip_rate,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tool_error_rate": self.tool_error_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_samples": self.latency_samples,
            "first_token_p50_ms": self.first_token_p50_ms,
            "first_token_p95_ms": self.first_token_p95_ms,
            "first_token_samples": self.first_token_samples,
            "cost_total": self.cost_total,
            "cost_per_task": self.cost_per_task,
            "currency": self.currency,
            "usage_estimated_ratio": self.usage_estimated_ratio,
            "total_tokens": self.total_tokens,
            "judge_score": self.judge_score,
            "judge_samples": self.judge_samples,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "ragas_samples": self.ragas_samples,
        }


# --------------------------------------------------------------------------- #
# 分位
# --------------------------------------------------------------------------- #


def percentile(
    values: Sequence[float], q: float, *, min_samples: int = 1
) -> float | None:
    """线性插值分位（与 `bench_baseline.py` 同一算法）。

    自己实现而不是引 numpy：评测层不想为 10 行数学拖一个依赖，
    而且 W1 的基线表用的是这个算法，换实现就等于换口径。

    `min_samples` 未达标时返回 `None`——宁可显示 "-"，也不显示一个
    由 3 个点算出来、却写着 p95 的数字。
    """

    if not values or len(values) < max(min_samples, 1):
        return None

    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q 必须在 [0, 1]，实际 {q}")

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #


def aggregate(results: Iterable[EvalResult]) -> TaskMetrics:
    """把一批结果折成 `TaskMetrics`。

    幂等、无副作用、不看时间——同一批结果调两次必然得到同一个数，
    否则"可复现"这条 Gate 就是空的。
    """

    items = list(results)
    metrics = TaskMetrics(total=len(items))

    if not items:
        return metrics

    latencies: list[float] = []
    first_tokens: list[float] = []
    ragas_rows: list[RagasScores] = []
    exact_costs: list[float] = []
    currencies: set[str] = set()
    estimated = 0
    logged = 0

    for item in items:
        if item.status is CaseStatus.PASSED:
            metrics.passed += 1
        elif item.status is CaseStatus.FAILED:
            metrics.failed += 1
        elif item.status is CaseStatus.INTERRUPTED:
            metrics.interrupted += 1
        else:
            metrics.skipped += 1

        metrics.tool_calls += item.metrics.tool_calls
        metrics.tool_errors += item.metrics.tool_errors
        metrics.total_tokens += item.metrics.total_tokens

        if item.ragas is not None and item.ragas.has_any:
            ragas_rows.append(item.ragas)

        if not item.metrics.has_log:
            continue

        logged += 1
        latencies.append(_non_negative(item.metrics.latency_ms))

        if item.metrics.first_token_ms is not None:
            first_tokens.append(_non_negative(item.metrics.first_token_ms))

        if item.metrics.usage_estimated:
            estimated += 1
        elif item.metrics.cost is not None:
            exact_costs.append(item.metrics.cost)

        # 错误路径的 RequestLog 不带价目，`currency` 落的是 dataclass 默认值 USD
        # （而真正计价的模型是 CNY）。若把它算进币种集合，一批里只要有一条 429，
        # 就会误报「多币种」并让整次评测失败。错误样本不参与币种判定。
        if item.metrics.currency and not item.metrics.error_code:
            currencies.add(item.metrics.currency)

    scored = metrics.passed + metrics.failed
    metrics.success_rate = (metrics.passed / scored) if scored else None
    metrics.interrupted_rate = metrics.interrupted / metrics.total
    metrics.skip_rate = metrics.skipped / metrics.total

    if metrics.tool_calls:
        metrics.tool_error_rate = metrics.tool_errors / metrics.tool_calls

    # 延迟分位覆盖所有拿到日志的样本（含中断）。中断的慢请求也是用户真实感受到的，
    # 把它排除会让"上游抖了"看起来像"延迟变好了"。样本数同时落盘，读者自行判断。
    metrics.latency_samples = len(latencies)
    metrics.first_token_samples = len(first_tokens)
    metrics.latency_p50_ms = percentile(latencies, 0.5)
    metrics.latency_p95_ms = percentile(latencies, 0.95, min_samples=_MIN_P95_SAMPLES)
    metrics.first_token_p50_ms = percentile(first_tokens, 0.5)
    metrics.first_token_p95_ms = percentile(
        first_tokens, 0.95, min_samples=_MIN_P95_SAMPLES
    )

    if exact_costs:
        metrics.cost_total = sum(exact_costs)
        metrics.cost_per_task = metrics.cost_total / len(exact_costs)

    if len(currencies) > 1:
        raise ValueError(
            f"同一批结果出现多币种 {sorted(currencies)}；评测层不做汇率换算"
            "（换算属于 W11 成本看板）"
        )

    metrics.currency = next(iter(currencies), None)
    metrics.usage_estimated_ratio = (estimated / logged) if logged else 0.0

    # RAGAS：分母是"算出值的样本数"，不是总数。
    metrics.ragas_samples = len(ragas_rows)
    metrics.faithfulness = _mean_of(ragas_rows, "faithfulness")
    metrics.answer_relevancy = _mean_of(ragas_rows, "answer_relevancy")
    metrics.context_precision = _mean_of(ragas_rows, "context_precision")

    return metrics


def by_difficulty(results: Iterable[EvalResult]) -> dict[int, TaskMetrics]:
    """按难度分档聚合。样本为 0 的档位不出现（不是 0 分）。"""

    buckets: dict[int, list[EvalResult]] = {}

    for item in results:
        buckets.setdefault(item.difficulty, []).append(item)

    return {level: aggregate(group) for level, group in sorted(buckets.items())}


def by_kind(results: Iterable[EvalResult]) -> dict[str, TaskMetrics]:
    """按调用形态（chat / stream / tool_call / error）聚合。

    为什么要分开：`latency_p95` 把"一次性返回"和"逐帧吐字"混在一起算一个 p95，
    两种形态的延迟本来就不是一个分布（流式的 total latency 天然含整段生成时间）。
    混算出来的 p95 既不代表流式体验，也不代表同步调用——它只是两个分布的和。
    """

    buckets: dict[str, list[EvalResult]] = {}

    for item in results:
        buckets.setdefault(item.kind, []).append(item)

    return {kind: aggregate(group) for kind, group in sorted(buckets.items())}


def by_tag(results: Iterable[EvalResult]) -> dict[str, TaskMetrics]:
    """按 tag 聚合。一条 case 有多个 tag 时，会同时计入每个 tag 的分母——
    所以各 tag 的 total 之和大于总 case 数，这是特性不是 bug（报告里要注明）。
    """

    buckets: dict[str, list[EvalResult]] = {}

    for item in results:
        for tag in item.tags:
            buckets.setdefault(tag, []).append(item)

    return {tag: aggregate(group) for tag, group in sorted(buckets.items())}


def _mean_of(rows: Sequence[RagasScores], field_name: str) -> float | None:
    """按字段取均值，跳过 None。全 None 时返回 None（不是 0.0）。"""

    values = [
        value
        for row in rows
        if (value := getattr(row, field_name)) is not None
    ]

    if not values:
        return None

    return round(sum(values) / len(values), 4)


# --------------------------------------------------------------------------- #
# 小工具：日志字段容错
# --------------------------------------------------------------------------- #


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0

    if isinstance(value, int):
        return value

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_str(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value)

    # 日志里 request_id/trace_id 缺失会被替换成 "-"（as_dict 的行为），
    # 读回来必须还原成 None，否则统计"有多少条没有 trace"会永远得 0。
    return None if text in ("", "-") else text


def _non_negative(value: float | None) -> float:
    if value is None:
        return 0.0

    return max(0.0, float(value))

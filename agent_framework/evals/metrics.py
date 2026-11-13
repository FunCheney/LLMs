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

    # 断言全部通过。进分母。
    PASSED = "passed"

    # 断言不通过（模型答错 / 工具报错 / 格式不符）。进分母。
    FAILED = "failed"

    # 上游 5xx、限流、鉴权失败导致没跑成。**不进分母**：
    # 把它算成答错，会让团队去调 prompt 而不是去看厂商状态页。
    INTERRUPTED = "interrupted"

    # 本档位没测（离线档跳过需要真 key 的 case）。**不进分母**，
    # 但 `skip_rate` 要落盘——高跳过率下的 100% 成功率没有意义。
    SKIPPED = "skipped"

    @property
    def counts_toward_success(self) -> bool:
        """只有 passed/failed 进 success_rate 分母。"""

        return self in (CaseStatus.PASSED, CaseStatus.FAILED)


@dataclass(slots=True)
class CaseMetrics:
    """单条 case 的原始观测值。字段与 `RequestLog` 一一对应，不做二次加工。"""

    # 整条请求的墙钟耗时（ms）：从发起上游调用到收完最后一个字节。
    # `None` 不是"很快"，是"没拿到 RequestLog"（离线跳过、埋点没挂上）。
    latency_ms: float | None = None

    # 第一个 token 到达的耗时（ms）。流式档的体验核心；同步档等于整段返回时间。
    # 沿用 W1 的 `first_token_ms` 定义，W4 的 judge 调用也用它统计"评审本身多慢"。
    first_token_ms: float | None = None

    # 上游计费口径的输入 token（含 system prompt、历史消息、judge 的 rubric 全文）。
    # 是上游回传值还是本地估算，由下面的 `usage_estimated` 单独标记，不在这里混。
    prompt_tokens: int = 0

    # 上游计费口径的输出 token。含被截断的部分——`finish_reason=length` 时它等于
    # max_tokens，读者不能把它当成"答案就这么长"。
    completion_tokens: int = 0

    # `None` 表示"没有成本数据"（离线夹具不计价），和 `0.0`（免费/零用量）不同。
    cost: float | None = None

    # 计价币种，来自 `models.yaml` 的 `pricing.currency`（本项目 CNY）。
    # 聚合时只统计非错误样本：错误日志带的是 dataclass 默认值 USD，
    # 混进来会让一批里出现"多币种"并让整次评测失败。
    currency: str | None = None

    # token 数是不是本地估出来的（上游没回 usage，例如部分兼容层或中断的流）。
    # True 的样本**不进成本均值**：估算值与实测值混在一起求的均值，
    # 既不是实测也不是上界。占比写进 `TaskMetrics.usage_estimated_ratio`。
    usage_estimated: bool = False

    # 这条请求被我们自己重发过几次（重试 / fallback 到备用模型）。0 = 一次成功。
    retry_count: int = 0

    # 归一化后的错误码（`UPSTREAM_ERROR` / `RATE_LIMITED` / `SERVER_BUSY` ...）。
    # 它是 `classify_failures` 码表的唯一输入，所以必须是码而不是异常类名——
    # 离线档与在线档的异常可能来自不同进程，日志才是共同语言。
    error_code: str | None = None

    # 与 trace 对齐的标识。日志里缺失会被写成 "-"，读回来还原成 None，
    # 否则"有多少条没有 trace"永远统计成 0。
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

    # 答案的陈述里，有多少能被 context 支撑（分母是陈述数，不是字数）。
    # 衡量"有没有编"，是三个指标里唯一直接对冲幻觉的那个。
    faithfulness: float | None = None

    # 答案落回问题所关心内容的程度。衡量"有没有答偏"，与 factfulness 无关：
    # 一篇完全正确但答非所问的答案，这里会很低而 `faithfulness` 仍可很高。
    answer_relevancy: float | None = None

    # 给出的 context 里真正相关的比例（1 - 噪声率）。衡量"召回端有没有灌水"，
    # 是检索层唯一能被 RAGAS 三指标覆盖到的一环。
    context_precision: float | None = None

    # 计算方式：`lexical`（词面匹配，零成本可复算）/ `llm`（走 gateway 判蕴含）。
    # 同一个 0.72，这两种方式的可信度差一个档，报告里不写就没法解读。
    method: str = "lexical"

    # `faithfulness` 的分子分母规模：判定了几条陈述、给了几段 context。
    # 只有 2 条陈述的 1.0 和 20 条陈述的 1.0 不是同一件事。
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

    # 用例标识，与 golden jsonl 的 `id` 同源。跨周对比趋势时靠它对上同一条 case。
    case_id: str

    # 终态。分母口径全看它（见 `CaseStatus.counts_toward_success`）。
    status: CaseStatus

    # 调用形态（chat / stream / tool_call / error），对应 golden 的 `kind`。
    # 延迟必须按它分组看：流式与同步的延迟本来就不是一个分布。
    kind: str = "chat"

    # 难度分档 1/2/3（1=单轮直答，2=需一次检索或工具，3=多步或多源引用）。
    # 用来回答"是难题变差了还是简单题掉队了"，只看总分看不出来。
    difficulty: int = 1

    # 自由标签（research / citation / adversarial ...）。一条 case 可带多个，
    # 按 tag 聚合时分母会重复计数，这是特性不是 bug。
    tags: tuple[str, ...] = ()

    # 观测值（延迟/token/成本/错误码）。全部来自 `RequestLog`，不在这里二次计算。
    metrics: CaseMetrics = field(default_factory=CaseMetrics)

    # 断言的失败文案，一条失败一个元素。措辞会被 `classify_failures` 当输入，
    # 所以改文案等于改分类结果，不能随手改。
    failures: tuple[str, ...] = ()

    # 失败分类（`FailureCategory` 的值）。由分类器填，聚合层只做计数——
    # "这算不算幻觉"的判定必须能单独被测试，不能散在聚合逻辑里。
    categories: tuple[str, ...] = ()

    # 人类可读的补充说明（截断的原文、异常栈首行）。**不进任何指标**，
    # 只在排查单条 case 时看。
    detail: str | None = None

    # W4 D2 起：有 references 的 case 才有值。`None` 表示"这条算不了"。
    ragas: RagasScores | None = None

    @property
    def passed(self) -> bool:
        return self.status is CaseStatus.PASSED


@dataclass(slots=True)
class TaskMetrics:
    """一批 case 的聚合指标。字段名与 `evals/history/*.json` 的 metrics 块对齐。"""

    # 样本总数 = passed + failed + interrupted + skipped。四项之和恒等于它。
    total: int = 0
    passed: int = 0
    failed: int = 0

    # 上游 5xx / 限流 / 鉴权失败。单列而不计入分母：厂商抖一下不是模型答错。
    interrupted: int = 0

    # 本档位没测（离线档跳过了需要真 key 的 case）。也不进分母。
    skipped: int = 0

    # `passed / (passed + failed)`。样本为 0 时是 `None` 而不是 `0.0`——
    # "没测"和"全错"在趋势图上是两件事。
    success_rate: float | None = None

    # 中断占比、跳过占比。二者与 `success_rate` 互补，用来判断"绿"是不是因为
    # 大量样本被跳过——只看 success_rate 会把这种情况误读成"系统很稳"。
    interrupted_rate: float | None = None
    skip_rate: float | None = None

    # 工具层：总调用次数、其中报错的次数（子集）、报错占比。
    tool_calls: int = 0
    tool_errors: int = 0
    tool_error_rate: float | None = None

    # 端到端延迟分位（ms）。覆盖所有拿到日志的样本，**含中断的慢请求**：
    # 把它排除会让"上游抖了"看起来像"延迟变好了"。
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None

    # 首 token 延迟分位（ms）。比端到端更能反映交互体感，两者要分开看。
    first_token_p50_ms: float | None = None
    first_token_p95_ms: float | None = None

    # 分位的样本数：p95 只有 3 个样本时，读者有权知道。
    latency_samples: int = 0
    first_token_samples: int = 0

    # 成本：整批总花费、单任务均值（只算 `usage_estimated=False` 的样本）、币种。
    cost_total: float | None = None
    cost_per_task: float | None = None
    currency: str | None = None

    # 估算成本样本占"有日志样本"的比例。>0 时报告必须显著标注"部分成本为估算"，
    # 否则 W11 的成本看板会把它当实测值用。
    usage_estimated_ratio: float = 0.0

    # 全批 token 总量（输入 + 输出），估算样本也算进来——它是容量度量，不是计费度量。
    total_tokens: int = 0

    # W4 D1 填：judge 加权均分（1–5）。没开 judge 或全部解析失败时仍是 None。
    judge_score: float | None = None

    # 参与 judge 均分的样本数（= 解析成功的条数）。与 `total` 的差额就是
    # "judge 链路坏掉的条数"，这个差额突然变大是故障信号。
    judge_samples: int = 0

    # W4 D2 填：RAGAS 三指标，**只对能算的样本取均值**。
    # 三个字段互相正交，要一起看：faithfulness 高而 context_precision 低，
    # 说明"没编但召回的料不相关"——问题在检索层，不在提示词。
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None

    # 参与 RAGAS 均值的样本数（三个指标各自的非 None 条数可能不同，
    # 这里取 `has_any` 的条数）。远小于 `total` 说明数据集缺 references。
    ragas_samples: int = 0

    # -- W9 D5 填：Trajectory（过程）层 ---------------------------------------
    # 这一层回答"怎么做到的"。缺了它，"多 Agent 成功率 +8%"无法归因——
    # 分不清是分工对了、工具选对了，还是只是多花 token 多试了几次。
    # 口径与取值规则见 `evals/scripts/trajectory_eval.py` 的模块 docstring。
    #
    # **谁会填它们**：`agent_paradigm_ab.py` + `trajectory_eval.py` 这条链，
    # 因为只有规划器会产出 `RunTrace`。54 条 golden 是 chat / tool_call 形态，
    # 跑的是 `LLMGateway`，压根没有轨迹——所以 `run_eval` 落盘里这四个字段是
    # `null`，不是 0。**"字段存在"不等于"有数"**，看 history 时别读错。
    tool_selection_acc: float | None = None
    invalid_call_rate: float | None = None
    # 平均工具执行次数（**不是**角色轮数：多 Agent 的一次交接可能含多次调用）。
    steps_used: float | None = None
    path_repeat_rate: float | None = None

    # 各指标的实际样本数。均值不带样本数，读者就无从判断它可不可信。
    trajectory_samples: int = 0

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
            "tool_selection_acc": self.tool_selection_acc,
            "invalid_call_rate": self.invalid_call_rate,
            "steps_used": self.steps_used,
            "path_repeat_rate": self.path_repeat_rate,
            "trajectory_samples": self.trajectory_samples,
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

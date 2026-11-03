"""失败自动分类（W3 D5）：规则先行，LLM 兜底。

为什么规则必须先行
------------------
`UPSTREAM` 和 `HALLUCINATION` 混在一起，是一份评测报告最致命的错误：前者说明
**今天厂商在发版**，后者说明**我们的系统在编造**。团队看到"失败率 30%"去调 prompt，
查两天发现是对方 503——这类事故在别人的复盘里叫"归因错误"，在这里只是分类器少写了一个分支。

所以：能从 `error_code` / 字段判定的，绝不交给模型判。规则判不了才轮到 W4 的 judge，
而 judge 的结论必须带 `category=judge` 标记，让读者知道这条分类是"猜"的。

判据顺序（先命中先返回）
------------------------
1. `error_code` 存在 → 上游/超时（这类根本不该进 `failed`，但兜一道）
2. `tool_errors > 0` → `TOOL_ERROR`
3. `finish_reason=length` 或成本超预算 → `BUDGET`
4. 帧序列不符 / 结构化断言不符 → `FORMAT`
5. 有 `references` 但内容断言不成立 → `HALLUCINATION`（无引用支撑）
6. 其余 → `ASSERTION`（确实答错了，但归因需要人看）
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from enum import StrEnum

from evals.metrics import CaseStatus, EvalResult

__all__ = [
    "CALLER_CODES",
    "RELIABILITY_CODES",
    "FailureCategory",
    "classify",
    "classify_result",
    "is_quality_failure",
    "is_reliability_code",
    "summarize",
]


class FailureCategory(StrEnum):
    """失败归因。前四个可由字段判定，后三个需要看语义。"""

    UPSTREAM = "upstream"  # 5xx / 限流 / 鉴权 —— 可靠性问题，不是模型笨
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"  # arguments_error / 未声明调用
    BUDGET = "budget"  # 超出 per-task 预算，或 finish_reason=length
    FORMAT = "format"  # 输出结构不合 expect（帧序列、必填字段）
    HALLUCINATION = "hallucination"  # 有 references 但答案缺乏支撑
    RETRIEVAL_MISS = "retrieval_miss"  # W5 接检索后才有输入，先留枚举
    REQUEST = "request"  # 我们的请求/配置写错了（400/能力不支持），不是模型答错
    ASSERTION = "assertion"  # 断言不成立且无法机械归因 → 交给人

    @property
    def is_reliability(self) -> bool:
        """厂商侧可靠性（上游/超时）。不计入"模型质量"结论。"""

        return self in (FailureCategory.UPSTREAM, FailureCategory.TIMEOUT)

    @property
    def counts_against_model(self) -> bool:
        """这条失败该不该算进"模型答得好不好"。

        三档：厂商抖动（UPSTREAM/TIMEOUT）和"我们自己传错了"（REQUEST）
        都不该算到模型头上。区别很重要——`SERVER_BUSY` 是我们自己的限流器吐的，
        把它算成模型答错，会让人去调 prompt。
        """

        return self not in (
            FailureCategory.UPSTREAM,
            FailureCategory.TIMEOUT,
            FailureCategory.REQUEST,
        )


# 错误码 → 分类。用 code 而不是异常类型：分类器只读日志，不 import 异常树
# （离线档与在线档的错误可能来自不同进程，日志才是唯一共同输入）。
#
# **这里是唯一一份码表**（W4 D5 案例 3）。之前 `run_eval` 与 `classify_failures`
# 各有一份，10 个已知错误码里有 6 个在两处判定不同：`SERVER_BUSY`（我们自己的
# 限流器吐的 503）在 run_eval 里被算成"质量失败"，在分类器里是"上游问题"。
# 结果是一批被限流的请求把 success_rate 拉低，而报告说"模型答错了"。
RELIABILITY_CODES = {
    "UPSTREAM_AUTH",
    "UPSTREAM_ERROR",
    "UPSTREAM_TIMEOUT",
    "RATE_LIMITED",
    "SERVER_BUSY",
}

_TIMEOUT_CODES = {"UPSTREAM_TIMEOUT", "SERVER_BUSY"}

# 我方问题：请求本身不合法、能力不匹配、配置写错。既不是厂商故障，也不是模型答错。
CALLER_CODES = {
    "BAD_REQUEST",
    "CAPABILITY_UNSUPPORTED",
    "CONFIG_ERROR",
    "TASK_NOT_FOUND",
    "CONTENT_FILTERED",
}


def is_reliability_code(code: str | None) -> bool:
    """这个错误码是否属于"厂商侧故障"（→ `interrupted`，不进成功率分母）。

    未知码返回 `True`（按可靠性处理）：宁可少算一次"模型答错"，
    也不要把一个新出现的厂商故障记成质量退化。
    """

    if not code:
        return False

    return code not in CALLER_CODES


# 出现在 failures 文本里的结构性关键词。匹配的是 runner 自己生成的断言文案，
# 不是模型输出——所以用子串匹配是安全的（文案由本项目控制）。
_FORMAT_MARKERS = ("帧序列", "finish_reason 期望")
_BUDGET_MARKERS = ("预算", "budget")


def classify(
    *,
    error_code: str | None = None,
    failures: Sequence[str] = (),
    tool_errors: int = 0,
    finish_reason: str | None = None,
    cost: float | None = None,
    budget: float | None = None,
    has_references: bool = False,
) -> list[FailureCategory]:
    """把一条失败拆成一个或多个归因。

    允许返回多个：一条 case 可以既工具报错又断言不成立（工具错了，后面的断言当然也不成立）。
    只报第一个会掩盖根因链。
    """

    categories: list[FailureCategory] = []

    if error_code:
        if is_reliability_code(error_code):
            categories.append(
                FailureCategory.TIMEOUT
                if error_code in _TIMEOUT_CODES
                else FailureCategory.UPSTREAM
            )
        else:
            # 我方请求/配置问题：不是厂商故障，也不该算模型答错。
            categories.append(FailureCategory.REQUEST)
        # 上游失败时，下游断言必然不成立。此时再报 FORMAT/HALLUCINATION 是纯噪声，
        # 而且会让"上游抖动"被误读成"模型编造"。
        return categories

    if tool_errors > 0:
        categories.append(FailureCategory.TOOL_ERROR)

    if finish_reason == "length" or (budget is not None and cost is not None and cost > budget):
        categories.append(FailureCategory.BUDGET)

    joined = " | ".join(failures)

    if any(marker in joined for marker in _FORMAT_MARKERS):
        categories.append(FailureCategory.FORMAT)

    # 有引用锚点却没命中内容断言 → 看起来像"编了没出处的答案"。
    # 注意：这只是**线索**，最终判定归 W4 judge；报告的措辞写"疑似无引用支撑"。
    if has_references and any(
        keyword in joined for keyword in ("文本里没有", "文本里没有出现")
    ):
        categories.append(FailureCategory.HALLUCINATION)

    if not categories:
        categories.append(FailureCategory.ASSERTION)

    return categories


def classify_result(
    result: EvalResult,
    *,
    budget: float | None = None,
    has_references: bool = False,
    finish_reason: str | None = None,
) -> list[FailureCategory]:
    """对一条 `EvalResult` 归类。通过/跳过的返回空列表。"""

    if result.status is not CaseStatus.FAILED:
        return []

    return classify(
        error_code=result.metrics.error_code,
        failures=result.failures,
        tool_errors=result.metrics.tool_errors,
        finish_reason=finish_reason,
        cost=result.metrics.cost,
        budget=budget,
        has_references=has_references,
    )


def summarize(items: Iterable[Sequence[FailureCategory]]) -> dict[str, int]:
    """按分类计数，供报告使用。空输入返回空 dict（不是全 0 的假表）。"""

    counter = Counter(category.value for group in items for category in group)

    return dict(sorted(counter.items()))


def is_quality_failure(categories: Sequence[FailureCategory]) -> bool:
    """这条失败是否该算进"模型质量"讨论。

    上游/超时一律不算——它们反映可靠性。报告里两类分开列，避免一条 502
    把成功率拉低 20 个点之后，团队跑去改 prompt。
    """

    return bool(categories) and any(item.counts_against_model for item in categories)

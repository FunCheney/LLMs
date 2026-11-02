"""指标层的边界用例（W3 D1 / Gate「指标可定义」）。

这一组的重点是**分母**：`success_rate` 什么时候返回 `None`、什么时候把某类结果排除在分母外。
模型答错和上游挂掉混在一起统计，是评测系统最常见也最难自查的错误——因为报告上
只有一个百分比，看不出它是由哪批样本算出来的。
"""

from __future__ import annotations

import pytest

from evals.metrics import (
    CaseMetrics,
    CaseStatus,
    EvalResult,
    aggregate,
    by_difficulty,
    by_tag,
    percentile,
)


def _result(
    case_id: str,
    status: CaseStatus = CaseStatus.PASSED,
    *,
    latency_ms: float | None = 100.0,
    first_token_ms: float | None = None,
    cost: float | None = 0.001,
    usage_estimated: bool = False,
    difficulty: int = 1,
    tags: tuple[str, ...] = (),
    tool_calls: int = 0,
    tool_errors: int = 0,
    currency: str | None = "CNY",
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        status=status,
        difficulty=difficulty,
        tags=tags,
        metrics=CaseMetrics(
            latency_ms=latency_ms,
            first_token_ms=first_token_ms,
            cost=cost,
            currency=currency,
            usage_estimated=usage_estimated,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            prompt_tokens=10,
            completion_tokens=5,
        ),
    )


# --------------------------------------------------------------------------- #
# 空批次
# --------------------------------------------------------------------------- #


def test_empty_batch_returns_none_not_zero() -> None:
    """没有样本时返回 None。返回 0.0 会让趋势图画出一次不存在的塌陷。"""

    metrics = aggregate([])

    assert metrics.total == 0
    assert metrics.success_rate is None
    assert metrics.latency_p50_ms is None
    assert metrics.cost_per_task is None
    assert metrics.usage_estimated_ratio == 0.0
    assert metrics.judge_score is None


def test_rates_on_empty_batch_do_not_divide_by_zero() -> None:
    """空批次的比率一律 None：分母为 0 时"占比"没有定义，写 0.0 会被读成"零中断"。"""

    metrics = aggregate([])

    assert metrics.interrupted_rate is None
    assert metrics.skip_rate is None
    assert metrics.tool_error_rate is None


# --------------------------------------------------------------------------- #
# 全失败
# --------------------------------------------------------------------------- #


def test_all_failed_gives_zero_success_rate() -> None:
    """全错是 0.0，不是 None——这是"真的都答错了"，必须能在曲线上看见。"""

    metrics = aggregate([_result(f"c{i}", CaseStatus.FAILED) for i in range(3)])

    assert metrics.total == 3
    assert metrics.failed == 3
    assert metrics.success_rate == 0.0


def test_all_failed_keeps_latency_samples() -> None:
    """答错也有日志：延迟分位不能因为"失败"就把样本抽掉。"""

    metrics = aggregate(
        [_result(f"c{i}", CaseStatus.FAILED, latency_ms=100.0 + i) for i in range(5)]
    )

    assert metrics.latency_samples == 5
    assert metrics.latency_p50_ms == pytest.approx(102.0)


# --------------------------------------------------------------------------- #
# 分母口径：interrupted / skipped 不进 success_rate
# --------------------------------------------------------------------------- #


def test_interrupted_excluded_from_success_denominator() -> None:
    """这条是整个指标层的核心：上游 502 不是"模型答错"。

    5 条里 2 条通过、1 条答错、2 条被上游打断 → 12 条里 3 条进分母，成功率 2/3。
    若把中断算成失败，成功率会变成 2/5，团队会去调 prompt 而不是查上游。
    """

    metrics = aggregate(
        [
            _result("ok-1"),
            _result("ok-2"),
            _result("bad-1", CaseStatus.FAILED),
            _result("int-1", CaseStatus.INTERRUPTED, cost=None, currency=None),
            _result("int-2", CaseStatus.INTERRUPTED, cost=None, currency=None),
        ]
    )

    assert metrics.passed == 2
    assert metrics.failed == 1
    assert metrics.interrupted == 2
    assert metrics.success_rate == pytest.approx(2 / 3)
    assert metrics.interrupted_rate == pytest.approx(2 / 5)


def test_skipped_never_counts_as_passed() -> None:
    """离线档跑不了的语义题必须记 skipped，记成通过就是自欺。"""

    metrics = aggregate(
        [
            _result("ok-1"),
            _result("skipped-1", CaseStatus.SKIPPED, cost=None, currency=None),
        ]
    )

    assert metrics.passed == 1
    assert metrics.skipped == 1
    assert metrics.success_rate == 1.0
    assert metrics.skip_rate == pytest.approx(0.5)


def test_all_interrupted_has_none_success_rate() -> None:
    """全部被上游打断 → success_rate 为 None（没测到），而不是 0.0（都答错了）。"""

    metrics = aggregate(
        [
            _result(f"int{i}", CaseStatus.INTERRUPTED, cost=None, currency=None)
            for i in range(4)
        ]
    )

    assert metrics.success_rate is None
    assert metrics.interrupted_rate == 1.0


# --------------------------------------------------------------------------- #
# 成本：估算样本不进均值
# --------------------------------------------------------------------------- #


def test_estimated_usage_excluded_from_cost_average() -> None:
    """估算成本混进实测均值，得到的是个既非实测也非上界的数。"""

    metrics = aggregate(
        [
            _result("exact-1", cost=0.002),
            _result("exact-2", cost=0.004),
            _result("guess-1", cost=9.9, usage_estimated=True),
        ]
    )

    assert metrics.cost_total == pytest.approx(0.006)
    assert metrics.cost_per_task == pytest.approx(0.003)
    assert metrics.usage_estimated_ratio == pytest.approx(1 / 3)


def test_all_estimated_leaves_cost_per_task_none() -> None:
    """全是估算 → cost_per_task 为 None，并让 usage_estimated_ratio=1.0 说明原因。"""

    metrics = aggregate(
        [_result(f"g{i}", cost=0.5, usage_estimated=True) for i in range(3)]
    )

    assert metrics.cost_per_task is None
    assert metrics.cost_total is None
    assert metrics.usage_estimated_ratio == 1.0


def test_missing_cost_is_not_zero_cost() -> None:
    """离线夹具不计价：cost=None。它不该被当成 0 元算进均值，把均值稀释掉。"""

    metrics = aggregate(
        [
            _result("priced", cost=0.01),
            _result("unpriced", cost=None, currency=None),
        ]
    )

    assert metrics.cost_total == pytest.approx(0.01)
    assert metrics.cost_per_task == pytest.approx(0.01)


def test_mixed_currency_raises() -> None:
    """币种不一致直接报错：评测层不做汇率换算。"""

    with pytest.raises(ValueError, match="多币种"):
        aggregate([_result("a", currency="CNY"), _result("b", currency="USD")])


# --------------------------------------------------------------------------- #
# 延迟：分位口径与样本数
# --------------------------------------------------------------------------- #


def test_percentile_matches_bench_baseline_algorithm() -> None:
    """线性插值：与 W1 `bench_baseline.py` 同一算法，换实现就等于换口径。"""

    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0.5) == pytest.approx(25.0)
    assert percentile(values, 0.0) == pytest.approx(10.0)
    assert percentile(values, 1.0) == pytest.approx(40.0)
    assert percentile(values, 0.95) == pytest.approx(38.5)


def test_p95_requires_five_samples() -> None:
    """4 个点算 p95 只是把最大值改个名字，所以返回 None。"""

    metrics = aggregate([_result(f"c{i}", latency_ms=float(10 + i)) for i in range(4)])

    assert metrics.latency_p50_ms is not None
    assert metrics.latency_p95_ms is None
    assert metrics.latency_samples == 4


def test_p95_computed_at_five_samples() -> None:
    metrics = aggregate([_result(f"c{i}", latency_ms=float(10 * (i + 1))) for i in range(5)])

    assert metrics.latency_p95_ms == pytest.approx(48.0)


def test_first_token_only_counts_stream_samples() -> None:
    """阻塞式调用没有 first_token：不能把它的 latency 混进首 token 样本池。"""

    metrics = aggregate(
        [
            _result("blocking", latency_ms=900.0, first_token_ms=None),
            _result("s1", latency_ms=120.0, first_token_ms=20.0),
            _result("s2", latency_ms=140.0, first_token_ms=30.0),
        ]
    )

    assert metrics.first_token_samples == 2
    assert metrics.first_token_p50_ms == pytest.approx(25.0)
    assert metrics.latency_samples == 3


def test_negative_latency_clamped_not_propagated() -> None:
    """时钟回拨/脏数据不该把分位拉到负数，也不该静默变成 NaN。"""

    metrics = aggregate([_result("bad", latency_ms=-5.0)])

    assert metrics.latency_p50_ms == 0.0


# --------------------------------------------------------------------------- #
# 工具层
# --------------------------------------------------------------------------- #


def test_tool_error_rate_uses_attempts_as_denominator() -> None:
    """分母是尝试次数（含失败），写成 attempts+errors 会让错误率凭空减半。"""

    metrics = aggregate([_result("t1", tool_calls=4, tool_errors=1)])

    assert metrics.tool_calls == 4
    assert metrics.tool_errors == 1
    assert metrics.tool_error_rate == pytest.approx(0.25)


def test_no_tool_calls_yields_none_rate() -> None:
    assert aggregate([_result("c1")]).tool_error_rate is None


# --------------------------------------------------------------------------- #
# 分组
# --------------------------------------------------------------------------- #


def test_by_difficulty_omits_absent_levels() -> None:
    """样本为 0 的难度档不出现——不是 0 分。"""

    grouped = by_difficulty(
        [
            _result("d1-a", difficulty=1),
            _result("d3-a", CaseStatus.FAILED, difficulty=3),
            _result("d3-b", difficulty=3),
        ]
    )

    assert set(grouped) == {1, 3}
    assert grouped[1].success_rate == 1.0
    assert grouped[3].success_rate == pytest.approx(0.5)


def test_by_tag_counts_case_in_every_tag() -> None:
    """多 tag 的 case 会同时进各 tag 分母，各 tag total 之和大于总数是特性。"""

    grouped = by_tag(
        [
            _result("a", tags=("research", "citation")),
            _result("b", tags=("citation",)),
        ]
    )

    assert grouped["research"].total == 1
    assert grouped["citation"].total == 2


# --------------------------------------------------------------------------- #
# 日志字段容错
# --------------------------------------------------------------------------- #


def test_from_request_log_tolerates_missing_and_dash_fields() -> None:
    """`as_dict()` 会把空 trace_id 换成 "-"，读回来必须还原成 None。"""

    metrics = CaseMetrics.from_request_log(
        {
            "latency_ms": 12.5,
            "first_token_ms": None,
            "cost": 0,
            "currency": "CNY",
            "trace_id": "-",
            "span_id": "-",
            "usage_estimated": False,
        }
    )

    assert metrics.latency_ms == pytest.approx(12.5)
    assert metrics.trace_id is None
    assert metrics.span_id is None
    assert metrics.cost == 0.0


def test_from_request_log_handles_empty_payload() -> None:
    metrics = CaseMetrics.from_request_log(None)

    assert metrics.has_log is False
    assert metrics.total_tokens == 0


def test_bool_is_not_coerced_into_tokens() -> None:
    """`True` 在 Python 里是 int 的子类，不防会变成 1 个 token。"""

    metrics = CaseMetrics.from_request_log({"prompt_tokens": True})

    assert metrics.prompt_tokens == 0


def test_aggregate_is_idempotent() -> None:
    """同一批结果聚合两次必须一致，否则"可复现"这条 Gate 是空的。"""

    results = [_result(f"c{i}", latency_ms=float(i * 7)) for i in range(6)]

    assert aggregate(results).as_dict() == aggregate(results).as_dict()

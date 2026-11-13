"""W11 D1：SLO、误差预算、告警边界。

这组用例守四条：

1. **延迟用 p95 不用均值**——均值会把"1% 请求慢了 10 秒"抹平；
2. **成本要能区分 estimated**——估算占比高时数字不可信（W1 Day5 教训）；
3. **告警有持续时间**——单点抖动不该叫醒人；
4. **没有数据不等于完美**——`evaluate` 在无有效样本时返回 `None` 而不是 `1.0`。

第 4 条是监控最经典的骗局：链路挂了 → 没有日志 → 可用性 100%。
"""

from __future__ import annotations

import pytest

from ops.slo import (
    ALERT_RULES,
    SLI_THRESHOLDS,
    SLO_BY_NAME,
    SLOS,
    AlertRule,
    alert_fires,
    compliance,
    error_budget,
    evaluate,
    percentile,
    series_from_eval,
)

# --------------------------------------------------------------------------- #
# 口径
# --------------------------------------------------------------------------- #


def test_three_slos_cover_the_three_dod_dimensions() -> None:
    assert {slo.name for slo in SLOS} == {"availability", "latency_p95", "cost_per_task"}


def test_every_slo_declares_its_discipline() -> None:
    """口径必须写清边界（含不含 4xx、缺字段怎么办）。

    不写的话半年后没人知道这条 SLO 算不算 skipped，而两种算法都能自圆其说。
    """

    for slo in SLOS:
        assert slo.discipline.strip(), f"{slo.name} 缺少口径说明"


def test_percentile_is_nearest_rank() -> None:
    """与 `evals/metrics.py` 同口径：取真实存在的点，不插值。

    插值会造出"p95 = 1.4327ms"这种数据里不存在的值，
    而值班要看的是"最慢那批真实请求有多慢"。
    """

    samples = [float(n) for n in range(1, 101)]  # 1..100
    assert percentile(samples, 0.95) == 95.0
    assert percentile(samples, 1.0) == 100.0
    assert percentile(samples, 0.5) == 50.0
    assert percentile([], 0.95) is None


def test_percentile_rejects_bad_quantile() -> None:
    with pytest.raises(ValueError, match="q 必须在"):
        percentile([1.0], 0.0)


def test_latency_uses_p95_not_mean() -> None:
    """**反向用例**：长尾占比一旦跨过 5%，p95 从 10ms 直接跳到 5000ms，而均值几乎不动。

    顺带钉住一个容易误读的边界：nearest-rank 在**恰好** 5% 慢时 p95 仍是 10ms
    （rank=95，落在第 95 个样本上）。所以"p95 好看"不等于"没有长尾"——
    它只说明长尾还没到 5%。
    """

    one_slow = [10.0] * 99 + [5000.0]
    assert percentile(one_slow, 0.95) == 10.0  # 1% 慢，p95 看不见

    five_slow = [10.0] * 95 + [5000.0] * 5
    assert percentile(five_slow, 0.95) == 10.0  # 恰好 5%，仍在边界内

    six_slow = [10.0] * 94 + [5000.0] * 6
    assert percentile(six_slow, 0.95) == 5000.0  # 跨过 5%，立刻显形
    assert sum(six_slow) / len(six_slow) < 400  # 均值仍然"好看"


def test_compliance_respects_comparison_direction() -> None:
    availability = SLO_BY_NAME["availability"]
    latency = SLO_BY_NAME["latency_p95"]

    assert compliance(availability, 0.995) == 1
    assert compliance(availability, 0.97) == 0
    # lte 型：阈值来自 SLI_THRESHOLDS，不是 target(0.95)
    assert compliance(latency, SLI_THRESHOLDS["latency_p95"] - 1) == 1
    assert compliance(latency, SLI_THRESHOLDS["latency_p95"] + 1) == 0
    # 缺失字段不参与，而不是判失败
    assert compliance(latency, None) is None


def test_missing_samples_are_not_counted_as_healthy() -> None:
    """没有有效样本 → None，**不是** 1.0。

    链路挂了 → 没有日志 → 可用性 100%：这是监控最经典的骗局。
    """

    latency = SLO_BY_NAME["latency_p95"]
    assert evaluate(latency, [None, None]) is None
    # 部分缺失时只用有效样本，并在分母里体现（不把缺失算成失败）
    assert evaluate(latency, [None, 100.0, 2000.0]) == 0.5


def test_cost_estimated_ratio_is_read_not_guessed() -> None:
    """成本口径要能读回 `usage_estimated_ratio`，否则"成本下降"说不清是优化还是没数据。"""

    payload = {
        "by_difficulty": {
            "3": {
                "success_rate": 1.0,
                "latency_p95_ms": 800.0,
                "first_token_p95_ms": 300.0,
                "cost_per_task": 0.42,
                "usage_estimated_ratio": 0.35,
            }
        }
    }
    rows = series_from_eval(payload)
    assert rows[0]["usage_estimated_ratio"] == 0.35
    assert rows[0]["cost_per_task"] == 0.42


# --------------------------------------------------------------------------- #
# 误差预算
# --------------------------------------------------------------------------- #


def test_budget_exhausts_exactly_when_bad_exceeds_allowance() -> None:
    slo = SLO_BY_NAME["availability"]  # target 0.99 → 1000 条允许坏 10 条

    ok = error_budget(slo, 0.995, 1000)
    assert ok.bad == pytest.approx(5.0)
    assert ok.allowed_bad == pytest.approx(10.0)
    assert not ok.exhausted
    assert ok.remaining_ratio == pytest.approx(0.5)

    boundary = error_budget(slo, 0.99, 1000)
    assert not boundary.exhausted  # 正好用满不算耗尽

    blown = error_budget(slo, 0.985, 1000)
    assert blown.exhausted
    assert blown.remaining_ratio == 0.0


def test_burn_rate_normalizes_by_elapsed() -> None:
    """同样的错误数，在窗口第 1 天发生比最后 1 天严重得多。"""

    slo = SLO_BY_NAME["availability"]
    early = error_budget(slo, 0.98, 100, elapsed=0.1)  # 窗口才过 10%
    late = error_budget(slo, 0.98, 100, elapsed=1.0)

    assert early.burn_rate > late.burn_rate
    assert early.burn_rate == pytest.approx(late.burn_rate * 10)


def test_budget_rejects_empty_window() -> None:
    with pytest.raises(ValueError, match="window_total"):
        error_budget(SLO_BY_NAME["availability"], 1.0, 0)


def test_elapsed_must_be_within_window() -> None:
    with pytest.raises(ValueError, match="elapsed"):
        error_budget(SLO_BY_NAME["availability"], 1.0, 10, elapsed=1.5)


# --------------------------------------------------------------------------- #
# 告警：持续时间
# --------------------------------------------------------------------------- #


def test_alert_requires_the_declared_duration() -> None:
    rule = AlertRule(name="t", slo="availability", threshold=5.0, for_s=300.0)

    # 越线但只持续了 240s → 不响
    assert not alert_fires(rule, [(0.0, 9.0), (240.0, 9.0)])
    # 持续 300s → 响
    assert alert_fires(rule, [(0.0, 9.0), (300.0, 9.0)])


def test_single_spike_does_not_page() -> None:
    """单点抖动不该叫醒人。被误报叫醒三次之后，这个告警就会被静音。"""

    rule = AlertRule(name="t", slo="availability", threshold=5.0, for_s=300.0)
    assert not alert_fires(rule, [(0.0, 0.1), (60.0, 9.0), (120.0, 0.1)])


def test_recovery_resets_the_clock() -> None:
    """越线 → 恢复 → 又越线：第二段不能接上第一段的时长，否则"断续抖动"会被当成"持续故障"。"""

    rule = AlertRule(name="t", slo="availability", threshold=5.0, for_s=300.0)
    series = [(0.0, 9.0), (200.0, 9.0), (240.0, 0.1), (280.0, 9.0), (320.0, 9.0)]

    assert not alert_fires(rule, series)


def test_all_rules_have_duration_and_runbook_hint() -> None:
    """每条规则都得有 `for`：没有持续时间的告警等于把抖动直接转成呼机。"""

    for rule in ALERT_RULES:
        assert rule.for_s > 0, f"{rule.name} 缺少持续时间"
        assert rule.slo in SLO_BY_NAME, f"{rule.name} 指向了不存在的 SLO"

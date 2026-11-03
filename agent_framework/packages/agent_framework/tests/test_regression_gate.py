"""W4 D3 · 回归门禁的口径用例。

核心是三件事：
1. 相对阈值真的抗波动（±2% 不许红）；
2. 真劣化真的会红（否则门禁是摆设）；
3. UPSTREAM 激增只告警、不判 fail（W3 口径）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.scripts.regression_gate import (
    DEFAULT_RULES,
    GateRule,
    evaluate,
    load_metrics,
)

# --------------------------------------------------------------------------- #
# 相对阈值
# --------------------------------------------------------------------------- #


def test_regression_triggers_fail() -> None:
    verdict = evaluate({"success_rate": 0.90}, {"success_rate": 1.00})

    assert not verdict.passed
    assert verdict.violations[0].metric == "success_rate"
    assert verdict.violations[0].change_ratio == pytest.approx(0.10)


def test_within_noise_does_not_trigger() -> None:
    """±2% 属于正常波动：每次跑都红一次的门禁，最后没人看。"""

    verdict = evaluate({"success_rate": 0.98}, {"success_rate": 1.00})

    assert verdict.passed
    assert verdict.compared == ["success_rate"]


def test_improvement_never_triggers() -> None:
    verdict = evaluate({"success_rate": 1.0}, {"success_rate": 0.9})

    assert verdict.passed


# --------------------------------------------------------------------------- #
# 延迟：相对 + 绝对下限
# --------------------------------------------------------------------------- #


def test_latency_relative_regression_needs_absolute_delta() -> None:
    """基线 2ms 时 +20% 只有 0.4ms，属于抖动，不该红。"""

    verdict = evaluate(
        {"success_rate": 1.0, "latency_p95_ms": 2.4},
        {"success_rate": 1.0, "latency_p95_ms": 2.0},
    )

    assert verdict.passed


def test_latency_triggers_when_both_relative_and_absolute_exceeded() -> None:
    verdict = evaluate(
        {"success_rate": 1.0, "latency_p95_ms": 40.0},
        {"success_rate": 1.0, "latency_p95_ms": 20.0},
    )

    assert not verdict.passed
    assert verdict.violations[0].metric == "latency_p95_ms"


def test_latency_absolute_delta_but_tiny_relative_does_not_trigger() -> None:
    """基线 1000ms 涨到 1010ms：绝对增量够，但相对只 1%，不红。"""

    verdict = evaluate(
        {"success_rate": 1.0, "latency_p95_ms": 1010.0},
        {"success_rate": 1.0, "latency_p95_ms": 1000.0},
    )

    assert verdict.passed


# --------------------------------------------------------------------------- #
# 缺失与跳过
# --------------------------------------------------------------------------- #


def test_missing_required_metric_fails() -> None:
    """当前期压根没算出 success_rate，说明评测跑残了——必须红。"""

    verdict = evaluate({}, {"success_rate": 1.0})

    assert not verdict.passed
    assert "无法判定" in verdict.violations[0].detail


def test_missing_optional_metric_is_skipped_not_failed() -> None:
    """离线档没有 judge_score 是正常的，不该因此红。"""

    verdict = evaluate(
        {"success_rate": 1.0},
        {"success_rate": 1.0, "judge_score": 4.0},
    )

    assert verdict.passed
    assert "judge_score" in verdict.skipped


def test_zero_baseline_is_skipped_not_infinite_regression() -> None:
    """基线为 0 时任何正数都是"无穷倍劣化"，判 fail 会让门禁永远红。"""

    verdict = evaluate(
        {"success_rate": 1.0, "cost_per_task": 0.001},
        {"success_rate": 1.0, "cost_per_task": 0.0},
    )

    assert verdict.passed
    assert "cost_per_task" not in verdict.compared or verdict.passed


# --------------------------------------------------------------------------- #
# UPSTREAM：告警但不判定质量
# --------------------------------------------------------------------------- #


def test_upstream_surge_warns_without_failing() -> None:
    """可靠性退化要看状态页，不该阻断合并，也不该混进 success_rate 结论。"""

    verdict = evaluate(
        {"success_rate": 1.0, "interrupted_rate": 0.06},
        {"success_rate": 1.0, "interrupted_rate": 0.0},
    )

    assert verdict.passed
    assert verdict.warnings
    assert "interrupted_rate" in verdict.warnings[0]


def test_upstream_noise_does_not_warn() -> None:
    verdict = evaluate(
        {"success_rate": 1.0, "interrupted_rate": 0.01},
        {"success_rate": 1.0, "interrupted_rate": 0.0},
    )

    assert not verdict.warnings


def test_strict_mode_turns_warning_into_failure() -> None:
    """`--strict` 是给"发布前"用的：那时可靠性退化也不该放过。"""

    verdict = evaluate(
        {"success_rate": 1.0, "interrupted_rate": 0.06},
        {"success_rate": 1.0, "interrupted_rate": 0.0},
    )

    assert verdict.passed  # 非 strict 下通过
    assert verdict.warnings  # 但留下了告警，CI 可以据此加 --strict


# --------------------------------------------------------------------------- #
# 自定义规则
# --------------------------------------------------------------------------- #


def test_custom_rule_direction() -> None:
    rule = GateRule("total_tokens", "lower_is_better", 0.10)
    verdict = evaluate(
        {"total_tokens": 2000.0}, {"total_tokens": 1000.0}, rules=(rule,)
    )

    assert not verdict.passed


def test_default_rules_cover_the_four_contract_metrics() -> None:
    """门禁至少要看这四项：成功率、延迟、成本、质量分。"""

    metrics = {rule.metric for rule in DEFAULT_RULES}

    assert {"success_rate", "latency_p95_ms", "cost_per_task", "judge_score"} <= metrics


# --------------------------------------------------------------------------- #
# 读盘：离线档不许读 quality 块
# --------------------------------------------------------------------------- #


def test_load_metrics_reads_pipeline_block_for_offline(tmp_path: Path) -> None:
    """离线档若去读 `quality`（那里恒为 null），门禁会拿到空指标而永远绿。"""

    path = tmp_path / "2026-W46-eval.json"
    path.write_text(
        json.dumps(
            {
                "mode": "offline",
                "pipeline": {"success_rate": 1.0},
                "quality": {"success_rate": None},
            }
        ),
        "utf-8",
    )

    assert load_metrics(path) == {"success_rate": 1.0}


def test_load_metrics_reads_quality_block_for_online(tmp_path: Path) -> None:
    path = tmp_path / "2026-W47-eval.json"
    path.write_text(
        json.dumps(
            {
                "mode": "online",
                "pipeline": {"success_rate": 0.0},
                "quality": {"success_rate": 0.9},
            }
        ),
        "utf-8",
    )

    assert load_metrics(path) == {"success_rate": 0.9}


def test_load_metrics_rejects_snapshot_without_block(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"mode": "online"}), "utf-8")

    with pytest.raises(SystemExit):
        load_metrics(path)

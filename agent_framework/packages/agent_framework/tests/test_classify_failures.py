"""失败分类器的规则测试（W3 D5 / Gate 7「UPSTREAM 与 HALLUCINATION 不混淆」）。

这组用例的重点不是"分得准"，而是**分得稳**：上游 5xx 被归成幻觉，会让团队
去改 prompt 而不是看状态页。一次这样的误判，比十次分类不准的代价都大。
"""

from __future__ import annotations

from classify_failures import (
    FailureCategory,
    classify,
    is_quality_failure,
    summarize,
)


def test_upstream_code_short_circuits_to_reliability() -> None:
    """有 error_code 就只报上游，不再往下判格式/幻觉。

    下游断言在 502 时必然不成立，继续分类只会制造"模型编造"的假象。
    """

    categories = classify(
        error_code="UPSTREAM_ERROR",
        failures=["文本里没有 'x'", "帧序列期望 [...]，实际 []"],
        has_references=True,
    )

    assert categories == [FailureCategory.UPSTREAM]
    assert is_quality_failure(categories) is False


def test_rate_limited_is_upstream_not_hallucination() -> None:
    assert classify(error_code="RATE_LIMITED") == [FailureCategory.UPSTREAM]


def test_timeout_code_is_timeout() -> None:
    assert classify(error_code="UPSTREAM_TIMEOUT") == [FailureCategory.TIMEOUT]


def test_tool_error_detected_from_counter() -> None:
    categories = classify(tool_errors=2, failures=["tool_calls 里没有 'search_web'"])

    assert FailureCategory.TOOL_ERROR in categories
    assert FailureCategory.HALLUCINATION not in categories


def test_length_finish_reason_is_budget() -> None:
    assert classify(finish_reason="length") == [FailureCategory.BUDGET]


def test_cost_over_budget_is_budget() -> None:
    categories = classify(cost=0.12, budget=0.05)

    assert categories == [FailureCategory.BUDGET]


def test_frame_mismatch_is_format() -> None:
    categories = classify(failures=["帧序列期望 ['text_delta']，实际 ['tool_call']"])

    assert categories == [FailureCategory.FORMAT]


def test_missing_content_with_references_is_hallucination_suspect() -> None:
    """有引用锚点却答不出内容 → 疑似无引用支撑（最终判定归 W4 judge）。"""

    categories = classify(
        failures=["文本里没有 '工具注解'（实际 '...'）"],
        has_references=True,
    )

    assert categories == [FailureCategory.HALLUCINATION]


def test_missing_content_without_references_is_assertion() -> None:
    """没有引用锚点的普通问答，答偏了就是答偏了，不该扣"幻觉"的帽子。"""

    categories = classify(failures=["文本里没有 '工具注解'"], has_references=False)

    assert categories == [FailureCategory.ASSERTION]


def test_unknown_failure_falls_back_to_assertion() -> None:
    """规则判不出来时宁可说"需要人看"，也不猜一个类别。"""

    assert classify(failures=["某个新加的断言"]) == [FailureCategory.ASSERTION]


def test_multiple_categories_keep_root_cause_chain() -> None:
    """工具报错 + 断言不成立要同时报：只留第一个会掩盖根因。"""

    categories = classify(
        tool_errors=1,
        failures=["文本里没有 'x'"],
        has_references=True,
    )

    assert FailureCategory.TOOL_ERROR in categories
    assert FailureCategory.HALLUCINATION in categories


def test_is_quality_failure_ignores_pure_reliability() -> None:
    assert is_quality_failure([FailureCategory.UPSTREAM]) is False
    assert is_quality_failure([FailureCategory.TIMEOUT]) is False
    assert is_quality_failure([FailureCategory.ASSERTION]) is True
    assert is_quality_failure([]) is False


def test_summarize_counts_and_sorts() -> None:
    counts = summarize(
        [
            [FailureCategory.UPSTREAM],
            [FailureCategory.FORMAT, FailureCategory.TOOL_ERROR],
            [FailureCategory.FORMAT],
        ]
    )

    assert counts == {"format": 2, "tool_error": 1, "upstream": 1}


def test_summarize_empty_returns_empty_dict() -> None:
    """空输入不留一行全 0 的假表。"""

    assert summarize([]) == {}

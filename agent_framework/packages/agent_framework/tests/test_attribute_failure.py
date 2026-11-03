"""W4 D4 · 四层归因的用例。

四类样本各命中一层，外加两条"证据不足"：
- `tool_error` 但没有 tool span
- `hallucination` 但没有 retrieval span（W5 前是常态）

这两条是最容易做错的地方——把"埋点缺失"判成"某一层坏了"，
团队会去改一个没坏的模块。
"""

from __future__ import annotations

import pytest

from evals.metrics import CaseMetrics, CaseStatus, EvalResult
from evals.scripts.attribute_failure import (
    Attribution,
    Layer,
    Signals,
    attribute,
    attribute_result,
    from_case_payload,
)
from evals.scripts.classify_failures import FailureCategory

# --------------------------------------------------------------------------- #
# 四层
# --------------------------------------------------------------------------- #


def test_retrieval_layer_when_retrieved_but_nothing_cited() -> None:
    attribution = attribute(
        [FailureCategory.HALLUCINATION],
        Signals(
            retrieval_spans=1,
            retrieved_ids=("doc-1", "doc-2"),
            cited_ids=(),
        ),
    )

    assert attribution.layer is Layer.RETRIEVAL
    assert attribution.confidence >= 0.8
    assert any("命中 0 条" in item for item in attribution.evidence)


def test_retrieval_layer_when_cited_but_unsupported() -> None:
    """引用了检索结果、结论仍无支撑 → 检索到的内容本身不含答案，仍在检索层。"""

    attribution = attribute(
        [FailureCategory.HALLUCINATION],
        Signals(retrieval_spans=1, retrieved_ids=("doc-1",), cited_ids=("doc-1",)),
    )

    assert attribution.layer is Layer.RETRIEVAL
    assert attribution.confidence == pytest.approx(0.6)


def test_prompt_layer_for_format_failure() -> None:
    attribution = attribute(
        [FailureCategory.FORMAT],
        Signals(finish_reason="stop", answer_chars=42),
    )

    assert attribution.layer is Layer.PROMPT
    assert "format" in " ".join(attribution.evidence)


def test_hallucination_without_recorded_ids_is_unknown() -> None:
    """跑了检索却没记录 doc id：埋点不全，不能判"模型编造"，也不能判"检索坏"。"""

    attribution = attribute(
        [FailureCategory.HALLUCINATION],
        Signals(retrieval_spans=2, retrieved_ids=(), cited_ids=()),
    )

    assert attribution.layer is Layer.UNKNOWN
    assert "retrieved_ids 为空" in " ".join(attribution.evidence)


def test_tool_layer_on_arguments_error() -> None:
    attribution = attribute(
        [FailureCategory.TOOL_ERROR],
        Signals(tool_spans=1, tool_arguments_error=1),
    )

    assert attribution.layer is Layer.TOOL
    assert attribution.confidence >= 0.9


def test_tool_layer_on_http_error() -> None:
    attribution = attribute(
        [FailureCategory.TOOL_ERROR],
        Signals(tool_spans=2, tool_http_error=1),
    )

    assert attribution.layer is Layer.TOOL
    assert attribution.confidence == pytest.approx(0.8)


def test_model_layer_on_upstream_code() -> None:
    attribution = attribute(
        [FailureCategory.UPSTREAM],
        Signals(error_code="RATE_LIMITED", retry_count=2, fallback_from="qwen3.7-plus"),
    )

    assert attribution.layer is Layer.MODEL
    assert attribution.confidence >= 0.9
    assert any("RATE_LIMITED" in item for item in attribution.evidence)


def test_model_layer_on_server_busy() -> None:
    attribution = attribute([], Signals(error_code="SERVER_BUSY"))

    assert attribution.layer is Layer.MODEL


# --------------------------------------------------------------------------- #
# 证据不足：不许硬猜
# --------------------------------------------------------------------------- #


def test_tool_error_without_span_is_unknown_not_tool() -> None:
    """分类说工具错，但 trace 里没有 tool span → 埋点缺失，不是工具坏了。"""

    attribution = attribute([FailureCategory.TOOL_ERROR], Signals(tool_spans=0))

    assert attribution.layer is Layer.UNKNOWN
    assert attribution.confidence == 0.0
    assert not attribution.actionable
    assert any("埋点缺失" in item for item in attribution.evidence)


def test_hallucination_without_retrieval_span_is_unknown_not_model() -> None:
    """W5 前没有检索：不能把"疑似无引用支撑"升级成"模型编造"。"""

    attribution = attribute([FailureCategory.HALLUCINATION], Signals())

    assert attribution.layer is Layer.UNKNOWN
    assert any("无法区分" in item for item in attribution.evidence)


def test_empty_categories_is_unknown() -> None:
    attribution = attribute([], Signals())

    assert attribution.layer is Layer.UNKNOWN
    assert attribution.fix


# --------------------------------------------------------------------------- #
# 与 EvalResult / 落盘结构对接
# --------------------------------------------------------------------------- #


def test_attribute_result_falls_back_to_metrics() -> None:
    """信号不完整时用 metrics 补：tool_calls 当 span 数，tool_errors 当参数错误数。"""

    result = EvalResult(
        case_id="gold-t05",
        status=CaseStatus.FAILED,
        categories=(FailureCategory.TOOL_ERROR.value,),
        metrics=CaseMetrics(tool_calls=1, tool_errors=1),
    )

    attribution = attribute_result(result)

    assert attribution.layer is Layer.TOOL


def test_attribute_result_keeps_explicit_signals() -> None:
    result = EvalResult(
        case_id="c1",
        status=CaseStatus.FAILED,
        categories=(FailureCategory.UPSTREAM.value,),
        metrics=CaseMetrics(error_code="RATE_LIMITED", retry_count=2),
    )

    attribution = attribute_result(result, Signals(error_code="RATE_LIMITED"))

    assert attribution.layer is Layer.MODEL
    assert "retry_count=2" in attribution.evidence


def test_from_case_payload_reads_snapshot_shape() -> None:
    categories, signals = from_case_payload(
        {
            "id": "gold-t05",
            "status": "failed",
            "categories": ["tool_error"],
            "metrics": {"tool_calls": 0, "tool_errors": 0, "retry_count": 0},
        }
    )

    assert categories == (FailureCategory.TOOL_ERROR,)
    assert signals.tool_spans == 0


def test_attribution_serializes_layer_as_string() -> None:
    attribution: Attribution = attribute([], Signals(error_code="UPSTREAM_ERROR"))
    payload = attribution.as_dict()

    assert payload["layer"] == "model"
    assert payload["fix"]

"""W4 D2 · RAGAS 三指标的边界用例。

重点不是"数值算得准不准"，而是**算不了的时候会不会装作算了**：
无 context、空答案、判定失败，三种情况都必须返回 None 而不是 0。
"""

from __future__ import annotations

import pytest

from evals.judge.ragas_metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
    score_case,
    split_claims,
)
from evals.metrics import CaseStatus, EvalResult, RagasScores, aggregate

# --------------------------------------------------------------------------- #
# 陈述切分
# --------------------------------------------------------------------------- #


def test_split_claims_drops_short_fragments() -> None:
    """`"是的。"` 这类片段不构成可核对的陈述，进分母只会稀释指标。"""

    claims = split_claims("是的。幂等键需要在重试之间保持不变。如下：")

    assert claims == ["幂等键需要在重试之间保持不变。"]


def test_split_claims_handles_empty_answer() -> None:
    assert split_claims("") == []
    assert split_claims("   ") == []


def test_split_claims_keeps_decimal_numbers_together() -> None:
    """英文句点不切分，否则 "0.5" 会被切成 "0" 和 "5"。"""

    claims = split_claims("退避系数取 0.5 是常见做法。")

    assert len(claims) == 1


# --------------------------------------------------------------------------- #
# faithfulness
# --------------------------------------------------------------------------- #


async def test_faithfulness_without_context_returns_none() -> None:
    """无 context 是"算不了"，不是 0 分——这条最容易被改坏。"""

    value, claims = await faithfulness("任意答案。", [])

    assert value is None
    assert claims == 0


async def test_faithfulness_with_blank_context_returns_none() -> None:
    value, _ = await faithfulness("任意答案。", ["   "])

    assert value is None


async def test_faithfulness_near_verbatim_claim_is_supported() -> None:
    """词面判定只能认"近似原文"的陈述，这是它能力上限，也是它的诚实边界。"""

    value, claims = await faithfulness(
        "MCP 是模型接入工具与数据的协议。",
        ["MCP 是一套模型接入工具与数据的协议。"],
    )

    assert value == pytest.approx(1.0)
    assert claims == 1


async def test_lexical_support_misses_paraphrase() -> None:
    """同义改写会被判无支撑 —— 所以 `method=lexical` 是**下界**，不是估计。

    这条用例锁住这个已知缺陷：哪天有人"修好"它，必须同时换 `LLMSupport`，
    而不是悄悄调低阈值（调低会让无关文本也命中，faithfulness 变成噪声）。
    """

    value, _ = await faithfulness(
        "幂等键需要在重试之间保持不变。",
        ["客户端每次请求带一个幂等键，重试时必须复用同一个幂等键。"],
    )

    assert value == 0.0


async def test_faithfulness_partially_supported() -> None:
    """两条陈述，一条有支撑 → 0.5，不是"全有或全无"。"""

    value, claims = await faithfulness(
        "MCP 是模型接入工具与数据的协议。服务端用布隆过滤器存幂等键。",
        ["MCP 是一套模型接入工具与数据的协议。"],
    )

    assert claims == 2
    assert value == pytest.approx(0.5)


async def test_dangling_reference_scores_much_lower_than_grounded_one() -> None:
    """反向验证：引用与答案无关时，faithfulness 必须明显掉下来。"""

    grounded, _ = await faithfulness(
        "MCP 是模型接入工具与数据的协议。",
        ["MCP 是一套模型接入工具与数据的协议。"],
    )
    dangling, _ = await faithfulness(
        "指数退避要加抖动，避免所有客户端同时重试。",
        ["Kubernetes 的 Pod 生命周期分为 Pending、Running、Succeeded。"],
    )

    assert grounded is not None and dangling is not None
    assert grounded > dangling
    assert dangling < 0.5


async def test_faithfulness_failed_judgement_returns_none_not_partial() -> None:
    """判定过程报错时不能产出"半截数字"：0.5 会看起来像"一半有支撑"。"""

    class Boom:
        name = "boom"

        async def supported(self, claim: str, contexts: list[str]) -> bool:
            raise RuntimeError("上游挂了")

    value, claims = await faithfulness(
        "第一条陈述足够长可以计入。第二条陈述也足够长。",
        ["context"],
        support=Boom(),
    )

    assert value is None
    assert claims == 2


# --------------------------------------------------------------------------- #
# answer_relevancy / context_precision
# --------------------------------------------------------------------------- #


async def test_relevancy_is_none_on_empty_answer_or_question() -> None:
    assert await answer_relevancy("", "MCP 是什么？") is None
    assert await answer_relevancy("答案", "   ") is None


async def test_relevancy_prefers_on_topic_answer() -> None:
    on_topic = await answer_relevancy(
        "MCP 是让模型接入外部工具与数据源的协议。", "MCP 是什么？"
    )
    off_topic = await answer_relevancy(
        "Kubernetes 的 Pod 是最小调度单元。", "MCP 是什么？"
    )

    assert on_topic is not None and off_topic is not None
    assert on_topic > off_topic


def test_context_precision_counts_relevant_only() -> None:
    value = context_precision(
        "MCP 是什么？",
        ["MCP 是一套模型接入工具与数据的协议。", "Kubernetes Pod 调度单元。"],
    )

    assert value == pytest.approx(0.5)


def test_context_precision_without_contexts_is_none() -> None:
    assert context_precision("MCP 是什么？", []) is None


# --------------------------------------------------------------------------- #
# score_case：一条 case 的整体口径
# --------------------------------------------------------------------------- #


async def test_score_case_without_references_has_no_faithfulness() -> None:
    scores = await score_case(question="MCP 是什么？", answer="一套协议。", references=[])

    assert scores is not None
    assert scores.faithfulness is None  # 无 context
    assert scores.context_precision is None
    assert scores.answer_relevancy is not None


async def test_score_case_returns_none_when_nothing_to_score() -> None:
    assert await score_case(question="q", answer="", references=[]) is None


async def test_score_case_records_method_and_counts() -> None:
    """method 必须落盘：同一个数字由 LLM 判定还是词面匹配得来，可信度不同。"""

    scores = await score_case(
        question="MCP 是什么？",
        answer="MCP 是模型接入工具与数据的协议。",
        references=["MCP 是一套模型接入工具与数据的协议。"],
    )

    assert scores is not None
    assert scores.method == "lexical"
    assert scores.contexts == 1
    assert scores.claims >= 1



# --------------------------------------------------------------------------- #
# 聚合：None 不进分母
# --------------------------------------------------------------------------- #


def test_aggregate_skips_none_ragas_values() -> None:
    """一条 0.0、一条 1.0、一条 None → 均值 0.5，分母 2。"""

    results = [
        EvalResult(
            case_id="a",
            status=CaseStatus.PASSED,
            ragas=RagasScores(faithfulness=0.0, method="lexical"),
        ),
        EvalResult(
            case_id="b",
            status=CaseStatus.PASSED,
            ragas=RagasScores(faithfulness=1.0, method="lexical"),
        ),
        EvalResult(
            case_id="c",
            status=CaseStatus.PASSED,
            ragas=RagasScores(faithfulness=None, method="lexical"),
        ),
    ]
    metrics = aggregate(results)

    # 第三条全 None（`has_any=False`）不进样本数：否则分母里塞满"算不了"的 case
    assert metrics.ragas_samples == 2
    assert metrics.faithfulness == pytest.approx(0.5)


def test_aggregate_ragas_all_none_is_none_not_zero() -> None:
    results = [
        EvalResult(
            case_id="a",
            status=CaseStatus.PASSED,
            ragas=RagasScores(answer_relevancy=0.4),
        )
    ]
    metrics = aggregate(results)

    assert metrics.faithfulness is None
    assert metrics.answer_relevancy == pytest.approx(0.4)


def test_aggregate_without_any_ragas_reports_none() -> None:
    metrics = aggregate([EvalResult(case_id="a", status=CaseStatus.PASSED)])

    assert metrics.ragas_samples == 0
    assert metrics.judge_score is None
    assert metrics.judge_samples == 0
    assert metrics.as_dict()["faithfulness"] is None

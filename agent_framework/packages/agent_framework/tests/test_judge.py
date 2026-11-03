"""W4 D1 · Judge 与一致性系数的边界用例。

这几条测的不是"能不能打分"，而是**打不了分时会不会假装打了**：
解析失败必须 None、缺维度必须 None、空样本必须报错而不是返回 0。
"""

from __future__ import annotations

import pytest

from evals.judge.llm_judge import (
    HeuristicJudge,
    JudgeCase,
    LLMJudge,
    build_judge,
)
from evals.judge.rubric import (
    DEFAULT_RUBRIC,
    Dimension,
    JudgeResult,
    JudgeScore,
    Rubric,
    RubricParseError,
    parse_judge_payload,
)
from evals.judge.stats import (
    bucket,
    distribution,
    mean_absolute_error,
    nominal_kappa,
    quadratic_weighted_kappa,
    spearman,
)

# --------------------------------------------------------------------------- #
# rubric 结构
# --------------------------------------------------------------------------- #


def test_rubric_requires_anchors_on_1_3_5() -> None:
    """缺锚的维度会在构造时炸：没锚的分数区间，模型一律给中间的 4 分。"""

    with pytest.raises(ValueError, match="缺少"):
        Dimension(
            key="fact",
            label="事实",
            weight=1.0,
            question="对不对",
            anchors={3: "还行"},
        )


def test_rubric_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="权重之和"):
        Rubric(
            name="bad",
            version="v0",
            dimensions=(
                Dimension("a", "A", 0.5, "q", {1: "a", 3: "b", 5: "c"}),
                Dimension("b", "B", 0.2, "q", {1: "a", 3: "b", 5: "c"}),
            ),
        )


def test_render_contains_every_anchor_and_the_json_shape() -> None:
    text = DEFAULT_RUBRIC.render()

    for dimension in DEFAULT_RUBRIC.dimensions:
        assert dimension.key in text

        for anchor in dimension.anchors.values():
            assert anchor in text

    # 解析器只认这个结构，prompt 里必须把它写出来
    assert '"scores"' in text and "rationale" in text


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #


def _payload(**overrides: object) -> str:
    scores = {
        "fact": {"score": 4, "rationale": "全部陈述成立"},
        "grounding": {"score": 3, "rationale": "一处找不到对应"},
        "coverage": {"score": 5, "rationale": "主次都答了"},
        "concision": {"score": 4, "rationale": "有一句可删"},
    }
    scores.update(overrides)

    import json

    return json.dumps({"scores": scores}, ensure_ascii=False)


def test_parse_ok() -> None:
    result = parse_judge_payload(_payload(), DEFAULT_RUBRIC, alias="judge-x")

    assert result.available
    assert result.judge_alias == "judge-x"
    assert result.score_of("fact") == 4
    assert result.overall(DEFAULT_RUBRIC) == pytest.approx(
        4 * 0.4 + 3 * 0.25 + 5 * 0.2 + 4 * 0.15
    )


def test_parse_accepts_fenced_and_prefixed_output() -> None:
    """模型爱把 JSON 包在 ``` 里或前面加一句话；这是最高频的"不听话"，要容忍。"""

    fenced = "```json\n" + _payload() + "\n```"
    prefixed = "好的，评分如下：\n" + _payload()

    assert parse_judge_payload(fenced, DEFAULT_RUBRIC).available
    assert parse_judge_payload(prefixed, DEFAULT_RUBRIC).available


def test_parse_rejects_missing_dimension() -> None:
    import json

    broken = json.dumps({"scores": {"fact": {"score": 4}}}, ensure_ascii=False)

    with pytest.raises(RubricParseError, match="缺少维度"):
        parse_judge_payload(broken, DEFAULT_RUBRIC)


def test_parse_rejects_unknown_dimension() -> None:
    """多出来的维度通常意味着 prompt 与解析器版本错位，必须炸。"""

    with pytest.raises(RubricParseError, match="未定义"):
        parse_judge_payload(_payload(hallucination={"score": 1}), DEFAULT_RUBRIC)


def test_parse_rejects_decimal_and_out_of_range() -> None:
    with pytest.raises(RubricParseError, match="小数"):
        parse_judge_payload(_payload(fact={"score": 4.5}), DEFAULT_RUBRIC)

    with pytest.raises(RubricParseError, match="超出"):
        parse_judge_payload(_payload(fact={"score": 7}), DEFAULT_RUBRIC)


def test_parse_rejects_non_json() -> None:
    with pytest.raises(RubricParseError):
        parse_judge_payload("我觉得这个答案还不错，给 4 分", DEFAULT_RUBRIC)


def test_unavailable_result_has_no_overall() -> None:
    """解析失败 → overall 必须是 None，绝不能补一个中间分。"""

    result = JudgeResult(parse_error="boom")

    assert not result.available
    assert result.overall(DEFAULT_RUBRIC) is None


def test_missing_dimension_makes_overall_none() -> None:
    """少一个维度时不做"按已有维度重归一"——那会让只答一维的评审拿到 5 分。"""

    partial = JudgeResult(
        scores=(
            JudgeScore("fact", 5, ""),
            JudgeScore("grounding", 5, ""),
            JudgeScore("coverage", 5, ""),
        )
    )

    assert partial.overall(DEFAULT_RUBRIC) is None


# --------------------------------------------------------------------------- #
# LLMJudge：走 gateway，不裸 HTTP
# --------------------------------------------------------------------------- #


class _Registry:
    def __init__(self, client: object) -> None:
        self._client = client

    def get(self, alias: str) -> object:
        return self._client

    def get_model(self, alias: str) -> object:
        return None


class _Response:
    def __init__(self, text: str) -> None:
        from agent_framework.llm.types import FinishReason, Usage

        self.text = text
        self.finish_reason = FinishReason.STOP
        self.tool_calls: list[object] = []
        self.usage = Usage(input_tokens=700, output_tokens=120)
        self.provider = "mock"


class _Outcome:
    def __init__(self, text: str) -> None:
        self.response = _Response(text)
        self.alias = "judge-alias"
        self.retry_count = 0
        self.fallback_from = None


class _Gateway:
    """最小的 gateway 替身：记录 ChatCall，回一段固定文本。"""

    def __init__(self, text: str) -> None:
        from agent_framework.config import load_catalog

        self.catalog = load_catalog()
        self.calls: list[object] = []
        self._text = text

    async def chat(self, call: object) -> _Outcome:
        self.calls.append(call)

        return _Outcome(self._text)


async def test_llm_judge_uses_strong_route_and_temperature_zero() -> None:
    gateway = _Gateway(_payload())
    judge = LLMJudge(gateway)  # type: ignore[arg-type]

    result = await judge.score(
        JudgeCase(id="c1", question="MCP 是什么？", answer="一套接入协议")
    )

    assert judge.alias == gateway.catalog.routes["strong"].primary
    assert result.available

    call = gateway.calls[0]
    assert call.model == judge.alias  # type: ignore[attr-defined]
    assert call.opts["temperature"] == 0  # type: ignore[attr-defined]
    # 判分不复用被测的 task 路由，避免"judge 用便宜档打分"
    assert call.task == "research"  # type: ignore[attr-defined]


async def test_llm_judge_degrades_on_unparseable_output() -> None:
    """模型没按 JSON 答 → parse_error + None，且**不抛异常**。"""

    judge = LLMJudge(_Gateway("这个答案挺好的，4 分吧"))  # type: ignore[arg-type]

    result = await judge.score(JudgeCase(id="c1", question="q", answer="a"))

    assert not result.available
    assert result.parse_error
    assert result.overall(DEFAULT_RUBRIC) is None
    # 失败也要留下 token 账单：否则 judge 的成本会被系统性低估
    assert result.prompt_tokens == 700


async def test_heuristic_judge_never_claims_to_be_a_llm() -> None:
    judge = HeuristicJudge()

    result = await judge.score(
        JudgeCase(id="c1", question="MCP 是什么？", answer="MCP 是模型接入工具与数据的协议")
    )

    assert result.judge_alias == "heuristic-keyword"
    assert result.available


async def test_heuristic_judge_scores_empty_answer_lowest() -> None:
    result = await HeuristicJudge().score(
        JudgeCase(id="c1", question="MCP 是什么？", answer="  ")
    )

    assert {item.score for item in result.scores} == {1}


def test_build_judge_live_requires_gateway() -> None:
    """在线档缺 gateway 必须报错，不许静默退化成关键词匹配。"""

    with pytest.raises(ValueError, match="gateway"):
        build_judge(live=True, gateway=None)

    assert isinstance(build_judge(live=False), HeuristicJudge)


# --------------------------------------------------------------------------- #
# 一致性系数
# --------------------------------------------------------------------------- #


def test_kappa_is_one_on_perfect_agreement() -> None:
    values = [1, 2, 3, 4, 5, 3, 4]

    assert quadratic_weighted_kappa(values, values) == pytest.approx(1.0)
    buckets = [bucket(v) for v in values]

    assert nominal_kappa(buckets, buckets) == pytest.approx(1.0)


def test_kappa_is_zero_when_both_sides_use_one_value() -> None:
    """全给 5 分不是"一致性"，是没有区分度——返回 0 而不是 1。"""

    assert quadratic_weighted_kappa([5] * 6, [5] * 6) == 0.0


def test_weighted_kappa_punishes_far_offsets_more() -> None:
    """差 1 分和差 4 分不能同罪，这是选二次加权而非名义 kappa 的原因。"""

    # 两边都必须有变化：单边恒定会让 pe 退化到 1，kappa 在数学上无定义（返回 0）
    human = [5, 4, 3, 2, 1, 3]
    near = quadratic_weighted_kappa(human, [4, 4, 3, 2, 1, 3])
    far = quadratic_weighted_kappa(human, [1, 4, 3, 2, 1, 3])

    assert near > far > -1.0


def test_spearman_handles_ties() -> None:
    """人工标注里同一个分出现多次是常态，并列必须取平均秩。"""

    rho = spearman([1, 2, 2, 3, 4], [1.0, 2.0, 2.2, 2.8, 4.5])

    assert rho > 0.9


def test_spearman_returns_zero_when_one_side_is_constant() -> None:
    assert spearman([1, 2, 3, 4], [3.0, 3.0, 3.0, 3.0]) == 0.0


def test_stats_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="长度"):
        quadratic_weighted_kappa([1, 2], [1])

    with pytest.raises(ValueError, match="长度"):
        nominal_kappa(["low"], ["low", "high"])


def test_empty_sample_raises_instead_of_zero() -> None:
    """"没有样本"和"完全不一致"是两件事，不能都返回 0。"""

    with pytest.raises(ValueError):
        quadratic_weighted_kappa([], [])

    with pytest.raises(ValueError):
        mean_absolute_error([], [])


def test_distribution_exposes_score_pileup() -> None:
    assert distribution([4, 4, 4, 5]) == {4: 3, 5: 1}

"""评分标准（W4 D1）：每个分值配文字锚，否则 LLM 必然往 4 分堆。

为什么"锚"是这一层的全部价值
----------------------------
不给锚时，模型看到"1–5 分"会做一件很自然的事：给 4 分。它既安全又体面，
于是 54 条 case 里有 51 条是 4 分，均分 4.1，方差趋近 0 —— 这个分数无法区分
"好"和"还行"，趋势图上永远是条直线。**一个没有方差的指标等于没有指标。**

所以本模块只为两件事负责：
1. 把维度、权重、1/3/5 分锚定描述组织成可渲染的 rubric 文本；
2. 把模型返回的 JSON 解析成 `JudgeResult`，并在结构不符时**明确失败**
   （返回 `parse_error` 而不是猜一个中间分）。

评分区间取 1–5 且只允许整数：小数分（4.3）会让一致性校验失去意义——
人工标注做不到小数点后一位，kappa 也就无从计算。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_RUBRIC",
    "Dimension",
    "JudgeResult",
    "JudgeScore",
    "RUBRIC_VERSION",
    "Rubric",
    "RubricParseError",
    "parse_judge_payload",
]

RUBRIC_VERSION = "w4-v1"

# 允许模型把 JSON 包在 ```json fenced block 里：这不是我们要求的格式，
# 但它是真实世界里最高频的一种"不听话"，为此丢掉整批评测不值得。
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class RubricParseError(ValueError):
    """模型输出不符合 rubric 要求的 JSON 结构。

    单独一个异常类型是为了让调用方能把"模型没按格式答"与"网关报错"分开：
    前者降级为 `judge_score=None` 并告警，后者走 `LLMError` 的重试/中断路径。
    """


@dataclass(frozen=True, slots=True)
class Dimension:
    """一个评分维度。

    五个字段里只有 `anchors` 真正约束模型——`key`/`label`/`question` 是容器，
    换了它们分数不会变，但数据格式或 prompt 会变。逐个含义见字段注释。
    """

    # 进 JSON 与落盘的稳定标识（ASCII）。改它等于改历史数据格式：
    # 上一周的 `{"fact": 4}` 在新版本里会变成"未知维度"直接解析失败。
    key: str

    # 只出现在 prompt 里的人读名字（"事实准确"）。可以随时改文案，不影响数据。
    label: str

    # 权重，所有维度之和必须为 1.0。不是"重要性排序"：它直接乘进 overall，
    # 和不为 1 会让总分落到 1–5 之外，报告里看着像 bug。
    weight: float

    # 这个维度"评什么"的一句话说明，渲染进 prompt。必须是可判断的提问，
    # 不能是"是否优秀"这类形容词——模型对形容词的回答就是给 4 分。
    question: str

    # 1/3/5 三档锚定。2/4 分由"介于两档之间"隐含，不单独写——
    # 写全 5 档会让 prompt 长一倍，而模型对相邻档的分辨力并没有变好。
    anchors: Mapping[int, str]

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"{self.key}: weight 必须为正，实际 {self.weight}")

        missing = [score for score in (1, 3, 5) if score not in self.anchors]

        if missing:
            raise ValueError(
                f"{self.key}: 缺少 {missing} 分锚定。没有文字锚的分值区间，"
                "模型会一律给中间的 4 分，指标失去区分度"
            )


@dataclass(frozen=True, slots=True)
class Rubric:
    """一组维度 + 权重。`version` 进报告，用于解释"分数为什么变了"。"""

    # 人读名字（"answer-quality"）。进报告标题，不进解析逻辑。
    name: str

    # 版本号（如 `w4-v1`），随每条 `JudgeResult` 落盘。它的唯一用途是回答
    # "这两周的 judge_score 为什么不可比"——因为中间改过 rubric。
    version: str

    # 维度列表。顺序即 prompt 里的编号顺序，`overall` 按权重求和与顺序无关。
    dimensions: tuple[Dimension, ...] = field(default_factory=tuple)

    # 分值域固定：换成 1–10 会让历史数据不可比，且人工标注更难对齐。
    low: int = 1
    high: int = 5

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise ValueError("rubric 至少要有一个维度")

        keys = [item.key for item in self.dimensions]

        if len(set(keys)) != len(keys):
            raise ValueError(f"维度 key 重复：{keys}")

        total = sum(item.weight for item in self.dimensions)

        if abs(total - 1.0) > 1e-6:
            # 权重和不为 1 时 overall 会落在 1–5 之外，报告里看着像 bug。
            raise ValueError(f"权重之和必须为 1.0，实际 {total:.3f}")

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.dimensions)

    def dimension(self, key: str) -> Dimension:
        for item in self.dimensions:
            if item.key == key:
                return item

        raise KeyError(key)

    def render(self) -> str:
        """渲染成 system prompt。

        刻意不写"请严格评分"这类空话：真正约束模型的是**锚定描述**，
        而锚定描述必须是可观察的行为（"漏掉一个次要子问题"），
        不是形容词（"较好""一般"）。
        """

        lines = [
            "你是严格的答案质量评审员。按下面的维度和分值锚给答案打分。",
            f"分值范围 {self.low}–{self.high} 的整数，不许给小数。",
            "",
            "评分维度和锚定描述：",
        ]

        for index, item in enumerate(self.dimensions, start=1):
            lines.append(f"{index}. {item.label}（key={item.key}，权重 {item.weight:g}）")
            lines.append(f"   评什么：{item.question}")

            for score in sorted(item.anchors):
                lines.append(f"   {score} 分：{item.anchors[score]}")

            lines.append("")

        lines += [
            "只输出一个 JSON 对象，不要任何解释性前后缀：",
            '{"scores": {'
            + ", ".join(
                f'"{item.key}": {{"score": <整数>, "rationale": "<一句证据>"}}'
                for item in self.dimensions
            )
            + "}}",
            "",
            "rationale 必须指向答案里的具体内容（引用原文片段或指出缺失），",
            "不要写“整体不错”这类无法核对的评语。",
        ]

        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class JudgeScore:
    """单个维度的得分。`rationale` 为空视为不合格：无法核对的分数不可审计。"""

    # 与 `Dimension.key` 对应，用来把分数映射回维度与权重。
    key: str

    # 1–5 的整数。小数被 `parse_judge_payload` 直接判为解析失败，
    # 因为人工标注做不到小数点后一位，kappa 也就无从计算。
    score: int

    # 打分依据，必须指向答案里的具体内容。为空视为不合格：
    # 无法核对的分数不可审计，事后没人能说清"这个 3 分是怎么来的"。
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """一次评审的完整结果。

    `parse_error` 非空时 `scores` 为空、`overall` 为 `None`——**绝不补一个中间分**。
    "评不了"和"评了 3 分"在报告里是两件事：前者说明 judge 链路有问题，
    后者说明答案一般。混同会让 judge 的故障伪装成质量下降。
    """

    # 各维度得分。解析成功时长度恒等于 `Rubric.dimensions`（缺一个就整条失败，
    # 不做"按已有维度重归一"）。
    scores: tuple[JudgeScore, ...] = ()

    # 这次评审用的模型 alias。**必须落盘**：judge 与被评模型同源时会系统性
    # 偏好自己的措辞风格（self-preference bias），读者要靠这个字段自己判断。
    judge_alias: str = ""

    # 评审时的 rubric 版本。与当前常量不同时，说明这条分数来自旧口径。
    rubric_version: str = RUBRIC_VERSION

    # 非空即"这次没评成"：模型没按 JSON 答、缺维度、分数越界、或调用直接失败。
    # 此时 `scores` 为空、`overall()` 返回 None，**绝不补一个中间分**。
    parse_error: str | None = None

    # 评审自身的开销（ms / token / 成本 / 币种）。与"被评答案"的成本分开记，
    # W11 成本看板要能回答"judge 吃掉了多少预算"。
    latency_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    currency: str | None = None

    @property
    def available(self) -> bool:
        return self.parse_error is None and bool(self.scores)

    def score_of(self, key: str) -> int | None:
        for item in self.scores:
            if item.key == key:
                return item.score

        return None

    def overall(self, rubric: Rubric) -> float | None:
        """加权均值。缺维度时返回 `None`，不做"按已有维度重归一"。

        重归一看起来更宽容，实际会让"只答了一个维度"的评审拿到 5 分，
        然后这条 case 在趋势图上变成一个好看的点。
        """

        if not self.available:
            return None

        total = 0.0

        for item in rubric.dimensions:
            value = self.score_of(item.key)

            if value is None:
                return None

            total += value * item.weight

        return round(total, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": {item.key: item.score for item in self.scores},
            "rationales": {item.key: item.rationale for item in self.scores},
            "overall": None if not self.available else None,  # 由调用方带 rubric 填充
            "judge_alias": self.judge_alias,
            "rubric_version": self.rubric_version,
            "parse_error": self.parse_error,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
            "currency": self.currency,
        }


def parse_judge_payload(text: str, rubric: Rubric, *, alias: str = "") -> JudgeResult:
    """把模型输出解析成 `JudgeResult`。

    四种失败都会被明确报错，不猜：
    - 不是 JSON（含被 ``` 包住的情况先剥壳）
    - 缺 `scores`
    - 缺某个维度 / 多了未知维度（多余的维度通常意味着 prompt 与解析器版本错位）
    - 分值不是整数或超出 1–5
    """

    payload = _extract_json(text)

    if not isinstance(payload, dict):
        raise RubricParseError(f"输出不是 JSON 对象：{text[:120]!r}")

    raw_scores = payload.get("scores")

    if not isinstance(raw_scores, dict):
        raise RubricParseError(f"缺少 scores 对象：{text[:120]!r}")

    unknown = sorted(set(raw_scores) - set(rubric.keys))

    if unknown:
        raise RubricParseError(
            f"出现 rubric 未定义的维度 {unknown}；rubric 版本 {rubric.version}"
        )

    missing = [key for key in rubric.keys if key not in raw_scores]

    if missing:
        raise RubricParseError(f"缺少维度 {missing}；rubric 版本 {rubric.version}")

    scores: list[JudgeScore] = []

    for key in rubric.keys:
        entry = raw_scores[key]

        if isinstance(entry, int | float):
            # 容忍 "scores": {"fact": 4} 这种压缩写法：结构差异不影响分数解释。
            value, rationale = entry, ""
        elif isinstance(entry, dict):
            value = entry.get("score")
            rationale = str(entry.get("rationale") or "")
        else:
            raise RubricParseError(f"维度 {key} 的值既不是数字也不是对象：{entry!r}")

        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RubricParseError(f"维度 {key} 的 score 不是数字：{value!r}")

        numeric = float(value)

        if numeric != int(numeric):
            raise RubricParseError(
                f"维度 {key} 的 score={numeric} 是小数；小数分无法与人工标注对齐"
            )

        if not rubric.low <= int(numeric) <= rubric.high:
            raise RubricParseError(
                f"维度 {key} 的 score={int(numeric)} 超出 {rubric.low}–{rubric.high}"
            )

        scores.append(JudgeScore(key=key, score=int(numeric), rationale=rationale))

    return JudgeResult(
        scores=tuple(scores), judge_alias=alias, rubric_version=rubric.version
    )


def _extract_json(text: str) -> Any:
    stripped = text.strip()

    fenced = _FENCE.search(stripped)

    if fenced:
        stripped = fenced.group(1).strip()

    try:
        return json.loads(stripped)
    except ValueError as exc:
        # 有些模型会在 JSON 前后加一句"好的，评分如下："。只在这种情况下
        # 退一步找第一个 { 到最后一个 } —— 不扩大到"正则捞数字"，那太容易捞错。
        start = stripped.find("{")
        end = stripped.rfind("}")

        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except ValueError:
                pass

        raise RubricParseError(f"输出不是合法 JSON：{exc}；原文 {text[:120]!r}") from exc


# --------------------------------------------------------------------------- #
# 默认 rubric：四个维度覆盖"说对了 / 有出处 / 说全了 / 说清了"
# --------------------------------------------------------------------------- #

DEFAULT_RUBRIC = Rubric(
    name="answer-quality",
    version=RUBRIC_VERSION,
    dimensions=(
        Dimension(
            key="fact",
            label="事实准确",
            weight=0.40,
            question="答案里的事实性陈述是否成立，有没有可以指出的错误",
            anchors={
                1: "存在明确的事实错误，会把读者引向错误做法",
                3: "主体正确，但有一处细节不严谨（如把上限说成常见值）",
                5: "全部事实性陈述都成立，且没有含糊其辞的兜底说法",
            },
        ),
        Dimension(
            key="grounding",
            label="引用支撑",
            weight=0.25,
            question="答案中的关键结论能否被给定引用支撑；没有引用时是否明确标注为推断",
            anchors={
                1: "引用了不存在的来源，或把推断说成既定事实",
                3: "结论与引用方向一致，但有一处结论在引用里找不到对应",
                5: "每条关键结论都能在引用中找到对应，超出引用的部分明说是推断",
            },
        ),
        Dimension(
            key="coverage",
            label="覆盖度",
            weight=0.20,
            question="是否回答了问题的主问题和它的次要子问题",
            anchors={
                1: "答非所问，或只重复了问题",
                3: "覆盖主问题，但缺 1 个次要子问题",
                5: "主问题与全部次要子问题都有明确回答，没有遗漏",
            },
        ),
        Dimension(
            key="concision",
            label="简洁",
            weight=0.15,
            question="是否在回答问题的前提下尽量短，有没有与问题无关的铺陈",
            anchors={
                1: "大段与问题无关的内容，读者要自己找答案",
                3: "有 1–2 句可删的铺垫或重复",
                5: "每句话都在推进答案，删任何一句都会丢信息",
            },
        ),
    ),
)

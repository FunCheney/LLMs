"""Judge 实现（W4 D1）：走 gateway，不自己发 HTTP。

三条纪律，每条都对应一个"看起来能跑但会翻车"的写法
--------------------------------------------------
1. **alias 从 `catalog.routes["strong"]` 读**，不在代码里写死 `claude-sonnet-4-6`。
   写死的后果不是"跑不起来"，而是模型换名字/换供应商后评分数值整体漂移，
   而报告上只写着"质量下降"（ADR-004/009）。
2. **temperature 恒 0**。评审需要的是同一个答案永远得到同一个分；
   温度 0 也去不掉全部随机性（上游实现有浮点非确定性），但能把方差压到可测。
3. **解析失败不抛断整批**：记 `parse_error` 与 warning，`overall=None`。
   54 条里有一条模型没按 JSON 答，不该毁掉整晚的评测——但它也**不能变成 3 分**。

防自偏好
--------
Judge 与被评模型同源时，会系统性偏好自己的措辞风格（self-preference bias）。
这里做两件事：`strong` tier 与被评的 `cheap` tier 默认不同源；
`JudgeResult.judge_alias` 落进报告，让读者能自己判断这次是不是同一个模型。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_framework.llm.client import ChatCall, LLMGateway
from agent_framework.llm.errors import LLMError
from agent_framework.llm.types import Message, MessageRole
from agent_framework.observability import compute_cost

from evals.judge.rubric import (
    DEFAULT_RUBRIC,
    JudgeResult,
    JudgeScore,
    Rubric,
    RubricParseError,
    parse_judge_payload,
)
from evals.judge.text import overlap_ratio

__all__ = [
    "HeuristicJudge",
    "Judge",
    "JudgeCase",
    "LLMJudge",
    "build_judge",
]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JudgeCase:
    """送进评审的最小信息集。

    刻意不带 `expect`：把断言里的关键词给 judge 看，等于让它照着考官答案打分，
    一致性会很好看而毫无意义。
    """

    # case 标识，只用于日志与告警定位。**不进 prompt**：模型知道"这是 golden-007"
    # 没有任何好处，反而给了它按编号猜难度的机会。
    id: str

    # 用户原始问题。judge 靠它判 `coverage`（有没有答到点上）与 `concision`
    # （铺陈是否与问题无关）。
    question: str

    # 待评审的答案正文。为空时判 1 分（空答案），不判 `parse_error`——
    # "模型没答"是质量问题，"judge 没评成"才是链路问题。
    answer: str

    # 允许引用的来源（golden 的 `references[].target`）。judge 用它判 `grounding`：
    # 结论能不能在这些来源里找到。**不是**答案里已经引了什么。
    references: tuple[str, ...] = ()

    # case 标签，随评审结果一起带出，用于"哪类题最差"的切片。不影响打分。
    tags: tuple[str, ...] = ()


@runtime_checkable
class Judge(Protocol):
    """评审器协议。`cost`/`latency` 是协议的一部分——W11 成本看板要按 judge 分账。"""

    # 实现标识（`llm` / `heuristic-keyword`）。它决定了报告里 `judge_alias` 写什么，
    # 所以**不允许**规则基线伪装成 LLM judge——否则读者会把关键词匹配的分数
    # 当成模型评审的结论。
    name: str

    async def score(self, case: JudgeCase) -> JudgeResult: ...


def _render_user_prompt(case: JudgeCase) -> str:
    parts = ["## 问题", case.question, "", "## 待评审答案", case.answer or "（空答案）"]

    if case.references:
        parts += ["", "## 允许引用的来源（仅这些）"]
        parts += [f"- {item}" for item in case.references]
    else:
        parts += [
            "",
            "## 允许引用的来源",
            "（本条没有给出引用；答案中若宣称有出处，按无支撑处理）",
        ]

    return "\n".join(parts)


class LLMJudge:
    """走 gateway 的评审器。`alias` 缺省取 strong tier 的 primary。"""

    name = "llm"

    def __init__(
        self,
        gateway: LLMGateway,
        rubric: Rubric = DEFAULT_RUBRIC,
        *,
        alias: str | None = None,
        task: str = "research",
    ) -> None:
        self._gateway = gateway
        self._rubric = rubric
        self._task = task
        # 从路由表取，而不是 `catalog.models[...]` 里挑一个 tier == strong：
        # 路由表才是"谁来干重活"的唯一真相（有人把 strong 的 primary 换掉时，
        # 这里必须跟着换，而不是继续用旧模型当裁判）。
        self._alias = alias or gateway.catalog.routes["strong"].primary

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    async def score(self, case: JudgeCase) -> JudgeResult:
        call = ChatCall(
            messages=[
                Message(role=MessageRole.SYSTEM, content=self._rubric.render()),
                Message(role=MessageRole.USER, content=_render_user_prompt(case)),
            ],
            task=self._task,
            model=self._alias,
            # 温度 0：同一答案重复评审要得到同一个分，否则一致性校验没法做。
            opts={"temperature": 0},
        )

        started = time.perf_counter()

        try:
            outcome = await self._gateway.chat(call)
        except LLMError as exc:
            # 评审失败不是"答案差"。记 parse_error（语义是"这次没评成"），
            # 让上层把它算成 judge 不可用，而不是 0 分。
            _LOG.warning("judge 调用失败 case=%s: %s", case.id, exc)
            return JudgeResult(
                judge_alias=self._alias,
                rubric_version=self._rubric.version,
                parse_error=f"judge 调用失败：{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        latency_ms = (time.perf_counter() - started) * 1000
        response = outcome.response
        usage = response.usage
        model = self._gateway.catalog.models[self._alias]

        try:
            parsed = parse_judge_payload(
                response.text or "", self._rubric, alias=self._alias
            )
        except RubricParseError as exc:
            _LOG.warning("judge 输出不可解析 case=%s: %s", case.id, exc)
            return JudgeResult(
                judge_alias=self._alias,
                rubric_version=self._rubric.version,
                parse_error=str(exc),
                latency_ms=latency_ms,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )

        return JudgeResult(
            scores=parsed.scores,
            judge_alias=self._alias,
            rubric_version=self._rubric.version,
            latency_ms=round(latency_ms, 3),
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cost=compute_cost(model.pricing, usage),
            currency=model.pricing.currency,
        )


class HeuristicJudge:
    """**规则基线**，不是 LLM Judge。

    存在的理由有两个，都很具体：
    1. 没有 key 时，`judge_agreement.py` 仍然需要一个"会给出分"的东西，
       否则一致性校验的代码路径一次都跑不到；
    2. 它给一致性系数提供一个**地板**：如果 LLM judge 的 kappa 不比这堆关键词
       匹配更高，那说明花在 judge 上的钱没买到任何东西。

    它只看三件可数的事：问题关键词是否出现在答案里、答案长度、引用 target 的
    词是否出现在答案里。**它判不了事实对错**，所以 `fact` 维度刻意用覆盖度代理，
    报告里的 `judge_alias` 写成 `heuristic-keyword`，不允许它冒充 LLM judge。
    """

    name = "heuristic-keyword"

    def __init__(self, rubric: Rubric = DEFAULT_RUBRIC) -> None:
        self._rubric = rubric

    async def score(self, case: JudgeCase) -> JudgeResult:
        answer = case.answer or ""

        if not answer.strip():
            return JudgeResult(
                scores=tuple(
                    JudgeScore(key=item.key, score=1, rationale="空答案")
                    for item in self._rubric.dimensions
                ),
                judge_alias=self.name,
                rubric_version=self._rubric.version,
            )

        coverage = _keyword_coverage(case.question, answer)
        grounding = _keyword_coverage(" ".join(case.references), answer)

        scores = (
            JudgeScore(
                key="fact",
                score=_band(coverage),
                rationale=f"问题关键词覆盖率 {coverage:.0%}（代理指标，非事实判定）",
            ),
            JudgeScore(
                key="grounding",
                score=_band(grounding) if case.references else 3,
                rationale=(
                    f"引用词覆盖率 {grounding:.0%}"
                    if case.references
                    else "本条无引用可核，取中位"
                ),
            ),
            JudgeScore(
                key="coverage",
                score=_band(coverage),
                rationale=f"问题关键词命中 {coverage:.0%}",
            ),
            JudgeScore(
                key="concision",
                score=_length_band(len(answer)),
                rationale=f"答案 {len(answer)} 字",
            ),
        )

        return JudgeResult(
            scores=scores,
            judge_alias=self.name,
            rubric_version=self._rubric.version,
            latency_ms=0.0,
        )


def _keyword_coverage(source: str, answer: str) -> float:
    """源文本的实词，有多少出现在答案里。

    第一版用中文二元切片，一致性 kappa 只有 0.075（≈ 随机）：改写后的答案
    几乎不会复用问题里连续的两个字，覆盖率恒低，分数退化成"答案长度分"。
    换成共享的 `content_tokens`（Latin 词 + 去停用 CJK 字）后 kappa 到 0.35。

    不引 jieba：一个 40MB 的分词依赖换不来判"事实对错"的能力，
    而这一层真正缺的正是后者。
    """

    return overlap_ratio(source, answer)


def _band(ratio: float) -> int:
    """覆盖率 → 1/3/5。刻意只落在锚定的三档，不造 2/4 分。"""

    if ratio >= 0.6:
        return 5

    if ratio >= 0.3:
        return 3

    return 1


def _length_band(length: int) -> int:
    if length > 1200:
        return 1

    if length > 500:
        return 3

    return 5


def build_judge(
    *,
    live: bool,
    gateway: LLMGateway | None = None,
    rubric: Rubric = DEFAULT_RUBRIC,
    alias: str | None = None,
) -> Judge:
    """按档位选实现。

    在线档缺 gateway 直接报错而不是静默退化成 heuristic：静默退化会让
    `make eval --live` 打出一份"有 judge_score"的报告，而那分数来自关键词匹配。
    """

    if not live:
        return HeuristicJudge(rubric)

    if gateway is None:
        raise ValueError("在线档必须有 gateway（judge 走 gateway，不准裸 HTTP）")

    return LLMJudge(gateway, rubric, alias=alias)


def mean_overall(results: Sequence[JudgeResult], rubric: Rubric) -> float | None:
    """一批评审的均分。全部不可用时返回 `None`（不是 0）。"""

    values = [item.overall(rubric) for item in results]
    usable = [value for value in values if value is not None]

    if not usable:
        return None

    return round(sum(usable) / len(usable), 4)


def summarize_tokens(results: Sequence[JudgeResult]) -> dict[str, Any]:
    """judge 自身的成本摘要（W11 成本看板要按 judge / 被测分账）。"""

    usable = [item for item in results if item.parse_error is None]
    cost = sum(item.cost or 0.0 for item in usable if item.cost is not None)
    currencies = {item.currency for item in usable if item.currency}

    return {
        "calls": len(results),
        "parsed": len(usable),
        "parse_failures": len(results) - len(usable),
        "prompt_tokens": sum(item.prompt_tokens for item in usable),
        "completion_tokens": sum(item.completion_tokens for item in usable),
        "cost": round(cost, 6) if cost else None,
        "currency": next(iter(currencies), None),
        "avg_latency_ms": (
            round(
                sum(item.latency_ms or 0.0 for item in usable) / len(usable),
                3,
            )
            if usable
            else None
        ),
    }


def case_from_payload(payload: Mapping[str, Any]) -> JudgeCase:
    """把 golden 的一行转成 `JudgeCase`。

    `references` 在 golden 里是 `{kind, target}` 列表，这里只取 `target`：
    judge 不需要知道它是 url 还是 file，只需要能说"这个来源里有没有这条结论"。
    """

    references = tuple(
        str(item.get("target", ""))
        for item in payload.get("references", ())
        if isinstance(item, Mapping)
    )

    return JudgeCase(
        id=str(payload.get("id", "")),
        question=_question_of(payload),
        answer=str(payload.get("answer", "")),
        references=references,
        tags=tuple(str(tag) for tag in payload.get("tags", ())),
    )


def _question_of(payload: Mapping[str, Any]) -> str:
    for message in payload.get("messages", ()):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return str(message.get("content") or "")

    return ""

"""RAGAS 三指标（W4 D2）：给"主观分"补上"可复算的客观分"。

为什么不用 `ragas` 这个库
------------------------
它能做的事被三件事拖住了：`datasets`/`pyarrow` 一串重依赖、默认隐藏 prompt 与
模型调用、把指标口径埋在版本里。**为三个还没接检索的指标引入 200MB 依赖，
换来的是一层看不懂的黑箱**——而这一层最需要的恰恰是"口径能被复核"。

所以：只实现定义，接口稳定后再换库。这也符合仓库纪律"只写被用到的"。

三个指标的口径（详见 `docs/w4/metric-spec-ragas.md`）
-----------------------------------------------------
- `faithfulness`：答案的**陈述**里，有多少能被 context 支撑。分母是陈述数。
- `answer_relevancy`：答案有多少落在问题关心的内容上。
- `context_precision`：给的 context 里，有多少是真正相关的（噪声比例）。

**没有 context 时一律返回 `None`，不是 0.0。** 把"算不了"当 0 会把均值拉下来，
而读者只会看到"faithfulness 掉了"，然后去改 prompt——实际是数据集缺引用。

两种判定方式
------------
默认 `lexical`（词面支撑，零成本、可复算）；可选 `LLMSupport`（走 gateway 判蕴含，
更准但每条要一次调用）。`RagasScores.method` 记录用了哪种，报告里必须显示。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_framework.llm.client import ChatCall, LLMGateway
from agent_framework.llm.errors import LLMError
from agent_framework.llm.types import Message, MessageRole

from evals.judge.text import bigram_coverage, content_tokens, overlap_ratio
from evals.metrics import RagasScores

__all__ = [
    "ClaimSupport",
    "LexicalSupport",
    "LLMSupport",
    "answer_relevancy",
    "context_precision",
    "faithfulness",
    "score_case",
]

_LOG = logging.getLogger(__name__)

# 句子切分：中文句号/问号/感叹号/分号 + 换行。英文句点不带后接空格，
# 否则会把 "0.5" 切成两句。
_SENTENCE = re.compile(r"[^。！？；\n]+[。！？；]?")

# 太短的片段（"是的。""如下："）不构成可核对的陈述，进分母只会稀释指标。
_MIN_CLAIM_CHARS = 8

# 词面支撑的阈值（二元组口径）：claim 的连续二字有 60% 出现在 context 里才算支撑。
# 用单字时上下文一长就会恒为 1.0（"任何中文句子都能凑够单字重合"），
# 所以这里换二元组——代价是改写的结论会被判无支撑，属已知下界（见文档 §6）。
_SUPPORT_THRESHOLD = 0.6

# 主题相关性（问题 vs context / 问题 vs 答案）仍用单字召回口径。
_TOPICAL_THRESHOLD = 0.5


@runtime_checkable
class ClaimSupport(Protocol):
    """判定"一条陈述是否被 contexts 支撑"。"""

    name: str

    async def supported(self, claim: str, contexts: Sequence[str]) -> bool: ...


@dataclass(frozen=True, slots=True)
class LexicalSupport:
    """词面支撑：claim 的实词有多大比例出现在 context 里。

    便宜、可复算，但对**同义改写**不友好——所以报告里 `method=lexical` 的数字
    只能横向比（同一批 case 之间），不能对外宣称"faithfulness 0.72"这种绝对结论。
    """

    # 实现标识，写进 `RagasScores.method`。`lexical` 与 `llm` 的同一个数字
    # 可信度差一个档，报告里必须显示。
    name: str = "lexical"

    # 二元组覆盖率阈值：claim 的连续二字有这么大比例出现在 context 里才算支撑。
    # 调高会更严格（漏判增加），调低会放过改写结论（误判增加）。默认 0.6。
    threshold: float = _SUPPORT_THRESHOLD

    async def supported(self, claim: str, contexts: Sequence[str]) -> bool:
        if not content_tokens(claim):
            return False

        return bigram_coverage(claim, "\n".join(contexts)) >= self.threshold


@dataclass(slots=True)
class LLMSupport:
    """LLM 蕴含判定：把 claim 与 contexts 一起给强模型，问"是否被支撑"。

    一次调用判一条 claim（不做批量）：批量会让"某一条判错"污染整批，
    而这里最需要的是逐条可审计。成本按 claim 计，`faithfulness` 的报告里
    要带上调用次数。
    """

    # 走哪种 gateway。判蕴含这件事本身也要计成本，所以它必须是同一个网关，
    # 不能绕过路由表裸发 HTTP（ADR-004）。
    gateway: LLMGateway

    # 判定模型 alias。`None` 时在 `__post_init__` 里取 `routes["strong"].primary`——
    # 从路由表取而不是写死模型名，换模型时这里自动跟着走。
    alias: str | None = None

    # 实现标识，写进 `RagasScores.method`，用于说明这个 faithfulness 是 LLM 判的。
    name: str = "llm"

    def __post_init__(self) -> None:
        if self.alias is None:
            self.alias = self.gateway.catalog.routes["strong"].primary

    async def supported(self, claim: str, contexts: Sequence[str]) -> bool:
        prompt = (
            "判断下面的陈述能否由给定资料支撑。只回答 SUPPORTED 或 UNSUPPORTED。\n\n"
            f"## 资料\n{chr(10).join(f'- {item}' for item in contexts)}\n\n"
            f"## 陈述\n{claim}"
        )
        call = ChatCall(
            messages=[Message(role=MessageRole.USER, content=prompt)],
            task="research",
            model=self.alias,
            opts={"temperature": 0},
        )

        try:
            outcome = await self.gateway.chat(call)
        except LLMError as exc:
            # 判不了就当作"不支撑"？不。判不了要显式失败，由调用方把整条
            # faithfulness 置 None——否则上游抖一下，faithfulness 就掉一截。
            raise RuntimeError(f"claim 判定失败：{exc}") from exc

        answer = (outcome.response.text or "").strip().upper()

        return answer.startswith("SUPPORTED")


def split_claims(answer: str) -> list[str]:
    """把答案切成陈述句。空答案返回空列表（不是 [""]）。"""

    claims: list[str] = []

    for chunk in _SENTENCE.findall(answer or ""):
        text = chunk.strip()

        if len(text) >= _MIN_CLAIM_CHARS:
            claims.append(text)

    return claims


async def faithfulness(
    answer: str,
    contexts: Sequence[str],
    *,
    support: ClaimSupport | None = None,
) -> tuple[float | None, int]:
    """答案中可被 context 支撑的陈述占比，返回 `(值, 陈述数)`。

    - 无 context → `(None, 0)`：算不了，不是 0 分
    - 无陈述（空答案/全是短句）→ `(None, 0)`：分母为 0，指标无定义
    - 判定过程抛错 → `(None, n)`：部分失败不产出半截数字
    """

    if not contexts or not any(item.strip() for item in contexts):
        return None, 0

    claims = split_claims(answer)

    if not claims:
        return None, 0

    judge = support or LexicalSupport()
    supported = 0

    try:
        for claim in claims:
            if await judge.supported(claim, contexts):
                supported += 1
    except RuntimeError as exc:
        _LOG.warning("faithfulness 判定失败，本条记 None：%s", exc)

        return None, len(claims)

    return round(supported / len(claims), 4), len(claims)


async def answer_relevancy(
    answer: str, question: str, *, support: ClaimSupport | None = None
) -> float | None:
    """答案对问题的覆盖程度。

    RAGAS 原版是"用答案反生成问题再比相似度"；这里用**问题实词被答案覆盖的比例**，
    方向一致（答案有没有落在问题关心的内容上）、成本为 0、可复算。
    真正需要反生成时，把 `support` 换成 LLM 版即可——接口不变。
    """

    if not answer.strip() or not question.strip():
        return None

    if not content_tokens(question):
        return None

    return round(overlap_ratio(question, answer), 4)


def context_precision(
    question: str, contexts: Sequence[str], *, support: ClaimSupport | None = None
) -> float | None:
    """context 里"与问题相关"的比例。

    没有重排、没有检索时这就是个噪声度量：一条 case 给了 3 个 references，
    只有 1 个真相关 → 0.33。W5 接检索后，这里换成"检索结果的前 k 个里有几个相关"。
    """

    if not contexts:
        return None

    if not content_tokens(question):
        return None

    relevant = 0

    for context in contexts:
        if overlap_ratio(question, context) >= _TOPICAL_THRESHOLD:
            relevant += 1

    return round(relevant / len(contexts), 4)


async def score_case(
    *,
    question: str,
    answer: str,
    references: Sequence[str],
    support: ClaimSupport | None = None,
) -> RagasScores | None:
    """一条 case 的三指标。

    返回 `None` 的条件：既没有 references 也没有答案——这种 case 连"算不了"
    都算不上，它根本没进入 RAG 场景，不该出现在 `ragas_samples` 里。
    """

    if not answer.strip() and not references:
        return None

    judge = support or LexicalSupport()
    faith, claims = await faithfulness(answer, references, support=judge)

    return RagasScores(
        faithfulness=faith,
        answer_relevancy=await answer_relevancy(answer, question),
        context_precision=context_precision(question, references),
        method=judge.name,
        claims=claims,
        contexts=len(references),
    )

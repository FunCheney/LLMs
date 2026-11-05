"""共用的词元切分（W4）。**只此一份**。

为什么抽出来：`llm_judge` 的规则基线与 `ragas_metrics` 的词面支撑用同一套词元。
两份实现漂移的后果不是报错，而是两个指标对同一段文本给出不一致的判断，
而报告上两者并排显示，读者会以为"faithfulness 和覆盖度矛盾"。

一个必须避免的错误：中文句子没有空格，若把"2 字以上的连续串"当词，
`"幂等键需要在重试之间保持不变"` 会整个变成一个 token——它与任何改写过的话都不相等，
覆盖率恒为 0，指标退化成"是否原句照抄"。
所以：**Latin 词按空格切，CJK 按字切并去停用字**。
"""

from __future__ import annotations

__all__ = ["STOP_CHARS", "bigram_coverage", "content_tokens", "overlap_ratio"]

# 高频虚词与疑问词。留着它们会让"任何一个通顺的中文句子"命中一大半问题词，
# 使覆盖率失去区分度（实测：不去停用字时覆盖率几乎恒在 0.8 以上）。
STOP_CHARS = set(
    "的了是吗呢吧啊和与及或在有就都也很不没之其以为对把被从到会可能要让用这那一个"
    "什么怎样如何为何是否以及而且但是所以如果因为可以需要应该时候问题我们他们"
    "它你您自己上下左右里外前后当中"
)


def content_tokens(text: str) -> set[str]:
    """实词集合：ASCII 词（长度 ≥2）+ 去停用字后的 CJK 单字。"""

    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())

    latin = {
        word
        for word in cleaned.split()
        if len(word) >= 2 and word.isascii()
    }
    cjk = {
        ch
        for ch in cleaned
        if "\u4e00" <= ch <= "\u9fff" and ch not in STOP_CHARS
    }

    return latin | cjk


def bigram_coverage(claim: str, context: str) -> float:
    """claim 的连续二元组有多大比例出现在 context 里。

    和 `overlap_ratio`（单字）分工不同：
    - **单字**适合"主题相关性"（问题 vs 文档）：要召回，改写也要算命中；
    - **二元组**适合"这句话是否被原文支撑"：要精确，否则上下文一长，
      任何中文句子都能凑够单字重合，faithfulness 会恒为 1.0（实测发生过）。

    代价是：**改写的正确结论会被判无支撑**。所以它是下界，不是估计——
    这条写进了 `metric-spec-ragas.md`，报告里必须标明 `method=lexical`。
    """

    left = _bigrams(claim)

    if not left:
        return 0.0

    return len(left & _bigrams(context)) / len(left)


def _bigrams(text: str) -> set[str]:
    stripped = "".join(ch.lower() for ch in text if ch.isalnum())

    return {stripped[i : i + 2] for i in range(len(stripped) - 1)}


def overlap_ratio(source: str, target: str) -> float:
    """`source` 的实词有多大比例出现在 `target` 里。source 无实词时返回 0.0。"""

    tokens = content_tokens(source)

    if not tokens:
        return 0.0

    return len(tokens & content_tokens(target)) / len(tokens)

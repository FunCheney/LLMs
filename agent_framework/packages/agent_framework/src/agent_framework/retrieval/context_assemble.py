"""上下文组装（W6 D4）：去冗余 → 预算裁剪 → 首尾放置。

三个动作都在**检索之后、编号之前**完成。这个位置不是随便选的：`build_context`
一旦编好号，后续任何移动/删除都会让答案里的 `[n]` 指向错的段。所以顺序固定为

    检索结果 → 去冗余 → 按预算裁剪 → 首尾重排 → 编号（citation.build_context）

三条判断，每条都有反例：

- **去冗余用字符 n-gram 相似度，不引入 embedding。** 复用向量算相似度要先有
  embedding，而那意味着上下文组装要依赖模型——一个纯文本的后处理不该有这个依赖。
  4-gram Jaccard 对"同一段被切两次"这类近重复足够灵敏，对"同主题不同段"也不会误杀。
- **裁剪按顺序前缀裁，不跳着凑满预算。** 跳过一条放不下的块去装后面更小的块，
  会让"第 5 条进来了、第 3 条没进"，而这两条只差 200 token——事后无法解释为什么。
  前缀裁剪的规则只有一句："预算用完就停"，任何人能复算。
- **首尾放置改顺序不改编号。** lost-in-the-middle 说的是模型的注意力分布，
  不是检索的相关性分布。所以重排只能发生在编号之前；编号之后再重排，
  引用的段与编号就对不上了。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..knowledge.base import Chunk, ScoredChunk

# 中文 token 粗估：qwen / gpt 系 tokenizer 大约 1 token ≈ 1.5–1.8 个汉字。
# 取 1.6 是**偏保守**（估得多）——估少了会超出模型窗口，估多了只浪费一点预算，
# 两者代价不对称。
DEFAULT_CHARS_PER_TOKEN = 1.6

# 单次回答的上下文预算。3000 token ≈ 4800 汉字，够 6–8 个块，
# 再多也不会提升答案质量（反而触发 lost-in-the-middle）。
DEFAULT_TOKEN_BUDGET = 3000

# 近重复判定阈值。0.9 是"几乎逐字相同"，调低会开始合并同主题的不同段落，
# 而那正是需要并排对比的内容（冲突检测就靠它）。
DEFAULT_DEDUPE_THRESHOLD = 0.9

# 字符 n-gram 的 n。4 对中文合适：2 太碎（"检索"这种词到处都是），
# 6 太长（一个标点差异就让相似度掉一半）。
NGRAM_SIZE = 4

_STOP_CHARS = " \t\r\n　，。！？；：、,.!?;:（）()【】[]\"'“”‘’"


def estimate_tokens(text: str, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """字符数 → token 数的粗估。零依赖，误差 ±30% 对预算控制足够。

    不引 tiktoken：它在不同模型间不通用，引进来只会让人以为预算是精确的。
    """

    if chars_per_token <= 0:
        raise ValueError("chars_per_token 必须大于 0")
    return max(1, math.ceil(len(text or "") / chars_per_token))


def _normalize(text: str) -> str:
    return "".join(char for char in text or "" if char not in _STOP_CHARS)


def _ngrams(text: str, size: int = NGRAM_SIZE) -> set[str]:
    normalized = _normalize(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[i : i + size] for i in range(len(normalized) - size + 1)}


def similarity(left: Chunk, right: Chunk) -> float:
    """两块的相似度（4-gram Jaccard）。0 = 无关，1 = 逐字相同。"""

    a = _ngrams(left.text)
    b = _ngrams(right.text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class DedupeResult:
    kept: list[ScoredChunk] = field(default_factory=list)
    # 被合并掉的 chunk_id -> 保留下来的那个 chunk_id。报告要能说清"删了哪条、为什么"
    merged: dict[str, str] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return len(self.merged)


def deduplicate(
    chunks: Sequence[ScoredChunk],
    *,
    threshold: float = DEFAULT_DEDUPE_THRESHOLD,
) -> DedupeResult:
    """合并近重复块，保留**名次最靠前**的那条。

    保留靠前而不是保"最长"：长度不是质量的代理，而名次是本次检索对它相关性
    的判断。留最长会在"短版是摘要、长版含噪声"时选出更差的那条。

    只在**已排序**的序列上有意义（先来的赢），所以调用方必须先按相关度排好。
    """

    if not 0 < threshold <= 1:
        raise ValueError("threshold 必须落在 (0, 1]；0 表示全部合并，1 表示只合并逐字相同")

    kept: list[ScoredChunk] = []
    merged: dict[str, str] = {}

    for scored in chunks:
        duplicate_of: str | None = None
        for winner in kept:
            if similarity(winner.chunk, scored.chunk) >= threshold:
                duplicate_of = winner.chunk.chunk_id
                break
        if duplicate_of is None:
            kept.append(scored)
        else:
            merged[scored.chunk.chunk_id] = duplicate_of

    return DedupeResult(kept=kept, merged=merged)


@dataclass(frozen=True)
class TrimResult:
    kept: list[ScoredChunk] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    tokens: int = 0


def trim_to_budget(
    chunks: Sequence[ScoredChunk],
    token_budget: int,
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    reserve_tokens: int = 0,
) -> TrimResult:
    """按预算裁掉尾部。返回保留的块与**被裁掉的 chunk_id**。

    `reserve_tokens` 是留给"引用规则 + 问题 + 答案"的余量。不留的话，
    参考资料刚好填满窗口，模型没有空间写答案——表现为 finish_reason=length
    与半句答案，而日志上看是"检索很成功"。
    """

    if token_budget <= 0:
        raise ValueError("token_budget 必须大于 0")
    usable = token_budget - max(reserve_tokens, 0)
    if usable <= 0:
        return TrimResult(kept=[], dropped=[s.chunk.chunk_id for s in chunks], tokens=0)

    kept: list[ScoredChunk] = []
    dropped: list[str] = []
    used = 0

    for position, scored in enumerate(chunks):
        cost = estimate_tokens(scored.chunk.text, chars_per_token=chars_per_token)
        if used + cost > usable:
            # 从这里开始整段丢弃（前缀裁剪）。不"挑小的塞进来"：
            # 那样事后无法解释为什么第 3 条没进、第 5 条进了。
            dropped.extend(rest.chunk.chunk_id for rest in chunks[position:])
            break
        kept.append(scored)
        used += cost

    return TrimResult(kept=kept, dropped=dropped, tokens=used)


def order_for_attention(chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """把最相关的放首尾、次要的放中间：`1,3,5,…,6,4,2`。

    依据是 lost-in-the-middle：长上下文里模型对首尾的利用率明显高于中段。
    所以第 1 名留在开头，第 2 名放到结尾（最后一个位置同样高注意力），
    第 3 名放第二……中段留给最不相关的几条。

    **只改顺序，不改 `rank`**：`rank` 是检索名次，是归因与指标用的；
    这里的顺序是"给模型看的排布"。两者混成一个字段，报告里就分不清
    "它排第 2 是因为检索给了第 2，还是因为我们把它放到了第 2 位"。
    """

    if len(chunks) <= 2:
        return list(chunks)

    head = list(chunks[0::2])
    tail = list(chunks[1::2])[::-1]
    return head + tail


@dataclass(frozen=True)
class AssembledContext:
    """组装结果。每个被丢掉的块都能在 `dropped_*` 里找到原因。"""

    chunks: list[ScoredChunk] = field(default_factory=list)
    tokens: int = 0
    duplicate_of: dict[str, str] = field(default_factory=dict)
    dropped_over_budget: list[str] = field(default_factory=list)
    token_budget: int = 0
    attention_ordered: bool = False

    @property
    def dropped(self) -> int:
        return len(self.duplicate_of) + len(self.dropped_over_budget)

    def explain(self, chunk_id: str) -> str:
        """回答"这条为什么没进上下文"（W7 的检查点从这里开始）。"""

        if chunk_id in self.duplicate_of:
            return f"duplicate_of:{self.duplicate_of[chunk_id]}"
        if chunk_id in self.dropped_over_budget:
            return "over_token_budget"
        if any(scored.chunk.chunk_id == chunk_id for scored in self.chunks):
            return "included"
        return "not_in_candidates"


def assemble(
    chunks: Sequence[ScoredChunk],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    dedupe_threshold: float | None = DEFAULT_DEDUPE_THRESHOLD,
    attention_ordering: bool = True,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    reserve_tokens: int = 0,
) -> AssembledContext:
    """去冗余 → 裁剪 → 首尾重排。**不编号**，编号交给 `citation.build_context`。

    参数用 `None` 关掉某一步，而不是传 0/False 这种哨兵值：
    `dedupe_threshold=0` 是"全部合并"，与"不合并"差得远，很容易传错。
    """

    if dedupe_threshold:
        ordered = deduplicate(chunks, threshold=dedupe_threshold)
    else:
        ordered = DedupeResult(kept=list(chunks))

    trimmed = trim_to_budget(
        ordered.kept,
        token_budget,
        chars_per_token=chars_per_token,
        reserve_tokens=reserve_tokens,
    )
    final = order_for_attention(trimmed.kept) if attention_ordering else list(trimmed.kept)

    return AssembledContext(
        chunks=final,
        tokens=trimmed.tokens,
        duplicate_of=ordered.merged,
        dropped_over_budget=trimmed.dropped,
        token_budget=token_budget,
        attention_ordered=attention_ordering,
    )

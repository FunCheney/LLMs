"""上下文组装的契约（W6 D4）。

三个动作各有一次"看起来对、实际毁了溯源"的写法：

- 裁剪后不重编号 → 答案引用 `[7]`，而 context 只剩 5 条；
- 去冗余保留"更长的" → 留下带噪声的长版，丢掉干净的短版；
- 首尾放置改的是 `rank` 而不是顺序 → 检索名次被污染，指标再也对不上。
"""

from __future__ import annotations

import pytest
from agent_framework.knowledge import ScoredChunk, build_context, parse_citations
from agent_framework.retrieval import (
    assemble,
    deduplicate,
    estimate_tokens,
    order_for_attention,
    similarity,
    trim_to_budget,
)


def _chunk(
    chunk_id: str,
    *,
    text: str = "正文内容",
    page: int = 1,
) -> ScoredChunk:
    from agent_framework.knowledge import Chunk

    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc1",
            text=text,
            source_uri="docs/a.md",
            page_start=page,
            page_end=page,
        ),
        score=1.0,
        retriever="dense",
        rank=1,
    )


def test_identical_chunks_are_merged_keeping_the_better_rank() -> None:
    text = "品牌语气要克制，避免夸张形容词堆叠。"
    chunks = [_chunk("first", text=text), _chunk("dup", text=text)]

    result = deduplicate(chunks, threshold=0.9)

    assert [c.chunk.chunk_id for c in result.kept] == ["first"]
    assert result.merged == {"dup": "first"}


def test_different_sections_of_same_topic_are_not_merged() -> None:
    """同主题不同段必须都留下——冲突检测就靠两段并排对比。"""

    chunks = [
        _chunk("a", text="品牌语气应克制，避免夸张形容词堆叠，尤其禁用绝对化用语。"),
        _chunk("b", text="社媒发帖节奏按渠道分层：高频短内容配合周更长文，节奏稳定。"),
    ]

    result = deduplicate(chunks, threshold=0.9)

    assert result.removed == 0


def test_threshold_one_only_merges_verbatim() -> None:
    base = "品牌语气要克制，避免夸张的形容词堆叠。"
    near = "品牌语气要克制，避免夸张形容词堆叠。"
    chunks = [_chunk("a", text=base), _chunk("b", text=near)]

    assert deduplicate(chunks, threshold=1.0).removed == 0
    assert deduplicate(chunks, threshold=0.5).removed == 1


def test_similarity_is_symmetric_and_bounded() -> None:
    left = _chunk("a", text="检索流水线的预算按剩余时间递减")
    right = _chunk("b", text="检索流水线的预算是按剩余时间递减的")

    value = similarity(left.chunk, right.chunk)

    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(similarity(right.chunk, left.chunk))


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        deduplicate([], threshold=0)


def test_trim_keeps_prefix_and_reports_dropped_ids() -> None:
    """前缀裁剪：预算用完就停，不跳着挑小的塞进来（那种排布无法复算）。"""

    chunks = [_chunk(f"c{i}", text="字" * 100) for i in range(6)]
    result = trim_to_budget(chunks, token_budget=200, chars_per_token=1.0)

    assert [c.chunk.chunk_id for c in result.kept] == ["c0", "c1"]
    assert result.dropped == ["c2", "c3", "c4", "c5"]


def test_trim_reserves_tokens_for_answer() -> None:
    chunks = [_chunk("c0", text="字" * 100), _chunk("c1", text="字" * 100)]

    without = trim_to_budget(chunks, token_budget=250, chars_per_token=1.0)
    with_reserve = trim_to_budget(
        chunks, token_budget=250, chars_per_token=1.0, reserve_tokens=150
    )

    assert len(without.kept) == 2
    assert len(with_reserve.kept) == 1


def test_attention_order_puts_best_first_and_second_last() -> None:
    """`1,3,5,4,2`：第 1 名开头、第 2 名结尾——两端是注意力最高的位置。"""

    chunks = [_chunk(f"c{i}") for i in range(5)]

    ordered = order_for_attention(chunks)

    assert [c.chunk.chunk_id for c in ordered] == ["c0", "c2", "c4", "c3", "c1"]


def test_attention_order_does_not_touch_rank() -> None:
    """`rank` 是检索名次，重排只改顺序。混成一个字段，报告就分不清因果。"""

    chunks = [
        ScoredChunk(chunk=_chunk(f"c{i}").chunk, score=1.0, retriever="dense", rank=i + 1)
        for i in range(4)
    ]

    ordered = order_for_attention(chunks)

    # 顺序变成 c0,c2,c3,c1，但每条的 rank 仍是它被检索时的名次
    assert [c.chunk.chunk_id for c in ordered] == ["c0", "c2", "c3", "c1"]
    assert [c.rank for c in ordered] == [1, 3, 4, 2]


def test_assembled_context_explains_why_a_chunk_is_missing() -> None:
    text = "完全相同的段落，重复出现两次。" * 2
    chunks = [_chunk("keep", text=text), _chunk("dup", text=text)]

    assembled = assemble(chunks, token_budget=1000, dedupe_threshold=0.9)

    assert assembled.explain("dup") == "duplicate_of:keep"
    assert assembled.explain("keep") == "included"
    assert assembled.explain("never") == "not_in_candidates"


def test_trimmed_context_still_numbers_continuously() -> None:
    """裁剪到 5 条后，编号必须仍是 1..5 且能解析——不能留下悬空的 `[9]`。"""

    chunks = [_chunk(f"c{i}", text="字" * 100) for i in range(10)]
    assembled = assemble(
        chunks * 1, token_budget=600, dedupe_threshold=None, chars_per_token=1.0
    )
    context = build_context(assembled.chunks)

    assert len(context) == len(assembled.chunks)
    assert sorted(context.by_index()) == list(range(1, len(assembled.chunks) + 1))
    # 答案里引用最后一条也要能解析
    parsed = parse_citations(f"结论 [{len(context)}]", context)
    assert parsed.unknown == ()


def test_assemble_without_dedupe_keeps_everything() -> None:
    text = "逐字相同的段落。"
    chunks = [_chunk("a", text=text), _chunk("b", text=text)]

    assembled = assemble(chunks, token_budget=1000, dedupe_threshold=None)

    assert len(assembled.chunks) == 2
    assert assembled.duplicate_of == {}


def test_estimate_tokens_is_conservative() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("字" * 160, chars_per_token=1.6) == 100


def test_invalid_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="token_budget"):
        trim_to_budget([], token_budget=0)

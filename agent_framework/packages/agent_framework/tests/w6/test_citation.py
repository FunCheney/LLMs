"""引用溯源的契约（W6 D1）。

每条用例都对应一个"溯源形同虚设"的具体形态：

- 编号与映射不同源 → 答案 `[i]` 指向隔壁那段，而两边代码看起来都对；
- 引用缺失静默通过 → 准确率的分母被悄悄缩小；
- 出处缺失被当成正常 → 抽样报告全绿，但点开原文找不到东西；
- 混入相邻段被一刀判死 → 标注只覆盖 69/425，模型引对了没标的地方反而扣分。
"""

from __future__ import annotations

import pytest
from agent_framework.knowledge import (
    Chunk,
    CitationExpectation,
    ScoredChunk,
    audit_citations,
    build_context,
    citation_accuracy,
    citation_payload,
    missing_source_chunks,
    parse_citations,
)


def _chunk(
    chunk_id: str,
    *,
    text: str = "正文",
    source_uri: str = "docs/brand.md",
    page: int | None = 3,
    metadata: dict | None = None,
    document_id: str = "doc1",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        source_uri=source_uri,
        page_start=page,
        page_end=page,
        section_path=("第二章", "2.1 语气"),
        metadata={"document_title": "品牌手册", **(metadata or {})},
    )


def _scored(chunk: Chunk, *, score: float = 1.0, rank: int = 1) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, retriever="dense", rank=rank)


# --- build_context ----------------------------------------------------------


def test_numbering_and_mapping_come_from_the_same_loop() -> None:
    """文本里的 `[n]` 与映射的 key 必须逐一对上。

    这是 D1 最容易写错的地方：分两次遍历生成文本和映射时，任一侧的起点差 1
    都会让"答案引第 2 条"变成"引了第 1 条"，而两个函数各自看起来完全正确。
    """

    context = build_context(
        [
            _scored(_chunk("c1", text="第一段正文")),
            _scored(_chunk("c2", text="第二段正文")),
            _scored(_chunk("c3", text="第三段正文")),
        ]
    )

    mapping = context.by_index()

    assert sorted(mapping) == [1, 2, 3]
    assert [mapping[index].chunk_id for index in (1, 2, 3)] == ["c1", "c2", "c3"]
    for index, citation in mapping.items():
        assert f"[{index}]" in context.text
        # 映射里的段正文必须在文本里出现，否则编号指向了一个不在 context 里的块
        assert citation.excerpt[:6] in context.text


def test_context_carries_locator_for_human_verification() -> None:
    context = build_context([_scored(_chunk("c1", text="正文", page=152))])

    citation = context.citations[0]

    assert "docs/brand.md" in citation.locator
    assert "第 152 页" in citation.locator
    assert "2.1 语气" in citation.locator


def test_long_chunk_is_truncated_but_marked() -> None:
    """超长块要截断（否则挤满预算），但必须留省略号——不能让读者以为原文就这么短。"""

    context = build_context([_scored(_chunk("c1", text="字" * 2000))], max_chars_per_chunk=100)

    assert context.citations[0].excerpt.endswith("…")
    assert len(context.citations[0].excerpt) <= 101


def test_max_chars_must_be_positive() -> None:
    try:
        build_context([], max_chars_per_chunk=0)
    except ValueError as exc:
        assert "max_chars_per_chunk" in str(exc)
    else:  # pragma: no cover - 走到这里说明校验丢了
        raise AssertionError("max_chars_per_chunk=0 应报错")


# --- parse ------------------------------------------------------------------


def test_parse_dedupes_and_keeps_order() -> None:
    context = build_context(
        [_scored(_chunk("c1")), _scored(_chunk("c2")), _scored(_chunk("c3"))]
    )

    parsed = parse_citations("先说 A [2]，再说 B [1]，再补充 [2]。", context)

    assert parsed.indices == (2, 1)
    assert [c.chunk_id for c in parsed.citations] == ["c2", "c1"]


def test_parse_flags_unknown_index_instead_of_dropping_it() -> None:
    """编造的编号必须被记下来。静默丢弃会让"引用错"变成"引用少"。"""

    parsed = parse_citations("结论 [9]", build_context([_scored(_chunk("c1"))]))

    assert parsed.indices == ()
    assert parsed.unknown == (9,)


# --- audit ------------------------------------------------------------------


def test_no_citation_is_a_failure_not_a_format_nicety() -> None:
    context = build_context([_scored(_chunk("c1"))])

    audit = audit_citations("这是一段没有任何出处的回答。", context)

    assert audit.verdict == "no_citation"
    assert audit.ok is False


def test_unknown_index_verdict_wins_over_others() -> None:
    context = build_context([_scored(_chunk("c1"))])

    audit = audit_citations("结论 [1] 与 [7]", context)

    assert audit.verdict == "unknown_index"
    assert 7 in audit.parsed.unknown


def test_missing_source_is_detected_via_prefix() -> None:
    """桥接层用 `missing-source:` 占位时，必须判失败而不是"有出处"。"""

    context = build_context(
        [_scored(_chunk("c1", source_uri="missing-source:doc1"))]
    )

    audit = audit_citations("结论 [1]", context)

    assert audit.verdict == "source_missing"


def test_missing_source_is_detected_via_metadata_flag() -> None:
    """`metadata["source_uri_missing"]` 是桥接层的显式标记，同样要判失败。

    反向验证的入口：把一条 chunk 的出处标记为缺失，抽样必须把它算成失败，
    而不是静默通过。
    """

    context = build_context(
        [_scored(_chunk("c1", metadata={"source_uri_missing": True}))]
    )

    audit = audit_citations("结论 [1]", context)

    assert audit.verdict == "source_missing"


def test_off_target_when_no_cited_chunk_is_the_gold_one() -> None:
    """只引了邻居、一条都没引到标注段 → 引错。

    注意口径是"**全都**没中"才判失败：只要有一条命中，就说明答案指向了正确出处，
    其余引用记进 `unmatched` 并计入 precision（理由见 `audit_citations` 的
    docstring——标注只有 69/425，按"条条命中"判会把标注不全算成模型引错）。
    """

    context = build_context([_scored(_chunk("neighbor")), _scored(_chunk("gold"))])
    expected = CitationExpectation(chunk_ids=frozenset({"gold"}))

    audit = audit_citations("结论 [1]", context, expected=expected)

    assert audit.verdict == "off_target"
    assert audit.unmatched == (1,)
    assert audit.matched == ()
    assert audit.precision == 0.0


def test_mixed_citations_pass_with_precision_recorded() -> None:
    """引对了 + 混入非标注段 → 通过，但 precision 要如实反映噪声。

    这条是实测回填的：首次抽样 `golden-001` 引了 5 条，2 条是标注段、
    3 条是同页相邻块。判它失败会让人去改 prompt，而真正的问题是标注只覆盖 69 块。
    """

    context = build_context(
        [_scored(_chunk("gold")), _scored(_chunk("neighbor")), _scored(_chunk("other"))]
    )
    expected = CitationExpectation(chunk_ids=frozenset({"gold"}))

    audit = audit_citations("结论 [1][2][3]", context, expected=expected)

    assert audit.verdict == "ok"
    assert audit.matched == (1,)
    assert audit.unmatched == (2, 3)
    assert audit.precision == pytest.approx(1 / 3)
    # 噪声必须写进 reasons：不判失败 ≠ 不记录
    assert any("不在标注集内" in reason for reason in audit.reasons)


def test_precision_is_none_without_ground_truth() -> None:
    """没提供 ground truth 时 precision 是 None，不是 1.0。

    "没测"和"测出来满分"混成一个值，报告就没法区分"这周没抽样"和"这周全对"。
    """

    context = build_context([_scored(_chunk("gold"))])

    audit = audit_citations("结论 [1]", context)

    assert audit.precision is None
    assert audit.matched == ()


def test_ok_when_citation_hits_gold_chunk() -> None:
    context = build_context([_scored(_chunk("gold"))])
    expected = CitationExpectation(chunk_ids=frozenset({"gold"}))

    audit = audit_citations("结论 [1]", context, expected=expected)

    assert audit.verdict == "ok"
    assert audit.ok is True


def test_expected_falls_back_to_document_level_only_without_chunk_ids() -> None:
    context = build_context([_scored(_chunk("c1", document_id="doc1"))])

    audit = audit_citations(
        "结论 [1]", context, expected=CitationExpectation(document_ids=frozenset({"doc1"}))
    )

    assert audit.ok is True


# --- 准确率 ------------------------------------------------------------------


def test_accuracy_denominator_is_all_samples() -> None:
    """分母是全部抽样条数。若只统计"解析成功"的，漏引用越多指标越好看。"""

    context = build_context([_scored(_chunk("c1"))])
    good = audit_citations("结论 [1]", context)
    bad = audit_citations("没有引用的结论", context)

    assert citation_accuracy([good, bad]) == 0.5
    assert citation_accuracy([]) == 0.0


# --- 对外投影 ----------------------------------------------------------------


def test_payload_has_locator_and_no_retriever_leak() -> None:
    context = build_context([_scored(_chunk("c1"))])

    payload = citation_payload(context.citations)

    assert payload[0]["chunk_id"] == "c1"
    assert "locator" in payload[0]
    # retriever 是内部归因字段，不该出现在对外响应里（出去就删不掉）
    assert "retriever" not in payload[0]


def test_missing_source_chunks_supports_ingest_selfcheck() -> None:
    chunks = [
        _scored(_chunk("ok")),
        _scored(_chunk("bad", source_uri="missing-source:doc1")),
    ]

    assert missing_source_chunks(chunks) == ["bad"]

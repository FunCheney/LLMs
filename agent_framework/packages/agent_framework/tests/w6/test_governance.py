"""知识治理的契约（W6 D5）。

三个容易写反的地方，每个都会让"治理"变成"掩盖问题"：

- 过期直接丢弃 → 问"当前规定"时无答案，用户以为资料库没有；
- 低质量源只是"记一笔" → 排序不变，等于没治理；
- 冲突自动挑一个 → 错误结论被当成唯一答案，比暴露矛盾危险得多。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from agent_framework.knowledge import (
    Chunk,
    GovernancePolicy,
    ScoredChunk,
    conflict_components,
    conflict_note,
    source_domain,
)


def _scored(
    chunk_id: str,
    *,
    score: float = 1.0,
    source_uri: str = "docs/brand.md",
    metadata: dict | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc1",
            text=f"正文 {chunk_id}",
            source_uri=source_uri,
            metadata=metadata or {},
        ),
        score=score,
        retriever="dense",
        rank=1,
    )


# --- 质量 --------------------------------------------------------------------


def test_trusted_domain_keeps_full_weight() -> None:
    policy = GovernancePolicy()

    assert policy.quality_weight(
        Chunk(
            chunk_id="c1",
            document_id="d1",
            text="t",
            source_uri="https://docs.anthropic.com/en/docs/overview",
        )
    ) == pytest.approx(1.0)


def test_unknown_domain_is_downweighted() -> None:
    policy = GovernancePolicy()
    chunk = Chunk(
        chunk_id="c1", document_id="d1", text="t", source_uri="https://spam.example/x"
    )

    assert policy.quality_weight(chunk) == pytest.approx(0.6)


def test_local_path_is_treated_as_own_material() -> None:
    """自有稿件（仓库内路径）不该被当成"未知站点"降权。"""

    policy = GovernancePolicy()
    chunk = Chunk(chunk_id="c1", document_id="d1", text="t", source_uri="data/raw/a.pdf")

    assert policy.quality_weight(chunk) == pytest.approx(1.0)
    assert source_domain("data/raw/a.pdf") == ""


def test_declared_quality_wins_over_whitelist() -> None:
    policy = GovernancePolicy()
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        text="t",
        source_uri="https://docs.anthropic.com/x",
        metadata={"quality": 0.2},
    )

    assert policy.quality_weight(chunk) == pytest.approx(0.2)


def test_parse_failures_multiply_penalty() -> None:
    policy = GovernancePolicy()
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        text="t",
        source_uri="data/raw/a.pdf",
        metadata={"parse_failures": 2},
    )

    assert policy.weight(chunk) == pytest.approx(0.8)


def test_quality_downweight_changes_ranking() -> None:
    """低质量源分数更高也必须排到高质量源之后——这是"治理生效"的最低证据。"""

    policy = GovernancePolicy()
    chunks = [
        _scored("spam", score=0.90, source_uri="https://spam.example/x"),
        _scored("official", score=0.80, source_uri="https://docs.anthropic.com/x"),
    ]

    ordered = policy.apply(chunks)

    assert [c.chunk.chunk_id for c in ordered] == ["official", "spam"]
    assert ordered[0].score == pytest.approx(0.80)
    assert ordered[1].score == pytest.approx(0.54)


# --- 过期 --------------------------------------------------------------------


def test_expired_is_downweighted_not_dropped() -> None:
    """降权而非丢弃：丢掉会让"最新规定是什么"直接无答案。"""

    policy = GovernancePolicy()
    now = datetime(2026, 11, 7, tzinfo=UTC)
    chunks = [
        _scored("old", score=1.0, metadata={"valid_until": "2024-01-01"}),
        _scored("new", score=0.9, metadata={"valid_until": "2027-01-01"}),
    ]

    ordered = policy.apply(chunks, now=now)

    assert [c.chunk.chunk_id for c in ordered] == ["new", "old"]
    assert len(ordered) == 2  # 一条都没丢
    assert ordered[1].score == pytest.approx(0.5)
    assert ordered[1].chunk.metadata["expired"] is True


def test_expired_newer_than_now_is_not_penalized() -> None:
    policy = GovernancePolicy()
    now = datetime(2026, 11, 7, tzinfo=UTC)
    chunk = _scored("c1", metadata={"valid_until": (now + timedelta(days=3)).isoformat()})

    assert policy.is_expired(chunk.chunk, now=now) is False


def test_malformed_valid_until_is_ignored_not_fatal() -> None:
    """脏值不该让整次检索 500，也不该被当成"未过期"之外的东西。"""

    policy = GovernancePolicy()

    assert policy.is_expired(_scored("c1", metadata={"valid_until": "下个月"}).chunk) is False


# --- 冲突 --------------------------------------------------------------------


def test_conflict_components_group_a_chain() -> None:
    """A↔B、B↔C 是同一个主题的三种说法，不应算成两个独立问题。"""

    chunks = [
        _scored("a", metadata={"conflicts_with": ["b"]}),
        _scored("b", metadata={"conflicts_with": ["a", "c"]}),
        _scored("c", metadata={"conflicts_with": ["b"]}),
        _scored("d"),
    ]

    assert conflict_components(chunks) == [("a", "b", "c")]


def test_conflict_ignores_targets_not_in_context() -> None:
    """矛盾对象不在本次结果里时，答案里指不出来，就不要报"有冲突"。"""

    chunks = [_scored("a", metadata={"conflicts_with": ["zzz"]})]

    assert conflict_components(chunks) == []


def test_conflict_note_uses_given_numbers() -> None:
    chunks = [
        _scored("a", metadata={"conflicts_with": ["b"]}),
        _scored("b", metadata={"conflicts_with": ["a"]}),
    ]

    note = conflict_note(chunks, {"a": 2, "b": 5})

    assert note is not None
    assert "[2]" in note and "[5]" in note
    assert "不要自行裁决" in note


def test_conflict_note_is_none_without_conflicts() -> None:
    assert conflict_note([_scored("a")], {"a": 1}) is None


# --- 与其他模块的边界 ---------------------------------------------------------


def test_negative_scores_go_worse_after_downweighting() -> None:
    """cross-encoder logit 是负分时**不能乘权重**：-3 × 0.6 = -1.8 会让它变好。

    这是治理与精排同时开启时最隐蔽的一处：低质量内容被"降权"之后反而排到前面，
    而日志、分数、顺序看起来都正常。正确做法是负分走除法（-3 / 0.6 = -5）。
    """

    policy = GovernancePolicy()
    chunks = [
        _scored("spam", score=-2.0, source_uri="https://spam.example/x"),
        _scored("official", score=-2.5, source_uri="https://docs.anthropic.com/x"),
    ]

    ordered = policy.apply(chunks)

    by_id = {c.chunk.chunk_id: c for c in ordered}
    assert by_id["spam"].score == pytest.approx(-2.0 / 0.6)
    assert by_id["spam"].score < by_id["official"].score
    assert [c.chunk.chunk_id for c in ordered] == ["official", "spam"]


def test_disabled_policy_is_a_noop() -> None:
    chunks = [_scored("spam", score=0.9, source_uri="https://spam.example/x")]

    assert GovernancePolicy(enabled=False).apply(chunks)[0].score == pytest.approx(0.9)

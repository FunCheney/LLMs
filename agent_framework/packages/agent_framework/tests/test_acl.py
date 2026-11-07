"""ACL 过滤的契约（W6 D3）。

重点是**默认值**：安全实现的默认是"拒绝未标注"，不是"放行"。
这条写反了不会报错，只会让所有忘打标签的文档变成公开文档，
而单测通常只测"有权的人能看到"，测不出这个方向。
"""

from __future__ import annotations

from agent_framework.knowledge import Chunk, ScoredChunk
from agent_framework.retrieval import (
    ACLPolicy,
    Principal,
    RetrievalPipeline,
    filter_visible,
    make_acl_check,
)


def _chunk(chunk_id: str, *, acl: object = None) -> Chunk:
    metadata = {} if acl is None else {"acl": acl}
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=f"正文 {chunk_id}",
        source_uri="docs/brand.md",
        metadata=metadata,
    )


def _scored(chunk_id: str, *, acl: object = None) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(chunk_id, acl=acl), score=1.0, retriever="dense", rank=1)


class FakeRetriever:
    name = "fake"

    def __init__(self, hits: list[ScoredChunk]) -> None:
        self._hits = hits
        self.calls: list[int] = []

    async def retrieve(self, query: str, *, top_k: int = 10, filters=None):
        self.calls.append(top_k)
        return self._hits[:top_k]


# --- 策略 --------------------------------------------------------------------


def test_unlabeled_is_denied_by_default() -> None:
    """默认拒绝。漏打标签是新文档最常见的状态，不能等于"公开"。"""

    principal = Principal(subject_id="u1", groups=frozenset({"eng"}))
    decision = ACLPolicy().decide(_chunk("c1"), principal)

    assert decision.allowed is False
    assert decision.reason == "denied_no_acl"


def test_unlabeled_can_be_allowed_explicitly() -> None:
    decision = ACLPolicy(allow_unlabeled=True).decide(
        _chunk("c1"), Principal(subject_id="u1")
    )

    assert decision.allowed is True


def test_group_membership_allows() -> None:
    chunk = _chunk("c1", acl={"groups": ["growth", "legal"], "subjects": []})

    decision = ACLPolicy().decide(chunk, Principal("u1", frozenset({"legal"})))

    assert decision.allowed is True
    assert decision.reason == "allowed_group"


def test_subject_list_allows_specific_person() -> None:
    chunk = _chunk("c1", acl={"groups": [], "subjects": ["u-42"]})

    assert ACLPolicy().check(chunk, Principal("u-42"))
    assert not ACLPolicy().check(chunk, Principal("u-43", frozenset({"growth"})))


def test_public_group_allows_everyone() -> None:
    chunk = _chunk("c1", acl=["*"])

    assert ACLPolicy().check(chunk, Principal.anonymous())


def test_bare_list_is_treated_as_groups() -> None:
    """入库侧写裸列表是常态，收窄成一种写法会让那些文档全部被拒。"""

    chunk = _chunk("c1", acl=["growth"])

    assert ACLPolicy().check(chunk, Principal("u1", frozenset({"growth"})))


def test_malformed_acl_is_not_treated_as_public() -> None:
    """脏 metadata 不能变成放行理由——按未标注处理（默认拒绝）。"""

    chunk = _chunk("c1", acl=12345)

    assert ACLPolicy().check(chunk, Principal("u1")) is False


def test_anonymous_does_not_match_labeled_documents() -> None:
    chunk = _chunk("c1", acl=["growth"])

    decision = ACLPolicy().decide(chunk, Principal.anonymous())

    assert decision.allowed is False
    assert decision.reason == "denied_not_member"


def test_make_acl_check_returns_none_when_disabled() -> None:
    """不启用时返回 None，而不是"恒真函数"。

    pipeline 用 `is None` 判断"到底有没有过滤"，一个恒真函数会让诊断结果
    看起来像"过滤跑了但没有越权内容"。
    """

    assert make_acl_check(None) is None


def test_filter_visible_counts_without_leaking_text() -> None:
    chunks = [_chunk("ok", acl=["growth"]), _chunk("secret", acl=["legal"])]

    visible, blocked = filter_visible(chunks, Principal("u1", frozenset({"growth"})))

    assert [chunk.chunk_id for chunk in visible] == ["ok"]
    assert blocked == 1


# --- 与 pipeline 的接线 -------------------------------------------------------


async def test_pipeline_filters_after_recall_and_reports_counts() -> None:
    retriever = FakeRetriever(
        [
            _scored("secret", acl=["legal"]),
            _scored("public", acl=["*"]),
        ]
    )
    pipeline = RetrievalPipeline(retriever, budget_ms=500)

    outcome = await pipeline.run(
        "查询",
        top_k=2,
        acl_check=make_acl_check(Principal("u1", frozenset({"growth"}))),
    )

    assert [c.chunk.chunk_id for c in outcome.chunks] == ["public"]
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.filtered_out == 1
    assert outcome.diagnostics.acl_applied is True


async def test_overfetch_compensates_for_filtered_chunks() -> None:
    """过滤后还能给满 top_k —— 靠召回阶段的 over-fetch，不是靠第二轮回查。"""

    hits = [_scored(f"secret{i}", acl=["legal"]) for i in range(6)]
    hits += [_scored(f"ok{i}", acl=["growth"]) for i in range(6)]
    retriever = FakeRetriever(hits)
    pipeline = RetrievalPipeline(retriever, budget_ms=1000, recall_multiplier=3.0)

    outcome = await pipeline.run(
        "查询",
        top_k=4,
        acl_check=make_acl_check(Principal("u1", frozenset({"growth"}))),
    )

    assert retriever.calls == [12]  # 4 × 3
    assert len(outcome.chunks) == 4
    assert all(c.chunk.chunk_id.startswith("ok") for c in outcome.chunks)


async def test_acl_disabled_marks_diagnostics_false() -> None:
    pipeline = RetrievalPipeline(FakeRetriever([_scored("c1")]), budget_ms=500)

    outcome = await pipeline.run("查询", top_k=1)

    assert outcome.diagnostics is not None
    assert outcome.diagnostics.acl_applied is False
    assert outcome.diagnostics.filtered_out == 0


async def test_all_filtered_is_empty_not_error() -> None:
    """全被拦下不是故障：返回空 + 一条 warning，让归因区分"无权"与"坏了"。"""

    pipeline = RetrievalPipeline(
        FakeRetriever([_scored("secret", acl=["legal"])]), budget_ms=500
    )

    outcome = await pipeline.run(
        "查询", top_k=1, acl_check=make_acl_check(Principal("u1"))
    )

    assert outcome.chunks == []
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.filtered_out == 1
    assert outcome.diagnostics.ok is True


def test_policy_rejects_invalid_acl_type_silently_but_safely() -> None:
    """畸形值不抛错（一条脏数据不该 500），但必须落到"拒绝"。"""

    policy = ACLPolicy()

    assert policy.decide(_chunk("c1", acl=None), Principal("u1")).allowed is False
    assert policy.decide(_chunk("c2", acl=[""]), Principal("u1")).allowed is False


def test_decision_reasons_are_stable_strings() -> None:
    """reason 会被写进审计日志，改字符串等于改日志契约。"""

    policy = ACLPolicy()
    chunk = _chunk("c1", acl=["growth"])

    assert policy.decide(_chunk("c0", acl=["*"]), Principal("u1")).reason == "allowed_public"
    assert policy.decide(chunk, Principal("u1", frozenset({"growth"}))).reason == "allowed_group"
    assert policy.decide(chunk, Principal("u1")).reason == "denied_not_member"


def test_make_acl_check_uses_policy_override() -> None:
    check = make_acl_check(Principal("u1"), policy=ACLPolicy(allow_unlabeled=True))

    assert check is not None
    assert check(_chunk("c1")) is True

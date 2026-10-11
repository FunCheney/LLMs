from eval.matchers import find_first_matching_rank, hit_matches_rules
from eval.smoke_schema import SmokeMatchRules, load_smoke_dataset
from retrieval.types import RetrievalResult
from schemas import DocumentChunk


def _hit(
    chunk_id: str,
    section_title: str,
    content: str = "body",
    rank: int = 1,
) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc",
        document_title="Doc",
        section_title=section_title,
        section_path=[section_title],
        page_start=1,
        page_end=1,
        content=content,
        source_uri="doc.pdf",
        source_type="pdf",
    )
    return RetrievalResult(
        chunk_id=chunk_id,
        text=content,
        metadata=chunk.to_dict(),
        score=0.9,
        source="dense",
        chunk=chunk,
        rank=rank,
    )


def test_section_match_ignores_extra_whitespace():
    rules = SmokeMatchRules(any_section_title=["6.5 query()：真正的执行循环"])
    hit = _hit("c1", "6.5   query()：真正的执行循环")
    assert hit_matches_rules(hit, rules)


def test_section_match_can_use_section_path():
    rules = SmokeMatchRules(any_section_title=["22.2 五层权限架构"])
    chunk = DocumentChunk(
        chunk_id="c2",
        document_id="doc",
        document_title="Doc",
        section_title="Unknown",
        section_path=["第四部分", "22.2 五层权限架构"],
        page_start=1,
        page_end=1,
        content="permissions",
        source_uri="doc.pdf",
        source_type="pdf",
    )
    result = RetrievalResult(
        chunk_id="c2",
        text="permissions",
        metadata=chunk.to_dict(),
        score=0.5,
        source="sparse",
        chunk=chunk,
        rank=2,
    )
    assert hit_matches_rules(result, rules)


def test_find_first_matching_rank():
    rules = SmokeMatchRules(any_section_title=["8.3 流式响应的解析"])
    hits = [
        _hit("c1", "other", rank=1),
        _hit("c2", "8.3 流式响应的解析", rank=2),
    ]
    assert find_first_matching_rank(hits, rules) == 2


def test_load_smoke_dataset():
    dataset = load_smoke_dataset("data/eval/smoke_queries.yaml")
    assert len(dataset.queries) >= 10
    assert dataset.defaults.top_k == 5
    negative = next(q for q in dataset.queries if q.id == "smoke-010")
    assert negative.expect_hit is False

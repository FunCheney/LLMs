from generation.context import (
    build_context_blocks,
    citations_referenced_in_answer,
    hits_to_citations,
)
from generation.pipeline import answer_question
from retrieval.types import RetrievalResult
from schemas import DocumentChunk


def _hit(chunk_id: str, section: str, content: str, rank: int) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc",
        document_title="Claude Code 设计指南",
        section_title=section,
        section_path=[section],
        page_start=1,
        page_end=2,
        content=content,
        source_uri="data/raw/book.pdf",
        source_type="pdf",
    )
    return RetrievalResult(
        chunk_id=chunk_id,
        text=content,
        metadata=chunk.to_dict(),
        score=0.5,
        source="hybrid",
        chunk=chunk,
        rank=rank,
    )


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        return [
            _hit(
                "c1",
                "19.1 MCP 是什么",
                "MCP 是 Model Context Protocol。",
                1,
            )
        ]


def test_build_context_blocks_numbering():
    hits = [
        _hit("c1", "A", "first", 1),
        _hit("c2", "B", "second", 2),
    ]
    context = build_context_blocks(hits)
    assert "[1]" in context and "[2]" in context
    assert "first" in context and "second" in context


def test_citations_referenced_in_answer_filters():
    citations = hits_to_citations(
        [_hit("c1", "A", "first", 1), _hit("c2", "B", "second", 2)]
    )
    used = citations_referenced_in_answer("结论见 [1]。", citations)
    assert len(used) == 1
    assert used[0].chunk_id == "c1"


def test_answer_question_empty_hits():
    class EmptyRetriever:
        def retrieve(self, query: str, top_k: int = 5):
            return []

    response = answer_question(
        query="test",
        retriever=EmptyRetriever(),
        retrieval_mode="hybrid",
        dry_run=True,
    )
    assert response.insufficient_evidence is True
    assert response.citations == []


def test_answer_question_dry_run():
    response = answer_question(
        query="MCP 是什么",
        retriever=FakeRetriever(),
        retrieval_mode="hybrid",
        dry_run=True,
    )
    assert response.insufficient_evidence is False
    assert len(response.citations) == 1
    assert "Model Context Protocol" in response.answer

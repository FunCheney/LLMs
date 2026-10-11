from retrieval.rrf import reciprocal_rank_fusion
from retrieval.types import RetrievalResult
from schemas import DocumentChunk


def _hit(chunk_id: str, source: str = "dense") -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc",
        document_title="Doc",
        section_title=chunk_id,
        section_path=[chunk_id],
        page_start=1,
        page_end=1,
        content=f"content-{chunk_id}",
        source_uri="doc.pdf",
        source_type="pdf",
    )
    return RetrievalResult(
        chunk_id=chunk_id,
        text=chunk.content,
        metadata=chunk.to_dict(),
        score=0.0,
        source=source,
        chunk=chunk,
    )


def test_rrf_prefers_chunks_ranked_high_in_both_lists():
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("a"), _hit("c"), _hit("d")]

    fused = reciprocal_rank_fusion([dense, sparse], top_k=2)

    assert [hit.chunk_id for hit in fused] == ["a", "c"]
    assert fused[0].source == "hybrid"
    assert fused[0].rank == 1
    assert fused[0].score > fused[1].score

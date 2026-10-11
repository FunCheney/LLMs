from pathlib import Path

from embedding import build_bm25_index
from ingestion.storage import save_chunks_jsonl
from retrieval.spare_retriever import SparseRetriever
from schemas import DocumentChunk


def _sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="doc-p001-p001-c000",
            document_id="doc",
            document_title="Doc",
            section_title="Kafka",
            section_path=["Kafka"],
            page_start=1,
            page_end=1,
            content="Kafka consumer offset commit failed.",
            source_uri="data/raw/doc.pdf",
            source_type="pdf",
        ),
        DocumentChunk(
            chunk_id="doc-p002-p002-c001",
            document_id="doc",
            document_title="Doc",
            section_title="Weather",
            section_path=["Weather"],
            page_start=2,
            page_end=2,
            content="Today the weather is sunny and warm.",
            source_uri="data/raw/doc.pdf",
            source_type="pdf",
        ),
    ]


def test_sparse_retriever_returns_keyword_match(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "bm25.pkl"
    save_chunks_jsonl(_sample_chunks(), chunks_path)
    build_bm25_index(chunks_path=chunks_path, index_path=index_path)

    retriever = SparseRetriever(index_path=index_path)
    hits = retriever.retrieve("Kafka offset", top_k=1)

    assert len(hits) == 1
    assert hits[0].source == "sparse"
    assert hits[0].chunk_id == "doc-p001-p001-c000"

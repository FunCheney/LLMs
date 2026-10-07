from pathlib import Path

from embedding.fake import FakeEmbeddingProvider
from indexing.vector_index import build_local_index, search_local_index
from ingestion.storage import save_chunks_jsonl
from schemas import DocumentChunk


def _sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="doc-p001-p001-c000",
            document_id="doc",
            document_title="Doc",
            section_title="Python",
            section_path=["Python"],
            page_start=1,
            page_end=1,
            content="Python can read PDF files with PyMuPDF.",
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


def test_build_and_search_local_index(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "index.pkl"
    save_chunks_jsonl(_sample_chunks(), chunks_path)

    provider = FakeEmbeddingProvider(dimension=4)
    index = build_local_index(
        chunks_path=chunks_path,
        index_path=index_path,
        embedder=provider,
        batch_size=2,
    )

    assert index.count() == 2
    assert index_path.exists()

    results = search_local_index(
        index_path=index_path,
        query="Python PDF",
        embedder=provider,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0].id == "doc-p001-p001-c000"

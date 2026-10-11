from pathlib import Path

from embedding.fake import FakeEmbeddingProvider
from indexing.vector_index import build_local_index
from ingestion.storage import save_chunks_jsonl
from retrieval.retriever import DenseRetriever
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


def test_dense_retriever_returns_retrieval_result(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "index.pkl"
    save_chunks_jsonl(_sample_chunks(), chunks_path)

    provider = FakeEmbeddingProvider(dimension=4)
    build_local_index(
        chunks_path=chunks_path,
        index_path=index_path,
        embedder=provider,
        batch_size=2,
    )

    retriever = DenseRetriever(index_path=index_path, embedder=provider)
    hits = retriever.retrieve("Python PDF", top_k=1)

    assert len(hits) == 1
    assert hits[0].source == "dense"
    assert hits[0].rank == 1
    assert hits[0].chunk_id == "doc-p001-p001-c000"
    assert hits[0].chunk.section_title == "Python"
    assert hits[0].text.startswith("Python can read")

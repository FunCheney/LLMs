from pathlib import Path

from embedding.fake import FakeEmbeddingProvider
from indexing.bm25_index import build_bm25_index
from indexing.vector_index import build_local_index
from ingestion.storage import save_chunks_jsonl
from retrieval.factory import RetrieverConfig, create_retriever
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
            section_title="Kafka",
            section_path=["Kafka"],
            page_start=2,
            page_end=2,
            content="Kafka consumer offset commit failed.",
            source_uri="data/raw/doc.pdf",
            source_type="pdf",
        ),
    ]


def test_hybrid_retriever_uses_rrf(tmp_path: Path, monkeypatch):
    chunks_path = tmp_path / "chunks.jsonl"
    dense_index_path = tmp_path / "dense.pkl"
    bm25_index_path = tmp_path / "bm25.pkl"
    save_chunks_jsonl(_sample_chunks(), chunks_path)

    provider = FakeEmbeddingProvider(dimension=4)
    build_local_index(
        chunks_path=chunks_path,
        index_path=dense_index_path,
        embedder=provider,
        batch_size=2,
    )
    build_bm25_index(chunks_path=chunks_path, index_path=bm25_index_path)

    def fake_embedder(device=None):
        return provider

    monkeypatch.setattr(
        "retrieval.factory.BGEM3EmbeddingProvider",
        fake_embedder,
    )

    retriever = create_retriever(
        RetrieverConfig(
            mode="hybrid",
            dense_index_path=dense_index_path,
            bm25_index_path=bm25_index_path,
        )
    )
    hits = retriever.retrieve("Kafka offset", top_k=2)

    assert len(hits) == 2
    assert all(hit.source == "hybrid" for hit in hits)
    assert hits[0].chunk_id == "doc-p002-p002-c001"

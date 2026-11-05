from pathlib import Path

from rag_mvp.embedding import build_bm25_index
from rag_mvp.embedding.fake import FakeEmbeddingProvider
from rag_mvp.indexing.vector_index import build_local_index
from rag_mvp.ingestion.storage import save_chunks_jsonl
from rag_mvp.retrieval.factory import RetrieverConfig, create_retriever
from rag_mvp.schemas import DocumentChunk


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

    # patch 源模块：factory 里是函数内 `from rag_mvp.embedding import ...`，
    # 而 rag_mvp.embedding 的该符号由 __getattr__ 惰性导出，模块级并不存在。
    # 原写法 patch "retrieval.factory.BGEM3EmbeddingProvider" 会直接
    # AttributeError，等于这条测试从未真正验证过 RRF。
    monkeypatch.setattr(
        "rag_mvp.embedding.BGEM3EmbeddingProvider",
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

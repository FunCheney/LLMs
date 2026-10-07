from pathlib import Path
from typing import Any

from embedding.base import EmbeddingProvider
from ingestion.storage import load_chunks_jsonl
from vector_store.local import LocalVectorIndex
from vector_store.vector_store_types import VectorRecord

"""
chunks.jsonl
    ↓  load_chunks_jsonl
src/indexing/vector_index.py  →  build_local_index()
    ↓  embed_documents (分批)
LocalVectorIndex  →  data/index/local.pkl

search_cli.py  →  search_local_index()  →  打印 Top-K + metadata
"""

def chunk_to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "document_title": chunk["document_title"],
        "section_title": chunk.get("section_title", ""),
        "section_path": chunk.get("section_path") or [],
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "source_uri": chunk.get("source_uri", ""),
        "source_type": chunk.get("source_type", ""),
    }


def build_local_index(
    chunks_path: str | Path,
    index_path: str | Path,
    embedder: EmbeddingProvider,
    batch_size: int = 32,
) -> LocalVectorIndex:
    """
        1. 读 jsonl
        2. 按 batch_size 调用 embed_documents
        3. 每条写成 VectorRecord(id=chunk_id, text=content, metadata=..., embedding=...)
        4. index.save()，并自动创建 data/index/ 目录
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    index = LocalVectorIndex()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [item["content"] for item in batch]
        embeddings = embedder.embed_documents(texts)

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)}, expected {len(batch)}"
            )

        for chunk, embedding in zip(batch, embeddings):
            index.add(
                VectorRecord(
                    id=chunk["chunk_id"],
                    text=chunk["content"],
                    metadata=chunk_to_metadata(chunk),
                    embedding=embedding.vector,
                )
            )

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    return index


def search_local_index(
    index_path: str | Path,
    query: str,
    embedder: EmbeddingProvider,
    top_k: int = 5,
) -> list[tuple[VectorRecord, float]]:
    """
    LocalVectorIndex.load
    embed_text(query)
    点积排序（向量已 L2 归一化，等价 cosine）
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    index = LocalVectorIndex.load(index_path)
    if index.count() == 0:
        return []

    query_vector = embedder.embed_text(query).vector
    return index.search(query_vector, top_k=top_k)

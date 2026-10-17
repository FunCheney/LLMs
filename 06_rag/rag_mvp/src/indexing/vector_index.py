"""
本地向量索引的构建与检索（Phase 2 的入口层）。

数据流：

    chunks.jsonl
        ↓  load_chunks_jsonl
    src/indexing/vector_index.py  →  build_local_index()
        ↓  embed_documents（分批）
    LocalVectorIndex  →  data/index/local.pkl

    search_cli.py  →  search_local_index()  →  打印 Top-K + metadata

脚本 scripts/build_vector_index.py 负责调用 build_local_index，
把「读 jsonl → 批量向量化 → 落盘」串起来。
"""

from pathlib import Path

from embedding.base import EmbeddingProvider
from indexing.chunk_metadata import chunk_to_metadata
from ingestion.storage import load_chunks_jsonl
from vector_store.local import LocalVectorIndex
from vector_store.milvus import MilvusVectorOpt
from vector_store.vector_store_types import VectorRecord


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
    # 分批向量化：控制显存/内存占用，也便于观察进度
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [item["content"] for item in batch]
        embeddings = embedder.embed_documents(texts)

        # 防御性校验：向量数量必须与文本数量一致，否则记录会错位
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
    """兼容旧调用；新代码请使用 retrieval.DenseRetriever。"""
    from retrieval.retriever import DenseRetriever

    retriever = DenseRetriever(index_path=index_path, embedder=embedder)
    hits = retriever.retrieve(query, top_k=top_k)
    return [
        (
            VectorRecord(
                id=hit.chunk_id,
                text=hit.text,
                metadata=hit.metadata,
                embedding=[],
            ),
            hit.score,
        )
        for hit in hits
    ]


def build_vector_db_index(
    chunks_path: str | Path,
    embedder: EmbeddingProvider,
    dimension: int,
    batch_size: int = 32,
) -> MilvusVectorOpt:
    """
        1. 读 jsonl
        2. 按 batch_size 调用 embed_documents
        3. 每条写成 VectorRecord(id=chunk_id, text=content, metadata=..., embedding=...)
        4. index.save()， 写入到 vector_db
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    vector_opt = MilvusVectorOpt(
        milvus_uri="http://localhost:19530",
        token="root:Milvus",
        collection_name="vector_collection",
        embedding_dim=dimension,
    )


    # 分批向量化：控制显存/内存占用，也便于观察进度
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [item["content"] for item in batch]
        embeddings = embedder.embed_documents(texts)

        # 防御性校验：向量数量必须与文本数量一致，否则记录会错位
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)}, expected {len(batch)}"
            )

        for chunk, embedding in zip(batch, embeddings):
            print(f"insert: {chunk['chunk_id']}: {chunk['content']}")
            vector_opt.add(
                VectorRecord(
                    id=chunk["chunk_id"],
                    text=chunk["content"],
                    metadata=chunk_to_metadata(chunk),
                    embedding=embedding.vector,
                )
            )

    vector_opt.save(path='')

    return vector_opt

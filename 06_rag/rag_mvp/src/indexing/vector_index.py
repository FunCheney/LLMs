"""
向量索引的构建与检索（Phase 2 的入口层）。

数据流：

    chunks.jsonl
        ↓  load_chunks_jsonl
    build_index(chunks_path, VectorStoreConfig, embedder)
        ↓  embed_documents（分批）→ VectorRecord → writer.add()
    writer.save()  →  local: data/index/local.pkl  /  milvus: collection upsert

    search_cli.py  →  DenseRetriever(store=open_reader(cfg))  →  Top-K + metadata

脚本 scripts/build_vector_index.py 负责把「读 jsonl → 批量向量化 → 持久化」串起来，
后端由 --store-type local|milvus 决定。

build_local_index / build_vector_db_index / search_local_index 为兼容旧调用的包装，
新代码请直接使用 build_index + VectorStoreConfig。
"""

from pathlib import Path

from embedding.base import EmbeddingProvider
from indexing.chunk_metadata import chunk_to_metadata
from ingestion.storage import load_chunks_jsonl
from vector_store.base import VectorStore
from vector_store.config import VectorStoreConfig
from vector_store.vector_store_types import VectorRecord


def build_index(
    chunks_path: str | Path,
    config: VectorStoreConfig,
    embedder: EmbeddingProvider,
    batch_size: int = 32,
) -> VectorStore:
    """
    通用索引构建：读 jsonl → 分批向量化 → 写入 config 指定的后端。

    1. 读 jsonl
    2. 按 batch_size 调用 embed_documents
    3. 每条写成 VectorRecord(id=chunk_id, text=content, metadata=..., embedding=...)
    4. writer.save()：local 落盘 config.index_path；milvus upsert 进 collection（幂等）
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    from vector_store.factory import create_writer

    writer = create_writer(config)
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
            writer.add(
                VectorRecord(
                    id=chunk["chunk_id"],
                    text=chunk["content"],
                    metadata=chunk_to_metadata(chunk),
                    embedding=embedding.vector,
                )
            )

    writer.save(config.index_path if config.store_type == "local" else "")
    return writer  # type: ignore[return-value]


def build_local_index(
    chunks_path: str | Path,
    index_path: str | Path,
    embedder: EmbeddingProvider,
    batch_size: int = 32,
):
    """兼容旧调用；等价于 build_index(..., VectorStoreConfig(store_type="local"))。"""
    return build_index(
        chunks_path=chunks_path,
        config=VectorStoreConfig(store_type="local", index_path=Path(index_path)),
        embedder=embedder,
        batch_size=batch_size,
    )


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
    milvus_uri: str | None = None,
    token: str | None = None,
    collection_name: str | None = None,
    drop_existing: bool = False,
):
    """
    兼容旧调用；等价于 build_index(..., VectorStoreConfig(store_type="milvus"))。
    连接参数未传时走 VectorStoreConfig 的 env 默认值（MILVUS_URI 等），
    不再硬编码 localhost:19530。
    """
    kwargs: dict = {"store_type": "milvus", "dimension": dimension, "drop_existing": drop_existing}
    if milvus_uri is not None:
        kwargs["uri"] = milvus_uri
    if token is not None:
        kwargs["token"] = token
    if collection_name is not None:
        kwargs["collection"] = collection_name
    return build_index(
        chunks_path=chunks_path,
        config=VectorStoreConfig(**kwargs),
        embedder=embedder,
        batch_size=batch_size,
    )

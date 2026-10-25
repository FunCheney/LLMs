"""
CLI：构建稠密向量索引（local pickle 或 Milvus，由 --store-type 决定）。

用法：
    uv run python scripts/build_vector_index.py                          # local（默认）
    uv run python scripts/build_vector_index.py --store-type milvus      # Milvus 服务
    uv run python scripts/build_vector_index.py --store-type milvus \
        --milvus-uri data/index/milvus.db --drop                         # Milvus Lite 重建

首次运行会从 HuggingFace（已切到 hf-mirror 镜像）下载 BGE-M3，耗时较长。
Milvus 连接参数优先级：CLI 参数 > 环境变量（MILVUS_URI / MILVUS_TOKEN / MILVUS_COLLECTION）。
"""

import argparse
from pathlib import Path

from embedding import BGEM3EmbeddingProvider
from indexing.vector_index import build_index
from vector_store.config import VectorStoreConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dense vector index from chunks.jsonl (local or milvus)",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Path to chunks.jsonl produced by build_chunks.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/index/local.pkl"),
        help="Path to save the local vector index (store_type=local)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for SentenceTransformer, e.g. cpu or mps",
    )
    parser.add_argument(
        "--store-type",
        "--index-type",  # 兼容旧参数名
        dest="store_type",
        choices=["local", "milvus"],
        default="local",
        help="Vector store backend (default: local)",
    )
    parser.add_argument("--milvus-uri", default=None, help="Milvus URI（默认 env MILVUS_URI 或 http://localhost:19530；Lite 传 .db 文件路径）")
    parser.add_argument("--milvus-token", default=None, help="Milvus token（默认 env MILVUS_TOKEN 或 root:Milvus）")
    parser.add_argument("--collection", default=None, help="Milvus collection 名（默认 env MILVUS_COLLECTION 或 rag_mvp_chunks）")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="构建前 drop 已有 collection（仅 milvus；local 本身就是全量覆盖）",
    )
    return parser.parse_args()


def main() -> None:
    """
    读 data/processed/chunks.jsonl
        → BGEM3EmbeddingProvider.embed_documents（分批，如 batch=32）
        → 每条 VectorRecord(id=chunk_id, text=content, metadata=章节/页码/source_uri, embedding=...)
        → local: data/index/local.pkl  /  milvus: collection upsert（幂等）
    """
    args = parse_args()
    embedder = BGEM3EmbeddingProvider(device=args.device)

    kwargs: dict = {
        "store_type": args.store_type,
        "index_path": args.output,
        "dimension": embedder.DIMENSION,
        "drop_existing": args.drop,
    }
    if args.milvus_uri is not None:
        kwargs["uri"] = args.milvus_uri
    if args.milvus_token is not None:
        kwargs["token"] = args.milvus_token
    if args.collection is not None:
        kwargs["collection"] = args.collection

    # 根据参数初始化向量存储配置
    config = VectorStoreConfig(**kwargs)

    index = build_index(
        chunks_path=args.chunks,
        config=config,
        embedder=embedder,
        batch_size=args.batch_size,
    )

    print(f"Chunks file: {args.chunks}")
    if args.store_type == "local":
        print(f"Index file: {args.output}")
    else:
        print(f"Milvus collection: {config.collection} @ {config.uri}")
    print(f"Indexed records: {index.count()}")
    print(f"Embedding model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.DIMENSION}")


if __name__ == "__main__":
    main()

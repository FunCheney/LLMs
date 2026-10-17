"""
CLI：构建本地稠密向量索引（data/index/local.pkl）。

用法：
    uv run python scripts/build_vector_index.py
    uv run python scripts/build_vector_index.py --device cpu --batch-size 32

首次运行会从 HuggingFace（已切到 hf-mirror 镜像）下载 BGE-M3，耗时较长。
"""

import argparse
from pathlib import Path

from embedding import BGEM3EmbeddingProvider
from indexing.vector_index import build_local_index, build_vector_db_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local vector index from chunks.jsonl",
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
        help="Path to save the local vector index",
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
        "--index-type",
        default="vector",
        help="give index type (milvus or local)",
    )
    return parser.parse_args()


def main() -> None:
    """
    读 data/processed/chunks.jsonl
        → BGEM3EmbeddingProvider.embed_documents（分批，如 batch=32）
        → 每条 VectorRecord(id=chunk_id, text=content, metadata=章节/页码/source_uri, embedding=...)
        → LocalVectorIndex.save("data/index/local.pkl")
    """
    args = parse_args()
    embedder = BGEM3EmbeddingProvider(device=args.device)

    if args.index_type == "milvus":
        index = build_vector_db_index(
            chunks_path=args.chunks,
            embedder=embedder,
            dimension=1024,
            batch_size=args.batch_size,
        )
    else:
        index = build_local_index(
            chunks_path=args.chunks,
            index_path=args.output,
            embedder=embedder,
            batch_size=args.batch_size,
        )

    print(f"Chunks file: {args.chunks}")
    print(f"Index file: {args.output}")
    print(f"Indexed records: {index.count()}")
    print(f"Embedding model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.DIMENSION}")


if __name__ == "__main__":
    main()

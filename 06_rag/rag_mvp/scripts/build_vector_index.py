import argparse
from pathlib import Path

from embedding import BGEM3EmbeddingProvider
from indexing.vector_index import build_local_index


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

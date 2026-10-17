"""
CLI：构建 BM25 稀疏索引（data/index/bm25.pkl）。

用法：
    uv run python scripts/build_bm25_index.py

    uv run python scripts/build_bm25_index.py \
        --chunks data/processed/chunks_test.jsonl \
        --output data/index/bm25_test.pkl

基于 jieba 分词，用于关键词检索；与向量索引一起支撑 hybrid 模式。
"""

import argparse
from pathlib import Path

from embedding import build_bm25_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BM25 index from chunks.jsonl",
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
        default=Path("data/index/bm25.pkl"),
        help="Path to save the BM25 index",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_bm25_index(
        chunks_path=args.chunks,
        index_path=args.output,
    )
    print(f"Chunks file: {args.chunks}")
    print(f"Index file: {args.output}")
    print(f"Indexed records: {index.count()}")


if __name__ == "__main__":
    main()

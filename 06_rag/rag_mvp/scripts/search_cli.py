"""
CLI：检索调试工具，支持 dense / sparse / hybrid 三种模式。

用法：
    uv run python scripts/search_cli.py "MCP 是什么" --mode hybrid --k 5
    uv run python scripts/search_cli.py "CLAUDE.md" --mode sparse --k 3

只做「检索 + 打印命中」，不调用 LLM，是验证索引与召回质量最常用的手段。
"""

import argparse
from pathlib import Path

from retrieval.display import print_hits
from retrieval.factory import RetrieverConfig, create_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search chunks with dense, sparse, or hybrid retrieval",
    )
    parser.add_argument("query", help="User question")
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid"],
        default="dense",
        help="Retriever mode",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=Path("data/index/local.pkl"),
        help="Path to the dense vector index",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=Path("data/index/bm25.pkl"),
        help="Path to the BM25 index",
    )
    parser.add_argument(
        "--snippet",
        type=int,
        default=200,
        help="Max characters of chunk content to print",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for SentenceTransformer, e.g. cpu or mps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RetrieverConfig(
        mode=args.mode,
        dense_index_path=args.dense_index,
        bm25_index_path=args.bm25_index,
        device=args.device,
    )

    # 提前检查所需索引是否存在，给出「先跑哪个脚本」的提示，避免晦涩报错
    if args.mode in {"dense", "hybrid"} and not args.dense_index.exists():
        raise SystemExit(
            f"Dense index not found: {args.dense_index}\n"
            "Run: uv run python scripts/build_vector_index.py"
        )
    if args.mode in {"sparse", "hybrid"} and not args.bm25_index.exists():
        raise SystemExit(
            f"BM25 index not found: {args.bm25_index}\n"
            "Run: uv run python scripts/build_bm25_index.py"
        )

    retriever = create_retriever(config)
    hits = retriever.retrieve(args.query, top_k=args.k)

    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"Results: {len(hits)}\n")
    print_hits(hits, snippet=args.snippet)


if __name__ == "__main__":
    main()

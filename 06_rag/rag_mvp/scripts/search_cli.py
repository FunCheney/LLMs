import argparse
from pathlib import Path

from embedding import BGEM3EmbeddingProvider
from indexing.vector_index import search_local_index

"""
目标： 终端里问一句话，看到 Top-K 和 metadata。
如：
    uv run python scripts/search_cli.py "MCP 是什么" --k 5
输出至少包含：rank、score、chunk_id、section_title、section_path、page_start/end、content 前 200 字。
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the local vector index",
    )
    parser.add_argument("query", help="User question")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/index/local.pkl"),
        help="Path to the local vector index",
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


def format_path(section_path: list[str]) -> str:
    if not section_path:
        return "-"
    return " > ".join(section_path)


def main() -> None:
    args = parse_args()
    if not args.index.exists():
        raise SystemExit(
            f"Index not found: {args.index}\n"
            "Run: uv run python scripts/build_vector_index.py"
        )

    embedder = BGEM3EmbeddingProvider(device=args.device)
    results = search_local_index(
        index_path=args.index,
        query=args.query,
        embedder=embedder,
        top_k=args.k,
    )

    print(f"Query: {args.query}")
    print(f"Index: {args.index}")
    print(f"Results: {len(results)}\n")

    for rank, (record, score) in enumerate(results, start=1):
        metadata = record.metadata
        content = record.text.replace("\n", " ")
        snippet = content[: args.snippet]
        if len(content) > args.snippet:
            snippet += "..."

        print("=" * 72)
        print(f"#{rank}  score={score:.4f}")
        print(f"chunk_id: {metadata.get('chunk_id', record.id)}")
        print(f"section: {metadata.get('section_title', '-')}")
        print(f"path: {format_path(metadata.get('section_path') or [])}")
        print(
            f"pages: {metadata.get('page_start')}-{metadata.get('page_end')}  "
            f"source: {metadata.get('source_uri', '-')}"
        )
        print(f"content: {snippet}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from generation.display import print_rag_response
from generation.pipeline import answer_question
from retrieval.factory import RetrieverConfig, create_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Q&A: retrieve with hybrid (default) then generate an answer with citations",
    )
    parser.add_argument("question", help="User question")
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retriever mode (default: hybrid)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=Path("data/index/local.pkl"),
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=Path("data/index/bm25.pkl"),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for embedding model",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: DEEPSEEK_MODEL or deepseek-chat)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only retrieve and print context, do not call the LLM",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to save RAGResponse as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    retriever = create_retriever(
        RetrieverConfig(
            mode=args.mode,
            dense_index_path=args.dense_index,
            bm25_index_path=args.bm25_index,
            device=args.device,
        )
    )

    response = answer_question(
        query=args.question,
        retriever=retriever,
        retrieval_mode=args.mode,
        top_k=args.k,
        dry_run=args.dry_run,
        model=args.model,
        temperature=args.temperature,
    )

    print_rag_response(response)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(response.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON saved to: {args.json}")


if __name__ == "__main__":
    main()

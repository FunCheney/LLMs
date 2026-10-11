import argparse
import sys
from pathlib import Path

from eval.smoke_runner import (
    format_console_report,
    run_smoke_evaluation,
    save_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run smoke retrieval evaluation with section-based matching",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/smoke_queries.yaml"),
        help="Path to smoke query YAML",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated modes: dense,sparse,hybrid",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override top_k from dataset defaults",
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
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/eval/reports/smoke_latest.json"),
        help="Where to write JSON report",
    )
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        default=True,
        help="Exit with code 1 if any query fails (default: true)",
    )
    parser.add_argument(
        "--no-fail-on-miss",
        action="store_false",
        dest="fail_on_miss",
        help="Always exit 0 even when queries fail",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = None
    if args.modes:
        modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    if args.dense_index.exists() is False and modes != ["sparse"]:
        if modes is None or "dense" in modes or "hybrid" in modes:
            raise SystemExit(
                f"Dense index not found: {args.dense_index}\n"
                "Run: uv run python scripts/build_vector_index.py"
            )
    if args.bm25_index.exists() is False:
        if modes is None or "sparse" in modes or "hybrid" in modes:
            raise SystemExit(
                f"BM25 index not found: {args.bm25_index}\n"
                "Run: uv run python scripts/build_bm25_index.py"
            )

    report, warnings = run_smoke_evaluation(
        dataset_path=args.dataset,
        modes=modes,
        top_k=args.k,
        dense_index_path=args.dense_index,
        bm25_index_path=args.bm25_index,
        device=args.device,
    )

    for warning in warnings:
        print(f"WARNING: {warning}")

    print(format_console_report(report))
    save_report(report, args.report)
    print(f"\nReport saved to: {args.report}")

    if args.fail_on_miss and not report.all_passed():
        sys.exit(1)


if __name__ == "__main__":
    main()

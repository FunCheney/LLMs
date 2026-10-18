"""
CLI：运行 Golden 检索评测（Phase 4）。

用法：
    uv run python scripts/run_golden_eval.py
    uv run python scripts/run_golden_eval.py --modes dense,hybrid --k 10

默认数据集：data/eval/golden_queries.yaml；报告写入 data/eval/reports/golden_latest.json。
默认若存在 Recall@10 未命中的用例会以退出码 1 结束（可用 --no-fail-on-miss 关闭），
方便接入 CI 做检索质量回归。
"""

import argparse
import sys
from pathlib import Path

from eval.golden_runner import (
    format_console_report,
    run_golden_evaluation,
    save_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run golden retrieval evaluation (Recall@K / MRR / nDCG)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/golden_queries.yaml"),
        help="Path to golden query YAML",
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
        help="Top-K to retrieve (default: dataset top_k, at least 10)",
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
        "--rerank",
        action="store_true",
        help="Additionally evaluate each mode with bge-reranker (报告键 <mode>+rerank)",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Custom reranker model (default: BAAI/bge-reranker-v2-m3)",
    )
    parser.add_argument(
        "--store-type",
        choices=["local", "milvus"],
        default="local",
        help="Dense vector store backend (default: local；milvus 用于双后端对齐验证)",
    )
    parser.add_argument("--milvus-uri", default=None, help="Milvus URI（Lite 传 .db 文件路径）")
    parser.add_argument("--milvus-token", default=None, help="Milvus token")
    parser.add_argument("--collection", default=None, help="Milvus collection 名")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/eval/reports/golden_latest.json"),
        help="Where to write JSON report",
    )
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        default=True,
        help="Exit with code 1 if any Recall@10 miss exists (default: true)",
    )
    parser.add_argument(
        "--no-fail-on-miss",
        action="store_false",
        dest="fail_on_miss",
        help="Always exit 0 even when queries miss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = None
    if args.modes:
        modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    # 按所选模式检查依赖的索引是否就绪（milvus 后端跳过本地文件检查）
    if args.store_type == "local" and args.dense_index.exists() is False and modes != ["sparse"]:
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

    report, warnings = run_golden_evaluation(
        dataset_path=args.dataset,
        modes=modes,
        top_k=args.k,
        dense_index_path=args.dense_index,
        bm25_index_path=args.bm25_index,
        device=args.device,
        use_reranker=args.rerank,
        rerank_model=args.rerank_model,
        store_type=args.store_type,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
        milvus_collection=args.collection,
    )

    for warning in warnings:
        print(f"WARNING: {warning}")

    print(format_console_report(report))
    save_report(report, args.report)
    print(f"\nReport saved to: {args.report}")

    # 存在 Recall@10 未命中的用例时以非零退出码结束，便于 CI 拦截
    if args.fail_on_miss:
        has_miss = any(
            any(not r.hits_at_k.get("10", False) for r in mode_report.results)
            for mode_report in report.modes.values()
        )
        if has_miss:
            sys.exit(1)


if __name__ == "__main__":
    main()

"""
CLI：运行 smoke 检索评测（Phase 4 早期形式）。

用法：
    uv run python scripts/run_smoke_retrieval.py
    uv run python scripts/run_smoke_retrieval.py --modes hybrid --k 5

默认数据集：data/eval/smoke_queries.yaml；报告写入 data/eval/reports/smoke_latest.json。
默认若存在未通过的用例会以退出码 1 结束（可用 --no-fail-on-miss 关闭），
方便接入 CI 做检索质量回归。
"""

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
    # 解析逗号分隔的模式列表，忽略空项
    modes = None
    if args.modes:
        modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    # 按所选模式检查依赖的索引是否就绪（只用 sparse 时无需向量索引）
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

    # 先打印语料一致性等告警（如 chunk 数与数据集声明不符）
    for warning in warnings:
        print(f"WARNING: {warning}")

    print(format_console_report(report))
    save_report(report, args.report)
    print(f"\nReport saved to: {args.report}")

    # 有失败用例时以非零退出码结束，便于 CI 拦截
    if args.fail_on_miss and not report.all_passed():
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
CLI：把原始文档解析并分块，产出 data/processed/chunks.jsonl。

用法：
    uv run python scripts/build_chunks.py

这是 Phase 1 的入口脚本：内部调用 ingest_document 完成
「解析 → 结构化分块 → 质量校验」，再落盘为 JSONL 并打印统计与问题。
"""

from pathlib import Path

from ingestion.pipeline import ingest_document
from ingestion.storage import save_chunks_jsonl


def main():
    pdf_path = Path("data/raw/《证券分析》本杰明•格雷厄姆.pdf")
    output_path = Path("data/processed/chunks_test.jsonl")

    # 解析 + 分块 + 校验（一条龙）
    chunks, report = ingest_document(
        path=pdf_path,
        document_id="证券分析-0001",
        document_title="证券分析",
        chunk_size=500,
        overlap=50,
    )
    save_chunks_jsonl(chunks, output_path)

    print(f"Saved {len(chunks)} chunks to: {output_path}")
    print("stats:", report.stats)
    # 打印致命错误（如有）
    if report.errors:
        print("errors:")
        for issue in report.errors:
            print(f"  [{issue.code}] {issue.message} {issue.chunk_id or ''}")
    # 打印告警（最多 20 条，其余折叠）
    if report.warnings:
        print("warnings:")
        for issue in report.warnings[:20]:
            print(f"  [{issue.code}] {issue.message} {issue.chunk_id or ''}")
        if len(report.warnings) > 20:
            print(f"  ... {len(report.warnings) - 20} more")


if __name__ == "__main__":
    main()

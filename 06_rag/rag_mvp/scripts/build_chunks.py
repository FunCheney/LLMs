from pathlib import Path

from ingestion.pipeline import ingest_document
from ingestion.storage import save_chunks_jsonl


def main():
    pdf_path = Path("data/raw/Claude Code设计指南.pdf")
    output_path = Path("data/processed/chunks.jsonl")

    chunks, report = ingest_document(
        path=pdf_path,
        document_id="claude-code-0001",
        document_title="Claude Code 设计指南",
        chunk_size=500,
        overlap=50,
    )
    save_chunks_jsonl(chunks, output_path)

    print(f"Saved {len(chunks)} chunks to: {output_path}")
    print("stats:", report.stats)
    if report.errors:
        print("errors:")
        for issue in report.errors:
            print(f"  [{issue.code}] {issue.message} {issue.chunk_id or ''}")
    if report.warnings:
        print("warnings:")
        for issue in report.warnings[:20]:
            print(f"  [{issue.code}] {issue.message} {issue.chunk_id or ''}")
        if len(report.warnings) > 20:
            print(f"  ... {len(report.warnings) - 20} more")


if __name__ == "__main__":
    main()

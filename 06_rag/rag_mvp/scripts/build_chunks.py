from pathlib import Path

from ingestion.chunker import build_chunks
from ingestion.pdf_parser import parse_pdf
from ingestion.storage import save_chunks_jsonl


def main():
    pdf_path = Path("data/raw/Claude Code设计指南.pdf")
    document_id = "claude-Code-0001"
    document_title = "claude-Code Documentation"
    output_path = Path("data/processed/chunks.jsonl")

    #1.解析pdf
    pages = parse_pdf(pdf_path)

    #2.页面转换为 chunks
    chunks = build_chunks(pages, document_id, document_title, chunk_size=500, overlap=50)

    # 3. 保存 jsonl
    save_chunks_jsonl(chunks, output_path)

    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
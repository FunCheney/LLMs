"""
ingestion 流水线编排：把「文件路径」一路加工成「可检索的 Chunk + 质量报告」。

一条龙步骤：
    load_pages（PDF/HTML → 逐页文本）
        → build_chunks（结构化分块，带章节/页码/来源）
        → validate_chunks（质量校验，产出 ValidationReport）

这是 Phase 1（文档处理）的总入口，脚本 scripts/build_chunks.py 直接调用它。
"""

from pathlib import Path

from ingestion.chunker import build_chunks
from ingestion.document_loader import load_pages
from schemas import DocumentChunk
from validation.chunk_validator import ValidationReport, validate_chunks


def ingest_document(
    path: str | Path,
    document_id: str,
    document_title: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> tuple[list[DocumentChunk], ValidationReport]:
    """
    解析并切分单个文档，返回 (chunks, 校验报告)。

    注意：这里只做「生成 + 校验」，不负责落盘；
    需要写文件的调用方请自行使用 save_chunks_jsonl。
    """
    path = Path(path)
    pages, source_type = load_pages(path)
    chunks = build_chunks(
        pages=pages,
        document_id=document_id,
        document_title=document_title,
        chunk_size=chunk_size,
        overlap=overlap,
        source_uri=str(path),
        source_type=source_type,
    )
    # 校验文本
    report = validate_chunks(chunks, chunk_size=chunk_size)
    return chunks, report

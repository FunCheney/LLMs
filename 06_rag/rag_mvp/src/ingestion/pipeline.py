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
    report = validate_chunks(chunks, chunk_size=chunk_size)
    return chunks, report

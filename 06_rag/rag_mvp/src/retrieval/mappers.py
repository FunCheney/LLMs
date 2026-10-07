from typing import Any

from retrieval.types import RetrievalResult
from schemas import DocumentChunk
from vector_store.vector_store_types import VectorRecord


def metadata_to_document_chunk(metadata: dict[str, Any], content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=metadata.get("chunk_id", ""),
        document_id=metadata.get("document_id", ""),
        document_title=metadata.get("document_title", ""),
        section_title=metadata.get("section_title", ""),
        section_path=list(metadata.get("section_path") or []),
        page_start=int(metadata.get("page_start") or 0),
        page_end=int(metadata.get("page_end") or 0),
        content=content,
        source_uri=metadata.get("source_uri", ""),
        source_type=metadata.get("source_type", ""),
    )


def vector_record_to_hit(
    record: VectorRecord,
    score: float,
    source: str,
) -> RetrievalResult:


    metadata = dict(record.metadata)
    chunk_id = metadata.get("chunk_id") or record.id
    chunk = metadata_to_document_chunk(metadata, record.text)

    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        source=source,
        chunk=chunk,
    )
